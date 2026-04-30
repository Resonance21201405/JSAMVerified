"""
ingestion.py — Person 1 (Data Engineer)
=========================================
Parses the BIS SP 21 PDF into structured per-IS-standard chunks.

Output: data/chunks.json
Each chunk looks like:
{
    "is_number": "IS 269: 1989",
    "title": "ORDINARY PORTLAND CEMENT, 33 GRADE",
    "year": "1989",
    "section": 1,
    "category": "Cement and Concrete",
    "sub_category": "Cement",
    "content": "<full text of the standard summary>",
    "scope": "<first sentence of the scope section if parseable>"
}

Usage:
    python src/ingestion.py --pdf data/SP21.pdf --output data/chunks.json
    python src/ingestion.py  # uses default paths
"""

import re
import json
import argparse
import logging
from pathlib import Path

import pdfplumber

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PDF = Path(__file__).parent.parent / "data" / "dataset.pdf"
DEFAULT_OUT = Path(__file__).parent.parent / "data" / "chunks.json"

# Section number → human-readable category name
SECTION_CATEGORY = {
    1:  "Cement and Concrete",
    2:  "Building Limes",
    3:  "Stones",
    4:  "Wood Products for Building",
    5:  "Gypsum Building Materials",
    6:  "Timber",
    7:  "Bitumen and Tar Products",
    8:  "Floor, Wall, Roof Coverings and Finishes",
    9:  "Water Proofing and Damp Proofing Materials",
    10: "Sanitary Appliances and Water Fittings",
    11: "Builder's Hardware",
    12: "Wood Products",
    13: "Doors, Windows and Shutters",
    14: "Concrete Reinforcement",
    15: "Structural Steels",
    16: "Light Metal and Their Alloys",
    17: "Structural Shapes",
    18: "Welding Electrodes and Wires",
    19: "Threaded Fasteners and Rivets",
    20: "Wire Ropes and Wire Products",
    21: "Glass",
    22: "Fillers, Stoppers and Putties",
    23: "Thermal Insulation Materials",
    24: "Plastics",
    25: "Conductors and Cables",
    26: "Wiring Accessories",
    27: "General",
}

# Sub-category keywords found in section ToC lines (Section 1 only)
CEMENT_SUBCATEGORIES = {
    "AGGREGATES":          "Aggregates",
    "CEMENT MATRIX":       "Cement Matrix Products",
    "ASBESTOS CEMENT":     "Asbestos Cement Products",
    "CONCRETE PIPES":      "Concrete Pipes",
    "CONCRETE MASONRY":    "Concrete Masonry",
    "TREATMENT OF":        "Concrete Treatment",
    "CEMENT":              "Cement",   # fallback for cement section
}

# Regex: "IS 269 : 1989" or "IS 1489 (Part 2) : 1991" etc.
IS_HEADER_RE = re.compile(
    r"IS\s+"                          # literal "IS "
    r"(\d+(?:\s*\([^\)]+\))?)"        # IS number, optionally with (Part N)
    r"\s*:\s*"                        # colon separator
    r"(\d{4})"                        # year
    r"\s+"                            # whitespace
    r"([A-Z][^\n]{5,})",              # title (at least 6 chars, starts uppercase)
    re.IGNORECASE,
)

# Detect section title page: line exactly "SECTION N"
SECTION_PAGE_RE = re.compile(r"^SECTION\s+(\d+)\s*$", re.MULTILINE)

# Detect "SUMMARY OF" marker (precedes every IS entry)
SUMMARY_OF_RE = re.compile(r"SUMMARY\s+OF", re.IGNORECASE)

# Extract scope sentence
SCOPE_RE = re.compile(
    r"(?:1\.|Scope)[^\—\-–]*[—\-–]\s*(.+?)(?:\.|$)",
    re.DOTALL | re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove control characters and normalise whitespace runs."""
    text = re.sub(r"\f", " ", text)                 # form feed
    text = re.sub(r"[ \t]+", " ", text)             # horizontal space
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    return text.strip()


def normalise_is_number(raw_num: str, year: str) -> str:
    """Return canonical form:  IS 269: 1989  or  IS 1489 (Part 2): 1991"""
    # Collapse internal spaces in the number part
    num = re.sub(r"\s+", " ", raw_num.strip())
    # Normalise Part notation: (Part1) → (Part 1)
    num = re.sub(r"\(Part\s*(\d+)\)", r"(Part \1)", num, flags=re.IGNORECASE)
    return f"IS {num}: {year}"


def extract_scope(content: str) -> str:
    """Try to pull the first scope sentence out of the full text."""
    m = SCOPE_RE.search(content)
    if m:
        scope = m.group(1).strip()
        # Truncate at first full stop
        dot = scope.find(".")
        if dot > 20:
            scope = scope[:dot].strip()
        return scope[:300]
    return ""


def guess_sub_category(title: str, category: str) -> str:
    """
    For Section 1 (Cement and Concrete), use the standard title to assign
    a finer sub-category. For other sections the category itself is enough.
    """
    if category != "Cement and Concrete":
        return category

    t = title.upper()
    if "AGGREGATE" in t:
        return "Aggregates"
    if "ASBESTOS" in t:
        return "Asbestos Cement Products"
    if "PIPE" in t or "MANHOLE" in t or "COVER" in t:
        return "Concrete Pipes and Precast"
    if (
        "MASONRY" in t or "BLOCK" in t or "PRECAST" in t
        or "FERROCEMENT" in t or "PANEL" in t
        or "PLANK" in t or "JOIST" in t or "CHANNEL" in t
        or "KERB" in t or "COPING" in t or "LINTEL" in t
        or "ROOFING SHEET" in t or "WALL SLAB" in t
        or "FLOOR" in t or "CABLE COVER" in t
        or "DOOR" in t or "WINDOW" in t
        or "FENCE" in t
    ):
        return "Cement Matrix Products"
    if "LIME" in t:
        return "Building Limes"
    # Most remaining entries in Section 1 are cement types
    return "Cement"


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def find_section_page_map(pdf) -> dict[int, int]:
    """
    Scan every page and return {section_number: 0-based_page_index}
    for the title page of each section.
    """
    mapping: dict[int, int] = {}
    for idx, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        m = SECTION_PAGE_RE.search(text)
        if m:
            sec = int(m.group(1))
            if sec not in mapping:
                mapping[sec] = idx
    return mapping


def page_to_section(page_idx: int, section_starts: dict[int, int]) -> int:
    """Return the section number that a given page belongs to."""
    sorted_secs = sorted(section_starts.items(), key=lambda x: x[1])
    current_sec = 1
    for sec_num, start_page in sorted_secs:
        if page_idx >= start_page:
            current_sec = sec_num
        else:
            break
    return current_sec


def parse_pdf(pdf_path: Path) -> list[dict]:
    """
    Main parse logic.

    Strategy
    --------
    1. Scan every page for "SUMMARY OF" as a chunk boundary.
    2. Accumulate page text until the next "SUMMARY OF" (or EOF).
    3. In the accumulated block find the IS number + title header.
    4. Build a structured chunk dict.
    """
    chunks: list[dict] = []

    log.info("Opening PDF: %s", pdf_path)
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        log.info("Total pages: %d", total_pages)

        # Build section boundary map once
        log.info("Building section boundary map …")
        section_starts = find_section_page_map(pdf)
        log.info("Found %d sections", len(section_starts))

        # -------------------------------------------------------------------
        # Pass: accumulate text blocks delimited by "SUMMARY OF"
        # -------------------------------------------------------------------
        current_pages: list[str] = []
        current_start_page = 0

        def flush_block(pages_text: list[str], start_page: int) -> None:
            """Parse and append a completed text block."""
            if not pages_text:
                return
            full_text = "\n".join(pages_text)
            full_text = clean_text(full_text)

            # Identify IS number + title inside this block
            m = IS_HEADER_RE.search(full_text)
            if not m:
                return  # No recognisable IS header → skip (could be ToC page etc.)

            raw_num = m.group(1)
            year    = m.group(2)
            title   = m.group(3).strip()

            # Clean up title: remove revision notes "(Fourth Revision)" etc.
            title = re.sub(r"\(.*?revision.*?\)", "", title, flags=re.IGNORECASE)
            title = re.sub(r"\s{2,}", " ", title).strip()
            # Remove trailing non-alpha chars
            title = title.rstrip(".,;:-").strip()

            is_number = normalise_is_number(raw_num, year)
            section   = page_to_section(start_page, section_starts)
            category  = SECTION_CATEGORY.get(section, "General")
            sub_cat   = guess_sub_category(title, category)
            scope     = extract_scope(full_text)

            chunks.append(
                {
                    "is_number":    is_number,
                    "title":        title.upper(),
                    "year":         year,
                    "section":      section,
                    "category":     category,
                    "sub_category": sub_cat,
                    "scope":        scope,
                    "content":      full_text,
                }
            )

        log.info("Parsing pages …")
        for idx, page in enumerate(pdf.pages):
            if idx % 100 == 0:
                log.info("  … page %d / %d  (%d chunks so far)",
                         idx + 1, total_pages, len(chunks))

            text = page.extract_text() or ""

            if SUMMARY_OF_RE.search(text):
                # Flush the previous block, start a new one
                flush_block(current_pages, current_start_page)
                current_pages = [text]
                current_start_page = idx
            else:
                current_pages.append(text)

        # Flush the final block
        flush_block(current_pages, current_start_page)

    log.info("Parsed %d IS standard chunks", len(chunks))
    return chunks


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(chunks: list[dict]) -> list[dict]:
    """
    If the same IS number appears more than once (e.g. ToC vs actual entry),
    keep only the one with the longest content (the real summary).
    """
    seen: dict[str, dict] = {}
    for chunk in chunks:
        key = chunk["is_number"]
        if key not in seen or len(chunk["content"]) > len(seen[key]["content"]):
            seen[key] = chunk
    deduped = list(seen.values())
    log.info("After deduplication: %d unique chunks", len(deduped))
    return deduped


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(chunks: list[dict]) -> None:
    """Log a quick sanity check and warn about obvious issues."""
    years_ok  = sum(1 for c in chunks if re.match(r"\d{4}$", c["year"]))
    has_scope = sum(1 for c in chunks if c["scope"])
    log.info("Validation — valid years: %d/%d, has scope: %d/%d",
             years_ok, len(chunks), has_scope, len(chunks))

    # Check that the 10 public test-set IS numbers are present
    PUBLIC_STANDARDS = [
        "IS 269: 1989", "IS 383: 1970", "IS 458: 2003",
        "IS 2185 (Part 2): 1983", "IS 459: 1992", "IS 455: 1989",
        "IS 1489 (Part 2): 1991", "IS 3466: 1988",
        "IS 6909: 1990", "IS 8042: 1989",
    ]
    found_ids = {c["is_number"] for c in chunks}
    for std in PUBLIC_STANDARDS:
        status = "✓" if std in found_ids else "✗ MISSING"
        log.info("  Public test standard %s: %s", std, status)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest BIS SP 21 PDF into chunks.json")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Path to the SP 21 dataset PDF  (default: data/dataset.pdf)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path  (default: data/chunks.json)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        log.error("PDF not found: %s", args.pdf)
        raise SystemExit(1)

    chunks = parse_pdf(args.pdf)
    chunks = deduplicate(chunks)
    validate(chunks)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    log.info("Saved %d chunks → %s", len(chunks), args.output)


if __name__ == "__main__":
    main()
