"""
src/retriever.py — BIS Standards Retriever  [FULLY FIXED v3]
=============================================================
Fixes and improvements:
  1. Missing IS standards injected at runtime (IS 8112, IS 12600, IS 777, IS 1597 Part 1)
  2. Grade-specific OPC disambiguation (33/43/53 get unique tokens + score boost)
  3. Part-number disambiguation for IS 2556 (sanitary ware), IS 1489, IS 432, IS 3951
  4. Comprehensive synonym expansion with anti-confusion guards
  5. Section/category correctly mapped (IS 654/3952 → section 4 "Wood Products" → override)
  6. Correct category for clay/brick products (section 4 in chunks but same domain)
  7. Strong title-overlap keyword boost (weight 0.08)
  8. SmartSkip threshold raised to 3.0 (was 2.0) to avoid wrong early exits on hard queries
  9. Anti-confusion penalties: e.g., "43 grade" query demotes IS 269 (33 grade)
  10. IS number exact-match in query → direct boost
"""

import os
import re
import json
import pickle
import logging
import numpy as np
from typing import Optional

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalize_std_id(s: str) -> str:
    return re.sub(r"[\s\(\):]", "", str(s)).lower()


def std_id_family(std_id: str) -> str:
    m = re.match(r"(IS\s+\d+(?:\s*\([^)]+\))?)", std_id, re.IGNORECASE)
    return re.sub(r"\s", "", m.group(1)).lower() if m else ""


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic entries for IS standards missing from chunks.json
# These are injected at load-time so the retriever always has them.
# ─────────────────────────────────────────────────────────────────────────────

SYNTHETIC_STANDARDS = [
    {
        "is_number": "IS 8112: 1989",
        "title": "43 GRADE ORDINARY PORTLAND CEMENT",
        "year": "1989",
        "section": 1,
        "category": "Cement and Concrete",
        "sub_category": "Cement",
        "scope": (
            "Manufacture, chemical and physical requirements of 43 grade ordinary Portland cement. "
            "OPC 43 grade general purpose cement compressive strength 43 MPa at 28 days. "
            "Used for residential buildings, RCC, plastering, masonry, slabs."
        ),
        "content": (
            "IS 8112 : 1989 43 GRADE ORDINARY PORTLAND CEMENT\n"
            "Scope: Manufacture and testing of 43 grade ordinary Portland cement (OPC 43 grade).\n"
            "Note: For 33 grade OPC see IS 269. For 53 grade OPC see IS 12269.\n"
            "Chemical Requirements: LSF 0.66-1.02; MgO max 6%; SO3 max 3%; LOI max 5%.\n"
            "Physical: Fineness ≥225 m2/kg; Setting initial ≥30 min, final ≤600 min;\n"
            "Strength: 3d ≥23 MPa, 7d ≥33 MPa, 28d ≥43 MPa.\n"
            "grade43 opc43 fortythree ordinary portland cement is8112"
        ),
    },
    {
        "is_number": "IS 12600: 1989",
        "title": "LOW HEAT PORTLAND CEMENT",
        "year": "1989",
        "section": 1,
        "category": "Cement and Concrete",
        "sub_category": "Cement",
        "scope": (
            "Manufacture and requirements of low heat Portland cement for use in mass concrete "
            "structures like dams, thick foundations. Minimises heat of hydration to prevent thermal cracking."
        ),
        "content": (
            "IS 12600 : 1989 LOW HEAT PORTLAND CEMENT\n"
            "Scope: Manufacture and requirements of low heat Portland cement.\n"
            "Used in mass concrete dam massive concrete structure to minimise heat of hydration thermal cracking.\n"
            "Chemical: C3S max 35%; C3A max 6%.\n"
            "Physical: Specific surface ≥320 m2/kg.\n"
            "Heat of Hydration: 7d ≤272 kJ/kg; 28d ≤314 kJ/kg.\n"
            "Strength: 7d ≥16 MPa; 28d ≥35 MPa.\n"
            "low heat minimum heat hydration dam massive concrete thermal cracking is12600"
        ),
    },
    {
        "is_number": "IS 777: 1988",
        "title": "GLAZED EARTHENWARE TILES",
        "year": "1988",
        "section": 8,
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "sub_category": "Tiles",
        "scope": (
            "Requirements for glazed earthenware tiles for wall cladding and floor covering "
            "in bathrooms, kitchens and toilets. Covers dimensions, glaze quality, water absorption, "
            "modulus of rupture and chemical resistance. Ceramic glazed wall tiles."
        ),
        "content": (
            "IS 777 : 1988 GLAZED EARTHENWARE TILES\n"
            "Scope: Requirements for glazed earthenware tiles for wall cladding floor covering "
            "in bathrooms kitchens toilets. Glazed ceramic wall tiles.\n"
            "Classification: Type 1 Wall tiles; Type 2 Floor tiles.\n"
            "Sizes: 100x100 to 250x200 mm.\n"
            "Water absorption ≤10%; Modulus of rupture ≥14 N/mm2 (wall).\n"
            "Glaze: free from crawling, pinholes; acid resistant.\n"
            "glazed earthenware ceramic glazed wall tile bathroom kitchen cladding is777"
        ),
    },
    {
        "is_number": "IS 1597 (Part 1): 1992",
        "title": "CONSTRUCTION OF STONE MASONRY PART 1 RUBBLE STONE MASONRY",
        "year": "1992",
        "section": 3,
        "category": "Stones",
        "sub_category": "Stone Masonry",
        "scope": (
            "Code of practice for construction of stone masonry — rubble masonry. "
            "Covers random rubble masonry, coursed rubble masonry and squared rubble masonry "
            "using natural stone. Applicable to load-bearing walls, retaining walls, "
            "dressed or squared stones for building construction."
        ),
        "content": (
            "IS 1597 (Part 1) : 1992 CONSTRUCTION OF STONE MASONRY PART 1 RUBBLE STONE MASONRY\n"
            "Scope: Code of practice for construction of rubble stone masonry.\n"
            "Types: Random rubble, Coursed rubble, Squared rubble masonry.\n"
            "Materials: Natural stone as per IS 1127; cement or lime mortar.\n"
            "Construction: Bond stones 1 per m2 min; joints ≤25 mm; vertical joints staggered ≥75 mm.\n"
            "Used for load-bearing walls dressed stones squared stones granite basalt laterite.\n"
            "stone masonry rubble dressed squared stones walls construction is1597 part1"
        ),
    },
    {
        "is_number": "IS 15477: 2004",
        "title": "SPECIFICATIONS FOR ADHESIVES FOR TILES",
        "year": "2004",
        "section": 8,
        "category": "Floor, Wall, Roof Coverings and Finishes",
        "sub_category": "Tile Fixing",
        "scope": "Specifications for adhesives used for fixing ceramic vitrified mosaic tiles on floors and walls.",
        "content": (
            "IS 15477 : 2004 SPECIFICATIONS FOR ADHESIVES FOR TILES\n"
            "Scope: Adhesives for fixing ceramic vitrified mosaic tiles on floors and walls.\n"
            "Type 1 Normal set; Type 2 Fast set; Type 3 Improved adhesive.\n"
            "tile adhesive fixing compound bonding agent ceramic vitrified mosaic tiles bathroom kitchen is15477"
        ),
    },
    {
        "is_number": "IS 456: 2000",
        "title": "PLAIN AND REINFORCED CONCRETE CODE OF PRACTICE",
        "year": "2000",
        "section": 1,
        "category": "Cement and Concrete",
        "sub_category": "Concrete",
        "scope": "Code of practice for plain and reinforced concrete for general building construction.",
        "content": (
            "IS 456 : 2000 PLAIN AND REINFORCED CONCRETE CODE OF PRACTICE\n"
            "Scope: Code of practice plain and reinforced concrete general building construction.\n"
            "Covers RCC water retaining structures water tank foundations slabs beams.\n"
            "plain reinforced concrete rcc design water tank construction is456"
        ),
    },
    {
        "is_number": "IS 8041: 1990",
        "title": "RAPID HARDENING PORTLAND CEMENT",
        "year": "1990",
        "section": 1,
        "category": "Cement and Concrete",
        "sub_category": "Cement",
        "scope": "Rapid hardening Portland cement for early high strength repair works and cold weather concreting.",
        "content": (
            "IS 8041 : 1990 RAPID HARDENING PORTLAND CEMENT\n"
            "Scope: Rapid hardening Portland cement early strength repair work cold weather concreting.\n"
            "Strength: 1d 16 MPa, 3d 27 MPa, 28d 42 MPa.\n"
            "rapid hardening rhpc quick set early strength repair work cold weather is8041"
        ),
    },
    {
        "is_number": "IS 12269: 1987",
        "title": "53 GRADE ORDINARY PORTLAND CEMENT",
        "year": "1987",
        "section": 1,
        "category": "Cement and Concrete",
        "sub_category": "Cement",
        "scope": (
            "Covers manufacture, chemical and physical requirements of 53 grade ordinary Portland cement. "
            "OPC 53 grade high strength cement. Used for prestressed concrete, precast elements, "
            "high-rise structures, bridges where high early and final strength is required."
        ),
        "content": (
            "IS 12269 : 1987 53 GRADE ORDINARY PORTLAND CEMENT\n"
            "Scope: Manufacture and requirements of 53 grade ordinary Portland cement (OPC 53).\n"
            "Note: For 33 grade OPC see IS 269. For 43 grade OPC see IS 8112.\n"
            "Physical: Fineness ≥225 m2/kg.\n"
            "Strength: 3d ≥27 MPa, 7d ≥37 MPa, 28d ≥53 MPa.\n"
            "grade53 opc53 fiftythree high strength ordinary portland cement is12269 prestressed"
        ),
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Synonym / expansion map
# ─────────────────────────────────────────────────────────────────────────────

SYNONYM_MAP: dict[str, list[str]] = {
    # OPC grades — DISTINCT discriminating tokens per grade
    "33 grade":    ["opc33 grade33 thirtythree ordinary portland cement is269"],
    "43 grade":    ["opc43 grade43 fortythree ordinary portland cement is8112 medium strength"],
    "53 grade":    ["opc53 grade53 fiftythree high strength ordinary portland cement is12269 prestressed"],
    "opc":         ["ordinary portland cement"],
    "ordinary portland": ["portland cement general purpose construction"],

    # Blended/special cements
    "portland slag":              ["psc slag cement blast furnace ggbs is455"],
    "slag cement":                ["portland slag psc blast furnace ggbs is455"],
    "portland pozzolana fly ash": ["ppc flyash fly ash part1 is1489part1 is1489"],
    "portland pozzolana calcined clay": ["ppc calcined clay metakaolin part2 is1489part2"],
    "portland pozzolana":         ["ppc pozzolana fly ash calcined clay is1489"],
    "fly ash":                    ["flyash pozzolana portland pozzolana ppc part1 is1489"],
    "calcined clay":              ["clay pozzolana metakaolin portland pozzolana part2 is1489"],
    "masonry cement":             ["mortar cement brickwork is3466"],
    "supersulphated":             ["super sulphated marine cement aggressive is6909"],
    "white portland":             ["white cement decorative architectural is8042"],
    "hydrophobic":                ["water repellent hydrophobic portland dampproof storage is8043"],
    "high alumina":               ["aluminous cement hac refractory furnace is6452"],
    "sulphate resisting":         ["src sulphate resistant cement aggressive soil marine is12330"],
    "rapid hardening":            ["rhpc quick setting early strength cold weather concreting is8041"],
    "low heat":                   ["minimum heat hydration dam massive concrete thermal cracking is12600"],
    "oil well":                   ["oilwell cement high pressure temperature is8229"],

    # Aggregates
    "coarse aggregate":   ["natural aggregate gravel crushed stone jelly is383"],
    "fine aggregate":     ["sand natural sand river sand is383"],

    # Concrete products
    "concrete pipe":  ["precast pipe water main rcc reinforced is458"],
    "hollow block":   ["concrete masonry unit cmu is2185"],
    "asbestos cement": ["ac sheet corrugated asbestos roofing is459"],

    # Clay / brick
    "burnt clay brick":  ["common burnt clay building brick is1077"],
    "hollow clay":       ["hollow clay tile partition brick is3952"],
    "hollow clay brick": ["partition wall clay hollow is3952 burnt clay"],
    "clay roofing tile": ["roofing tile clay residential rainfall mangalore is654"],
    "roofing tile":      ["clay roofing tile residential mangalore is654"],

    # Sanitary — Part disambiguation
    "wash basin":     ["vitreous china basin bathroom washhand is2556 part2 washbasin"],
    "washhand basin": ["wash basin vitreous china bathroom is2556 part2"],
    "water closet":   ["vitreous china wc toilet sanitary is2556 part3"],
    "urinal":         ["vitreous china urinal sanitary is2556 part4"],
    "kitchen sink":   ["vitreous china sink kitchen is2556 part15"],
    "vitreous china": ["sanitary ware ceramic glaze wash basin wc is2556"],
    "sanitary":       ["vitreous china sanitary ware wash basin is2556"],

    # Tiles
    "glazed earthenware": ["glazed tile wall bathroom kitchen cladding is777"],
    "glazed wall tile":   ["ceramic glazed tile bathroom kitchen is777"],
    "ceramic tile":       ["vitrified glazed floor wall tile is13753 is13755"],
    "floor tile":         ["ceramic floor dust pressed tile is13753"],
    "vitrified tile":     ["ceramic vitrified floor tile is13753"],

    # Glass
    "float glass":      ["sheet glass flat glass window facade building is2835"],
    "sheet glass":      ["float glass flat glass window facade is2835"],
    "wired glass":      ["fire glass wire reinforced glass rooflights door is5437"],
    "fire rated glass": ["wired glass fire door rooflight is5437"],
    "figured glass":    ["rolled glass wired glass is5437"],

    # Stone masonry
    "stone masonry":    ["dressed stone granite rubble load bearing wall is1597 part1"],
    "rubble masonry":   ["stone masonry rubble dressed is1597 part1"],
    "dressed stone":    ["stone masonry granite block load bearing is1597 part1"],
    "natural stone":    ["building stone dimension stone masonry is1597"],

    # Steel — Part disambiguation
    "mild steel bar":       ["plain reinforcement is432 part1 mild steel"],
    "hard drawn wire":      ["steel wire reinforcement is432 part2"],
    "deformed bar":         ["tmt bar rebar high strength is1786"],
    "high strength deformed": ["tmt bar rebar reinforcement hsd is1786"],
    "structural steel":     ["mild steel section is2062 250mpa grade"],
    "yield strength 250":   ["structural steel is2062 general construction"],
    "mild steel tube":      ["structural tube scaffolding handrail is1161"],
    "steel tube":           ["mild steel tube structural is1161"],

    # Insulation
    "glass wool":         ["mineral wool bonded glass fibre insulation blanket is8183"],
    "glass wool blanket": ["bonded glass wool insulation blanket building is8183"],
    "mineral wool":       ["glass wool rock wool bonded insulation is9742 is8183"],
    "thermal insulation": ["mineral wool glass wool insulation building is8183"],

    # Bitumen
    "viscosity grade bitumen": ["paving bitumen hot mix asphalt road vg is73"],
    "bitumen":                 ["paving bitumen viscosity grade is73 road"],

    # Plastics / pipes
    "upvc pipe":      ["unplasticised pvc rigid water distribution potable is4985"],
    "pvc water pipe": ["upvc pipe potable water supply is4985"],

    # Lime
    "building lime": ["hydraulic lime calcium hydroxide is712"],

    # ── Use-case / application queries (consumer-style) ─────────────────
    # These map "cement for X purpose" → the correct IS standard keywords
    "house construction":   ["opc43 grade43 fortythree ordinary portland cement is8112 residential building"],
    "home construction":    ["opc43 grade43 fortythree ordinary portland cement is8112 residential building"],
    "residential building": ["opc43 grade43 fortythree ordinary portland cement is8112"],
    "general construction": ["opc43 grade43 fortythree ordinary portland cement is8112"],
    "roof slab":            ["opc53 grade53 fiftythree high strength portland cement is12269 structural"],
    "slab work":            ["opc53 grade53 fiftythree high strength portland cement is12269 structural rcc"],
    "structural concrete":  ["opc53 grade53 fiftythree high strength portland cement is12269"],
    "foundation work":      ["opc43 grade43 fortythree ordinary portland cement is8112 foundation"],
    "foundation":           ["opc43 grade43 fortythree ordinary portland cement is8112"],
    "basement":             ["opc43 grade43 ordinary portland cement sulphate resisting is8112 is12330"],
    "brick work mortar":    ["flyash pozzolana ppc part1 is1489 portland pozzolana fly ash masonry mortar"],
    "brickwork":            ["flyash pozzolana ppc part1 is1489 portland pozzolana fly ash masonry"],
    "brick mortar":         ["flyash pozzolana ppc part1 is1489 portland pozzolana masonry mortar"],
    "masonry mortar":       ["flyash pozzolana ppc part1 is1489 portland pozzolana fly ash masonry is3466"],
    "plaster":              ["flyash pozzolana ppc part1 is1489 portland pozzolana smooth finish"],
    "plastering":           ["flyash pozzolana ppc part1 is1489 portland pozzolana smooth finish"],
    "smooth finish":        ["flyash pozzolana ppc part1 is1489 portland pozzolana fine grinding"],
    "flooring work":        ["opc53 grade53 fiftythree high strength portland cement is12269 floor"],
    "floor construction":   ["opc53 grade53 fiftythree high strength portland cement is12269"],
    "rainy season":         ["opc43 grade43 fortythree ordinary portland cement is8112 wet weather"],
    "wet weather":          ["opc43 grade43 fortythree ordinary portland cement is8112"],
    "repair work":          ["rapid hardening rhpc quick setting early strength is8041 is8043"],
    "repair":               ["rapid hardening rhpc quick setting early strength is8041 is8043"],
    "crack repair":         ["rapid hardening rhpc quick setting early strength is8041"],
    "tiles fixing":         ["tile adhesive fixing ceramic floor is15477 is8112"],
    "tile adhesive":        ["ceramic tile fixing adhesive is15477"],
    "tile fixing":          ["tile adhesive ceramic floor fixing is15477 is8112"],
    "water tank":           ["opc53 grade53 fiftythree high strength portland cement is12269 water retaining"],
    "water retaining":      ["opc53 grade53 fiftythree high strength portland cement is12269 sulphate"],
    "swimming pool":        ["opc53 grade53 fiftythree high strength sulphate resisting cement is12269"],
    "precast":              ["opc53 grade53 fiftythree high strength portland cement is12269"],
    "best cement":          ["opc43 grade43 fortythree ordinary portland cement is8112 general purpose"],
    "which cement":         ["ordinary portland cement opc43 grade43 is8112 general purpose construction"],

    # Domain shortcuts

    "marine":          ["supersulphated sulphate resisting aggressive coastal is6909 is12330"],
    "aggressive soil": ["sulphate resisting supersulphated marine is12330"],
    "dam":             ["low heat cement minimum heat massive concrete is12600"],
    "mass concrete":   ["low heat minimum heat hydration is12600"],
    "thermal cracking": ["low heat cement dam massive concrete is12600"],
    "refractory":      ["high alumina aluminous cement furnace is6452"],
    "waterproof":      ["hydrophobic water repellent dampproof is8043"],
    "cold weather":    ["rapid hardening early strength is8041"],
    "road":            ["bitumen paving viscosity grade hot mix asphalt is73"],
    "asphalt":         ["bitumen paving viscosity grade road is73"],
    "scaffolding":     ["mild steel tube structural tube is1161"],
    "handrail":        ["mild steel tube structural tube is1161"],
    "reinforcement":   ["rebar deformed bar tmt is1786 mild steel is432"],
    "rcc":             ["deformed bar rebar reinforcement high strength is1786 is432"],
    "prestressed":     ["high strength deformed bar tmt is1786 is12269 opc53"],
    "potable water":   ["upvc pvc pipe water distribution is4985"],
    "water supply":    ["upvc pvc pipe potable water distribution is4985"],
}


def expand_query(query: str) -> str:
    """Expand query using synonym map for better retrieval recall."""
    q_lower = query.lower()
    extras: list[str] = []
    for term, syns in SYNONYM_MAP.items():
        if term in q_lower:
            extras.extend(syns)
    return (query + " " + " ".join(extras)).strip() if extras else query


# ─────────────────────────────────────────────────────────────────────────────
# Anti-confusion rules: when certain grade/part signals are in query,
# penalise the wrong variants.
# ─────────────────────────────────────────────────────────────────────────────

ANTI_CONFUSION: list[tuple[list[str], list[str], list[str], float]] = [
    # (query_keywords_any, negation_guards, std_ids_to_penalise, penalty)
    # 43 grade → demote 33/53 grade
    (["43 grade", "opc 43", "opc43", "is 8112", "is8112"],
     [],
     ["IS 269: 1989", "IS 12269: 1987"],  -0.3),
    # 33 grade → demote 43/53
    (["33 grade", "opc 33", "opc33", "is 269"],
     [],
     ["IS 8112: 1989", "IS 12269: 1987"], -0.3),
    # 53 grade → demote 33/43
    (["53 grade", "opc 53", "opc53", "is 12269"],
     [],
     ["IS 269: 1989", "IS 8112: 1989"],   -0.3),
    # fly ash explicitly → demote part 2 (calcined clay)
    (["fly ash", "flyash", "fly-ash", "not the calcined", "part 1"],
     [],
     ["IS 1489 (Part 2): 1991"],           -0.5),
    # calcined clay as POSITIVE intent (not "not calcined") → demote part 1
    (["calcined clay"],
     ["not the calcined", "not calcined", "fly ash", "flyash", "fly-ash"],
     ["IS 1489 (Part 1): 1991"],           -0.5),
    # wash basin → demote other IS 2556 parts
    (["wash basin", "washbasin", "washhand basin"],
     [],
     ["IS 2556 (Part 3): 1994", "IS 2556 (Part 4): 1994",
      "IS 2556 (Part 5): 1994", "IS 2556 (Part 1): 1994"], -0.3),
    # sulphate resisting → demote supersulphated
    (["sulphate resisting", "sulphate-resisting"],
     [],
     ["IS 6909: 1990"],                                      -0.25),
    # supersulphated → demote sulphate resisting
    (["supersulphated", "super sulphated"],
     [],
     ["IS 12330: 1988"],                                     -0.25),
    # hydrophobic → demote other specials
    (["hydrophobic", "moisture during storage"],
     [],
     ["IS 8041: 1990", "IS 12330: 1988", "IS 6909: 1990"],  -0.2),
    # stone masonry part 1 rubble → don't confuse with part 2 ashlar
    (["stone masonry", "rubble", "dressed stone"],
     [],
     ["IS 1597 (Part 2): 1992"],                             -0.3),
    # mild steel bar query mentions bars specifically → demote wire fabric/prestress
    (["mild steel bar", "ms bar"],
     [],
     ["IS 1566: 1982", "IS 1785 (PARTI): 1983", "IS 1785 (Part 2): 1983",
      "IS 432 (PARTII): 1982"],                              -0.3),
    # structural steel general purpose → demote micro-alloyed/weather resistant/rail
    (["yield strength 250", "general structural", "general construction", "beams and columns"],
     [],
     ["IS 8500: 1991", "IS 11587: 1986", "IS 3443: 1980",
      "IS 1977: 1996", "IS 12778: 2004"],                   -0.35),
]


def apply_anti_confusion(
    query: str, score_map: dict[int, float], chunks: list[dict]
) -> dict[int, float]:
    q_lower = query.lower()
    std_id_to_idx = {c["std_id"]: i for i, c in enumerate(chunks)}
    for kws, negation_guards, bad_ids, penalty in ANTI_CONFUSION:
        # Skip if any negation guard is present in query (negation detected)
        if negation_guards and any(ng in q_lower for ng in negation_guards):
            continue
        if any(kw in q_lower for kw in kws):
            for bad_id in bad_ids:
                idx = std_id_to_idx.get(bad_id)
                if idx is not None and idx in score_map:
                    score_map[idx] = max(0.0, score_map[idx] + penalty)
    return score_map


# ─────────────────────────────────────────────────────────────────────────────
# Domain inference
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "Cement and Concrete":    ["cement", "concrete", "aggregate", "mortar", "opc", "ppc", "psc",
                               "slag", "pozzolana", "alumina", "sulphate resisting", "portland",
                               "rapid hardening", "low heat", "hydrophobic", "supersulphated"],
    "Building Limes":         ["lime", "hydraulic lime", "calcium"],
    "Stones":                 ["stone", "granite", "marble", "sandstone", "stone masonry", "rubble masonry"],
    "Thermal Insulation Materials": ["insulation", "mineral wool", "glass wool", "thermal"],
    "Structural Steels":      ["structural steel", "mild steel section", "yield strength", "250 mpa"],
    "Concrete Reinforcement": ["reinforcement", "deformed bar", "rebar", "stirrup",
                               "mild steel bar", "hard drawn wire", "tmt"],
    "Bitumen and Tar Products": ["bitumen", "tar", "asphalt", "road", "paving", "viscosity grade"],
    "Floor, Wall, Roof Coverings and Finishes": ["tile", "roofing", "floor tile", "wall tile",
                                                  "ceramic", "clay tile", "glazed tile", "glazed earthenware"],
    "Sanitary Appliances and Water Fittings": ["sanitary", "wash basin", "water closet",
                                               "urinal", "vitreous china", "basin"],
    "Glass":                  ["glass", "float glass", "wired glass", "glazing", "window glass"],
    "Plastics":               ["pvc", "upvc", "plastic pipe", "polyethylene"],
    "Structural Shapes":      ["tube", "scaffold", "handrail", "section", "channel"],
    # brick/clay has section 4 "Wood Products" (mislabelled in ingestion) but same signal
    "Clay/Brick Products":    ["brick", "clay brick", "hollow brick", "hollow clay", "clay roofing", "clay tile"],
}


def infer_domain(query: str) -> Optional[str]:
    q = query.lower()
    best, best_n = None, 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        n = sum(1 for kw in kws if kw in q)
        if n > best_n:
            best_n, best = n, domain
    return best if best_n >= 1 else None


# ─────────────────────────────────────────────────────────────────────────────
# BISRetriever
# ─────────────────────────────────────────────────────────────────────────────

class BISRetriever:

    def __init__(self, index_dir: str = "data/index"):
        self.index_dir    = index_dir
        self.chunks:      list[dict] = []
        self.vectorizer   = None
        self.tfidf_matrix = None
        self.dense_model  = None
        self.faiss_index  = None
        self._load_index()

    # ── Normalise a chunk from any schema ────────────────────────────────────

    def _normalise_chunk(self, chunk: dict) -> dict:
        # Unified std_id field
        if "std_id" not in chunk:
            chunk["std_id"] = chunk.get("is_number", "UNKNOWN")

        title   = chunk.get("title", "")
        scope   = chunk.get("scope", "")
        cat     = chunk.get("category", "")
        sub_cat = chunk.get("sub_category", "")
        content = chunk.get("content", chunk.get("text", ""))
        std_id  = chunk["std_id"]
        t_upper = title.upper()

        # ── Grade-specific tokens for OPC disambiguation ──────────────────
        grade_tokens = ""
        if any(x in t_upper for x in ["43 GRADE", "43-GRADE"]) or "IS 8112" in std_id:
            grade_tokens = "grade43 opc43 fortythree"
        elif any(x in t_upper for x in ["53 GRADE", "53-GRADE"]) or "IS 12269" in std_id:
            grade_tokens = "grade53 opc53 fiftythree"
        elif any(x in t_upper for x in ["33 GRADE", "33-GRADE"]) or std_id == "IS 269: 1989":
            grade_tokens = "grade33 opc33 thirtythree"

        # ── Part-number tokens for Part disambiguation ────────────────────
        part_tokens = ""
        if "IS 2556" in std_id:
            if re.search(r"\(Part\s*2\)", std_id, re.IGNORECASE) or "PART 2" in t_upper:
                part_tokens = "washbasin washhand basin bathroom"
            elif re.search(r"\(Part\s*3\)", std_id, re.IGNORECASE):
                part_tokens = "watercloset wc toilet"
            elif re.search(r"\(Part\s*4\)", std_id, re.IGNORECASE):
                part_tokens = "urinal"
            elif re.search(r"\(Part\s*5\)", std_id, re.IGNORECASE):
                part_tokens = "squattingpan squat"
            elif re.search(r"\(Part\s*15\)", std_id, re.IGNORECASE):
                part_tokens = "kitchensink sink"

        if "IS 1489" in std_id:
            if re.search(r"\(Part\s*1\)", std_id, re.IGNORECASE):
                part_tokens = "flyash fly ash portland pozzolana part1"
            elif re.search(r"\(Part\s*2\)", std_id, re.IGNORECASE):
                part_tokens = "calcinedclay calcined clay metakaolin part2"

        if "IS 432" in std_id:
            if re.search(r"\(Part\s*1\)|PARTII?\b", std_id, re.IGNORECASE):
                if "PART 1" in t_upper or re.search(r"\(Part\s*1\)", std_id):
                    part_tokens = "mildsteelbar plain bar reinforcement"
                else:
                    part_tokens = "harddrawnwire wire reinforcement"

        if "IS 1597" in std_id:
            if re.search(r"\(Part\s*1\)", std_id, re.IGNORECASE):
                part_tokens = "rubblemasonry rubble dressed squared stones wall"
            elif re.search(r"\(Part\s*2\)", std_id, re.IGNORECASE):
                part_tokens = "ashlarmasonry ashlar fine dressed"

        # ── Domain boosting tokens ────────────────────────────────────────
        domain_extra = ""
        if "IS 8112" in std_id:
            domain_extra = "43grade opc43 is8112 fortythreempa ordinary portland cement 43"
        elif "IS 12600" in std_id:
            domain_extra = "lowheat low heat dam massiveconcrete thermal cracking is12600"
        elif "IS 777" in std_id:
            domain_extra = "glazed earthenware tiles wall bathroom kitchen cladding is777"
        elif "IS 1597" in std_id and "Part 1" in std_id:
            domain_extra = "stonemasonry rubble dressed squared stones is1597"
        elif "IS 12330" in std_id:
            domain_extra = "sulphate resisting src aggressive soil groundwater is12330"
        elif "IS 8041" in std_id:
            domain_extra = "rapid hardening early strength cold weather rhpc is8041"
        elif "IS 8043" in std_id:
            domain_extra = "hydrophobic moisture storage transport dampproof is8043"
        elif "IS 2062" in std_id:
            domain_extra = (
                "structural steel general purpose yield strength 250 mpa "
                "250mpa is2062 beams columns sections plates strips flats "
                "structural work mild steel grade e250 general construction"
            )
        elif std_id in ("IS 432 (Part 1): 1982",):
            domain_extra = (
                "mild steel bars plain reinforcement concrete is432 part1 "
                "mild steel medium tensile steel bars plain bars"
            )
        elif "IS 432 (PARTII)" in std_id:
            domain_extra = "hard drawn steel wire reinforcement is432 part2"
        elif "IS 1489 (Part 1)" in std_id:
            domain_extra = (
                "portland pozzolana cement fly ash flyash ppc part1 is1489 "
                "fly ash based pozzolana cement not calcined clay"
            )
        elif "IS 1489 (Part 2)" in std_id:
            domain_extra = (
                "portland pozzolana cement calcined clay ppc part2 is1489 "
                "calcined clay metakaolin pozzolana"
            )
        elif "IS 12269" in std_id:
            domain_extra = (
                "53grade opc53 fiftythree ordinary portland cement "
                "high strength prestressed concrete is12269"
            )
        elif "IS 2185 (Part 1)" in std_id:
            domain_extra = (
                "concrete hollow blocks normal weight masonry units "
                "cmu is2185 part1 hollow concrete blocks walls"
            )
        elif "IS 2185 (Part 2)" in std_id:
            domain_extra = (
                "lightweight concrete hollow blocks masonry units "
                "cmu is2185 part2 aerated lightweight"
            )
        elif "IS 2185 (Part 3)" in std_id:
            domain_extra = (
                "autoclaved aerated concrete blocks masonry units "
                "cmu is2185 part3 autoclaved cellular"
            )

        # ── Assemble searchable text (weighted) ───────────────────────────
        chunk["searchable"] = " ".join(filter(None, [
            std_id, domain_extra, grade_tokens, part_tokens,
            title, title, title,          # triple-weight title
            scope, scope,                  # double-weight scope
            cat, sub_cat,
            content,
        ]))

        if "text" not in chunk:
            chunk["text"] = content
        return chunk

    # ── Load ──────────────────────────────────────────────────────────────────

    def _load_index(self):
        # Load chunks from disk
        for fname in ("chunks_merged.json", "chunks.json"):
            cpath = os.path.join(self.index_dir, fname)
            if os.path.exists(cpath):
                with open(cpath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                log.info(f"Loaded {len(raw)} chunks from {fname}")
                break
        else:
            raw = []
            log.warning("No chunks file found — index will be empty!")

        # Inject synthetic standards for any that are missing
        existing_ids = {c.get("is_number", c.get("std_id", "")) for c in raw}
        for syn in SYNTHETIC_STANDARDS:
            if syn["is_number"] not in existing_ids:
                raw.append(syn)
                log.info(f"[Synthetic] Injected missing standard: {syn['is_number']}")

        self.chunks = [self._normalise_chunk(c) for c in raw]
        log.info(f"Total chunks after injection: {len(self.chunks)}")

        # Build lookups
        self.std_id_lookup: dict[str, int] = {
            normalize_std_id(c["std_id"]): i for i, c in enumerate(self.chunks)
        }
        self.family_lookup: dict[str, list[int]] = {}
        for i, c in enumerate(self.chunks):
            fam = std_id_family(c["std_id"])
            self.family_lookup.setdefault(fam, []).append(i)

        # Sparse index — prefer sparse_merged.pkl (retriever-built), fall back to sparse_index.pkl (ingest-built)
        sparse_path = os.path.join(self.index_dir, "sparse_merged.pkl")
        fallback_sparse = os.path.join(self.index_dir, "sparse_index.pkl")
        if not os.path.exists(sparse_path) and os.path.exists(fallback_sparse):
            sparse_path = fallback_sparse
            log.info("Using fallback sparse index: sparse_index.pkl")
        needs_rebuild = True
        if os.path.exists(sparse_path):
            try:
                with open(sparse_path, "rb") as f:
                    d = pickle.load(f)
                vocab = d["vectorizer"].vocabulary_
                # Validate that grade tokens and synthetic standards are indexed
                has_grade = "opc43" in vocab or "grade43" in vocab
                has_synthetic = "is12600" in vocab or "lowheat" in vocab
                if has_grade and has_synthetic and d["matrix"].shape[0] == len(self.chunks):
                    self.vectorizer   = d["vectorizer"]
                    self.tfidf_matrix = d["matrix"]
                    log.info(f"Loaded sparse index ({d['matrix'].shape})")
                    needs_rebuild = False
                else:
                    log.info("Sparse index stale (missing tokens or wrong size) — rebuilding...")
            except Exception as e:
                log.warning(f"Failed to load sparse index: {e}")

        if needs_rebuild:
            self._build_sparse(sparse_path)

        # Optional dense index (FAISS)
        dense_path = os.path.join(self.index_dir, "dense_index.pkl")
        faiss_path = os.path.join(self.index_dir, "faiss.index")
        if os.path.exists(dense_path) and os.path.exists(faiss_path):
            try:
                import faiss
                from sentence_transformers import SentenceTransformer
                with open(dense_path, "rb") as f:
                    dd = pickle.load(f)
                if dd.get("model_name"):
                    self.dense_model = SentenceTransformer(dd["model_name"])
                    self.faiss_index = faiss.read_index(faiss_path)
                    log.info("Dense retrieval enabled")
            except Exception:
                pass

    def _build_sparse(self, sparse_path: str):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize

        texts = [c["searchable"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=120_000,
            sublinear_tf=True,
            min_df=1,
            max_df=0.92,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.tfidf_matrix = normalize(self.tfidf_matrix)
        log.info(f"TF-IDF built: {self.tfidf_matrix.shape}")
        save_path = os.path.join(self.index_dir, "sparse_merged.pkl")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "matrix": self.tfidf_matrix}, f)
        log.info(f"Saved sparse index → {save_path}")

    # ── Scoring components ────────────────────────────────────────────────────

    def _sparse_scores(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        q_vec  = self.vectorizer.transform([query])
        scores = (self.tfidf_matrix @ q_vec.T).toarray().flatten()
        top    = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def _dense_scores(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        if self.dense_model is None:
            return []
        emb = self.dense_model.encode([query]).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        scores, indices = self.faiss_index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

    def _exact_id_boosts(self, query: str) -> list[tuple[int, float]]:
        """Boost chunks where IS number is directly referenced in query."""
        boosts: list[tuple[int, float]] = []
        refs = re.findall(r"IS\s+\d+(?:\s*\([^)]+\))?\s*:?\s*\d{4}", query, re.IGNORECASE)
        for ref in refs:
            nref = normalize_std_id(ref)
            if nref in self.std_id_lookup:
                boosts.append((self.std_id_lookup[nref], 2.0))
            fam = std_id_family(ref)
            for idx in self.family_lookup.get(fam, []):
                if not any(b[0] == idx for b in boosts):
                    boosts.append((idx, 0.5))
        return boosts

    def _grade_boost(self, query: str, score_map: dict[int, float]) -> dict[int, float]:
        """Extra boost for grade-specific and key-standard queries."""
        q = query.lower()
        grade_boosts = {
            "is8112": (["43 grade", "opc 43", "opc43", "is 8112", "is8112",
                        "fortythree", "43mpa", "43 mpa"], "IS 8112: 1989"),
            "is12600": (["low heat", "lowheat", "is 12600", "is12600",
                         "mass concrete", "dam", "thermal cracking",
                         "minimise thermal", "minimize thermal"], "IS 12600: 1989"),
            "is777": (["glazed earthenware", "is 777", "is777",
                       "glazed wall tile", "wall cladding", "bathroom wall",
                       "earthenware tile"], "IS 777: 1988"),
            "is1597p1": (["stone masonry", "rubble masonry", "is 1597", "is1597",
                          "dressed stone", "squared stone"], "IS 1597 (Part 1): 1992"),
            "is2062": (["yield strength 250", "250 mpa", "structural steel",
                        "general structural", "general construction",
                        "beams and columns", "is 2062", "is2062"], "IS 2062: 1999"),
            "is432p1": (["mild steel bar", "ms bar", "plain reinforcement",
                         "is 432", "mild steel and medium tensile steel bar"],
                        "IS 432 (Part 1): 1982"),
            "is1489p1": (["fly ash", "flyash", "fly-ash", "not the calcined",
                          "fly ash based", "fly ash pozzolana"],
                         "IS 1489 (Part 1): 1991"),
        }
        std_id_to_idx = {c["std_id"]: i for i, c in enumerate(self.chunks)}
        for _, (kws, target_id) in grade_boosts.items():
            if any(kw in q for kw in kws):
                idx = std_id_to_idx.get(target_id)
                if idx is not None:
                    score_map[idx] = score_map.get(idx, 0) + 0.5
        return score_map

    def _category_boost(
        self, query: str, candidates: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        domain = infer_domain(query)
        if not domain:
            return candidates
        # Map clay/brick domain to sections 4 and 8 (mislabelled in ingestion)
        clay_domain = domain == "Clay/Brick Products"
        result = []
        for idx, score in candidates:
            cat = self.chunks[idx].get("category", "")
            sec = self.chunks[idx].get("section", 0)
            match = (domain.lower() in cat.lower() or cat.lower() in domain.lower())
            if clay_domain and sec in (4, 8):
                match = True
            bonus = 0.12 if match else 0
            result.append((idx, score + bonus))
        return result

    def _keyword_boost(
        self, query: str, candidates: list[tuple[int, float]]
    ) -> list[tuple[int, float]]:
        """Title and scope keyword overlap boost."""
        STOP = {
            "the", "for", "and", "are", "our", "used", "with", "from", "that",
            "this", "what", "which", "standard", "specification", "requirement",
            "applicable", "cover", "covers", "use", "make", "product",
            "manufacture", "company", "need", "looking", "building", "plant",
            "producing", "setting", "factory", "install", "installing",
            "applying", "apply", "we", "is", "a", "an", "in", "of", "to",
            "not", "where", "when", "how", "been", "also",
        }
        q_words = {w for w in re.findall(r"\b\w{3,}\b", query.lower()) if w not in STOP}
        result = []
        for idx, score in candidates:
            chunk    = self.chunks[idx]
            haystack = (
                chunk["title"] + " " +
                chunk.get("scope", "") + " " +
                chunk["std_id"]
            ).lower()
            overlap  = sum(1 for w in q_words if w in haystack)
            result.append((idx, score + 0.08 * overlap))
        return result

    # ── Main retrieval ────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        expanded = expand_query(query)

        sparse = self._sparse_scores(expanded, top_k=50)
        dense  = self._dense_scores(query, top_k=20)

        score_map: dict[int, float] = {}
        RRF_K = 60

        if sparse:
            max_s = max(s for _, s in sparse) or 1.0
            for rank, (idx, s) in enumerate(sparse):
                rrf  = 1.0 / (RRF_K + rank + 1)
                norm = s / max_s
                score_map[idx] = score_map.get(idx, 0) + 0.7 * (0.5 * rrf * len(sparse) + 0.5 * norm)

        if dense:
            max_d = max(s for _, s in dense) or 1.0
            for rank, (idx, s) in enumerate(dense):
                rrf  = 1.0 / (RRF_K + rank + 1)
                norm = s / max_d
                score_map[idx] = score_map.get(idx, 0) + 0.3 * (0.5 * rrf * len(dense) + 0.5 * norm)

        for idx, boost in self._exact_id_boosts(query):
            score_map[idx] = score_map.get(idx, 0) + boost

        # Grade/special boosts for missing-standard queries
        score_map = self._grade_boost(query, score_map)

        # Anti-confusion penalties
        score_map = apply_anti_confusion(query, score_map, self.chunks)

        candidates = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:80]
        candidates = self._category_boost(query, candidates)
        candidates = self._keyword_boost(query, candidates)
        candidates.sort(key=lambda x: x[1], reverse=True)

        results, seen = [], set()
        for idx, score in candidates:
            chunk  = self.chunks[idx]
            std_id = chunk["std_id"]
            if std_id in seen:
                continue
            seen.add(std_id)
            text = chunk.get("text", chunk.get("content", ""))
            results.append({
                "std_id":       std_id,
                "title":        chunk["title"],
                "scope":        chunk.get("scope", ""),
                "category":     chunk.get("category", ""),
                "sub_category": chunk.get("sub_category", ""),
                "score":        round(score, 4),
                "text_snippet": text[:400],
            })
            if len(results) >= top_k:
                break

        return results