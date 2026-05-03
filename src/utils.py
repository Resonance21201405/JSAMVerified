import re

def classify_query(query: str) -> str:
    """Classify a BIS query to guide retrieval strategy.

    Returns one of: 'ambiguous', 'comparison', 'direct_lookup',
    'domain_specific', 'normal'.
    """
    q = query.lower()

    if len(q.split()) <= 3:
        return "ambiguous"

    if any(k in q for k in ["difference", "compare", "which", "best", "type"]):
        return "comparison"

    if any(k in q for k in ["is code", "standard for", "what is", "what bis"]):
        return "direct_lookup"

    if any(k in q for k in ["grade", "cement", "aggregate", "pipe", "concrete",
                              "steel", "brick", "tile", "lime", "bitumen", "glass"]):
        return "domain_specific"

    return "normal"