GROWTH_KEYWORDS = [
    "growth",
    "revenue",
    "demand",
    "expansion",
    "momentum",
    "record",
    "improved",
    "increase",
]

RISK_KEYWORDS = [
    "pressure",
    "decline",
    "risk",
    "loss",
    "drop",
    "uncertainty",
    "headwind",
    "compression",
]


def extract_key_drivers(results, limit=3):
    growth = []
    risk = []

    for r in results:
        text = r["text"][:150]
        text_lower = r["text"].lower()
        word_count = len(r["text"].split())

        # Growth drivers
        if (
            r["intent"] == "EXPANSION"
            and word_count > 10
            and any(keyword in text_lower for keyword in GROWTH_KEYWORDS)
        ):
            growth.append(text)

        elif r["intent"] == "COST_PRESSURE":

            if (
                any(keyword in text_lower for keyword in RISK_KEYWORDS)
                and word_count > 10
            ):
                risk.append(text)

    return {
        "growth_drivers": growth[:limit],
        "risk_drivers": risk[:limit]
    }
