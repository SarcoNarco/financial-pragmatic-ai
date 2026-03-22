def extract_key_drivers(results, limit=5):
    growth = []
    risk = []

    for r in results:
        text = r["text"][:150]
        text_lower = r["text"].lower()

        # Growth drivers
        if r["intent"] == "EXPANSION":
            growth.append(text)

        # STRICT risk drivers (must contain negative indicators)
        elif r["intent"] == "COST_PRESSURE" and any(
            x in text_lower for x in [
                "pressure",
                "decline",
                "risk",
                "fall",
                "compression",
                "uncertainty",
                "headwind"
            ]
        ):
            risk.append(text)

    return {
        "growth_drivers": growth[:limit],
        "risk_drivers": risk[:limit]
    }
