def extract_key_drivers(results, limit=5):
    growth = []
    risk = []

    for r in results:
        text = r["text"][:150]
        text_lower = r["text"].lower()

        # Growth drivers
        if r["intent"] == "EXPANSION":
            growth.append(text)

        elif r["intent"] == "COST_PRESSURE":

            if (
                any(x in text_lower for x in [
                    "pressure",
                    "decline",
                    "risk",
                    "loss",
                    "drop",
                    "uncertainty",
                    "headwind",
                    "compression"
                ])
                and len(r["text"].split()) > 8
            ):
                risk.append(text)

    return {
        "growth_drivers": growth[:limit],
        "risk_drivers": risk[:limit]
    }
