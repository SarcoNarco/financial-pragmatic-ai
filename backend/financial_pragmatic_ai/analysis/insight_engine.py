def extract_key_drivers(results, limit=5):
    growth = []
    risk = []

    for result in results:
        text = result["text"][:150]

        if result["intent"] == "EXPANSION":
            growth.append(text)
        elif result["intent"] == "COST_PRESSURE":
            risk.append(text)

    return {
        "growth_drivers": growth[:limit],
        "risk_drivers": risk[:limit]
    }
