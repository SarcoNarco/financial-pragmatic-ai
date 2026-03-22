def compute_risk_score(intents):
    score = 50

    for item in intents:
        if item["intent"] == "COST_PRESSURE":
            score += 3
        elif item["intent"] == "STRATEGIC_PROBING":
            score += 1
        elif item["intent"] == "EXPANSION":
            score -= 2

    return max(0, min(100, score))


def derive_signal(score):
    if score >= 65:
        return "risk"
    elif score <= 35:
        return "growth"
    else:
        return "neutral"


def derive_market_prediction(score):
    if score >= 65:
        return "DOWN"
    elif score <= 35:
        return "UP"
    else:
        return "NEUTRAL"


def generate_insight(score):
    if score >= 65:
        return "Discussion indicates elevated financial or margin risk."
    elif score <= 35:
        return "Strong growth signals with positive business momentum."
    else:
        return "Mixed signals in management discussion."
