def compute_risk_score(intents):
    score = 45

    for item in intents:
        if item["intent"] == "COST_PRESSURE":
            score += 3
        elif item["intent"] == "STRATEGIC_PROBING":
            score += 1
        elif item["intent"] == "EXPANSION":
            score -= 1

    return max(5, min(95, score))


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


def generate_insight(score, intents):
    counts = {}
    total = len(intents)

    for item in intents:
        counts[item["intent"]] = counts.get(item["intent"], 0) + 1

    expansion = counts.get("EXPANSION", 0)
    cost = counts.get("COST_PRESSURE", 0)
    probing = counts.get("STRATEGIC_PROBING", 0)

    if total == 0:
        return "No interpretable signals were detected in the discussion."

    if score <= 35:
        if expansion >= cost:
            return (
                "Strong growth signals driven by expansion-heavy management commentary "
                "with limited cost pressure signals."
            )
        return "Growth-oriented messaging appears present but mixed with cost-related caution."

    if score >= 65:
        if cost > expansion:
            return (
                "Elevated risk profile led by persistent cost and margin pressure signals "
                "across management commentary."
            )
        return "Risk signals dominate this transcript despite intermittent growth commentary."

    if probing > 0 and cost > 0:
        return "Mixed outlook: analyst probing and cost pressure are balancing growth narratives."

    return "Balanced management commentary with no single dominant directional signal."


def compute_confidence(intents):
    counts = {}
    total = len(intents)

    if total == 0:
        return 0.0

    for item in intents:
        counts[item["intent"]] = counts.get(item["intent"], 0) + 1

    dominant = max(counts.values())
    confidence = dominant / total

    return round(confidence * 100, 2)


def detect_volatility(intents):
    if len(intents) <= 1:
        return "LOW"

    changes = 0

    for i in range(1, len(intents)):
        if intents[i]["intent"] != intents[i - 1]["intent"]:
            changes += 1

    volatility_ratio = changes / len(intents)

    if volatility_ratio > 0.5:
        return "HIGH"
    elif volatility_ratio > 0.3:
        return "MEDIUM"
    else:
        return "LOW"
