from financial_pragmatic_ai.analysis.market_predictor import predict_market_reaction


def compute_risk_score(intents):
    score = 50

    for item in intents:
        if item["intent"] == "COST_PRESSURE":
            score += 2
        elif item["intent"] == "STRATEGIC_PROBING":
            score += 1
        elif item["intent"] == "EXPANSION":
            score -= 1

    return max(0, min(100, score))


def compute_market_prediction(intents):
    risk_score = compute_risk_score(intents)
    market_prediction = predict_market_reaction(risk_score)
    return market_prediction


def detect_conflict(intents):
    speakers = {x["speaker"]: x["intent"] for x in intents}

    if speakers.get("CEO") == "EXPANSION" and speakers.get("CFO") == "COST_PRESSURE":
        return "Strategic conflict between growth and financial pressure"

    return None


def generate_advanced_insight(intents):
    speakers = {x["speaker"]: x["intent"] for x in intents}

    ceo = speakers.get("CEO")
    cfo = speakers.get("CFO")
    analyst = speakers.get("ANALYST")

    if ceo == "EXPANSION" and cfo == "COST_PRESSURE":
        return "Management signals growth, but financials indicate cost pressure -> margin risk."

    if analyst == "STRATEGIC_PROBING":
        return "Analyst questions suggest market skepticism about future performance."

    return "Mixed signals in management discussion."


class FinancialSignalEngine:

    def analyze(self, intents):
        return generate_advanced_insight(intents)
