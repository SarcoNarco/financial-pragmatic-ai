intent_weights = {
    "EXPANSION": 2,
    "COST_PRESSURE": -3,
    "STRATEGIC_PROBING": -1,
    "GENERAL_UPDATE": 0,
}

speaker_weights = {
    "CEO": 1.5,
    "CFO": 2.0,
    "ANALYST": 1.0,
}


def compute_risk_score(segments):
    score = 0

    for segment in segments:
        intent_val = intent_weights.get(segment["intent"], 0)
        speaker_val = speaker_weights.get(segment["speaker"], 1)
        score += intent_val * speaker_val

    return score


def normalize_score(score):
    return max(0, min(100, 50 - score * 5))


def detect_conflict(segments):
    speakers = {x["speaker"]: x["intent"] for x in segments}

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
