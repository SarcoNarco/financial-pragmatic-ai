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
