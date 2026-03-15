class FinancialSignalEngine:

    def analyze(self, intents):

        signals = []

        speaker_intents = {x["speaker"]: x["intent"] for x in intents}

        ceo = speaker_intents.get("CEO")
        cfo = speaker_intents.get("CFO")
        analyst = speaker_intents.get("ANALYST")

        if ceo == "EXPANSION" and cfo == "COST_PRESSURE":
            signals.append("margin_compression_risk")

        if analyst == "STRATEGIC_PROBING" and cfo == "COST_PRESSURE":
            signals.append("analyst_detected_risk")

        if ceo == "EXPANSION" and cfo == "GENERAL_UPDATE":
            signals.append("growth_narrative")

        return signals