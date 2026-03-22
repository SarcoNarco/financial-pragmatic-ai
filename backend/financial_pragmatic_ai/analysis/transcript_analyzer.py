import torch

from financial_pragmatic_ai.analysis.transcript_parser import parse_transcript
from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.analysis.conversation_vectorizer import vectorize_conversation
from financial_pragmatic_ai.models.conversation_interaction_model import ConversationInteractionModel

MODEL_PATH = "financial_pragmatic_ai/models/pragmatic_transformer_trained.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

INTENT_LABELS = [
    "EXPANSION",
    "COST_PRESSURE",
    "STRATEGIC_PROBING",
    "GENERAL_UPDATE",
]


def smooth_intents(results, window=5):
    smoothed = []

    for i in range(len(results)):
        window_slice = results[max(0, i - window): i + window + 1]

        weights = {}
        for j, r in enumerate(window_slice):
            distance = abs(i - (max(0, i - window) + j))
            weight = 1 / (distance + 1)
            weights[r["intent"]] = weights.get(r["intent"], 0) + weight

        dominant = max(weights, key=weights.get)

        smoothed.append({
            **results[i],
            "intent": dominant
        })

    return smoothed


class TranscriptAnalyzer:

    def __init__(self):

        self.model = FinancialPragmaticTransformer()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.conversation_model = ConversationInteractionModel()
        self.conversation_model.load_state_dict(
            torch.load("financial_pragmatic_ai/models/conversation_signal_model.pt")
        )
        self.conversation_model.eval()
    def predict_intent(self, text, speaker):
        return self.model.predict(text, speaker=speaker, target_device=device)

    def predict_conversation_signal(self, intents):

        vec = vectorize_conversation(intents).unsqueeze(0)

        with torch.no_grad():
            logits = self.conversation_model(vec)

        pred = torch.argmax(logits, dim=-1).item()

        labels = ["neutral", "risk", "growth"]

        return labels[pred]

    def analyze(self, raw_text):
        segments = parse_transcript(raw_text)
        results = []

        for seg in segments:
            intent = self.model.predict(seg["text"], seg["speaker"])
            text_lower = seg["text"].lower()

            # Override weak GENERAL_UPDATE predictions
            if intent in ["GENERAL_UPDATE", "EXECUTIVE"]:
                if any(
                    x in text_lower
                    for x in [
                        "strong growth",
                        "revenue increased",
                        "sales increased",
                        "record revenue",
                        "expanded operations",
                        "growth accelerated",
                        "higher demand"
                    ]
                ):
                    intent = "EXPANSION"
                elif any(
                    x in text_lower
                    for x in [
                        "performance",
                        "results",
                        "quarter",
                        "guidance",
                        "outlook",
                        "business growth",
                        "positive momentum"
                    ]
                ):
                    intent = "EXPANSION"
                elif any(
                    x in text_lower
                    for x in [
                        "cost pressure",
                        "margin pressure",
                        "decline",
                        "risk",
                        "inflation",
                        "headwinds",
                        "lower margins",
                        "compression"
                    ]
                ):
                    intent = "COST_PRESSURE"
                elif any(
                    x in text_lower
                    for x in [
                        "how",
                        "what",
                        "why",
                        "could you",
                        "guidance",
                        "outlook",
                        "expect going forward"
                    ]
                ):
                    intent = "STRATEGIC_PROBING"

            results.append({
                "speaker": seg["speaker"],
                "text": seg["text"],
                "intent": intent
            })

        results = smooth_intents(results)

        print("[DEBUG] SAMPLE OUTPUT:")
        for result in results[:5]:
            print(result)

        return results
