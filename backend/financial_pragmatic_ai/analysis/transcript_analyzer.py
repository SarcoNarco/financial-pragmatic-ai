import re
from signal import signal
import torch

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

    def parse_transcript(self, text):

        lines = text.strip().split("\n")

        segments = []

        for line in lines:

            line = line.strip()

            if not line:
                continue

            if ":" not in line:
                continue

            speaker, content = line.split(":", 1)

            speaker = speaker.strip().upper()

            if speaker in ["CEO", "CFO", "ANALYST"]:

                segments.append({
                    "speaker": speaker,
                    "text": content.strip()
                })

        return segments

    def predict_intent(self, text, speaker):
        return self.model.predict(text, speaker=speaker, target_device=device)

    def predict_conversation_signal(self, intents):

        vec = vectorize_conversation(intents).unsqueeze(0)

        with torch.no_grad():
            logits = self.conversation_model(vec)

        pred = torch.argmax(logits, dim=-1).item()

        labels = ["neutral", "risk", "growth"]

        return labels[pred]

    def analyze(self, transcript):

        segments = self.parse_transcript(transcript)

        results = []

        for seg in segments:

            intent = self.predict_intent(
                seg["text"],
                seg["speaker"]
            )

            results.append({
                "speaker": seg["speaker"],
                "text": seg["text"],
                "intent": intent
            })

        signal = self.predict_conversation_signal(results)

        return {
            "segments": results,
            "financial_signal": signal
        }
