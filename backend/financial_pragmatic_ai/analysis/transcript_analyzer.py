from pathlib import Path

import torch

from financial_pragmatic_ai.analysis.financial_signal_engine import (
    compute_risk_score,
    derive_signal,
)
from financial_pragmatic_ai.analysis.transcript_parser import parse_transcript
from financial_pragmatic_ai.models.conversation_attention_model import (
    INDEX_TO_SIGNAL,
    load_conversation_attention_model,
)
from financial_pragmatic_ai.models.financial_pragmatic_transformer_v2 import (
    FinancialPragmaticTransformer,
)
from financial_pragmatic_ai.models.finbert_intent_model import FinBERTIntentModel


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
FALLBACK_MODEL_PATH = MODELS_DIR / "pragmatic_transformer_trained.pt"
FINBERT_INTENT_PATH = MODELS_DIR / "finbert_intent.pt"
CONVERSATION_ATTENTION_PATH = MODELS_DIR / "conversation_attention.pt"

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

SPEAKER_ENCODING = {
    "CEO": [1.0, 0.0, 0.0],
    "CFO": [0.0, 1.0, 0.0],
    "ANALYST": [0.0, 0.0, 1.0],
}


def _speaker_vector(speaker: str) -> torch.Tensor:
    values = SPEAKER_ENCODING.get(speaker.upper(), [0.0, 0.0, 0.0])
    return torch.tensor(values, dtype=torch.float32)


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
        self.intent_model = None
        self.fallback_intent_model = None
        self._last_embeddings = []

        try:
            self.intent_model = FinBERTIntentModel(device=device)
            self.intent_model.load_weights(FINBERT_INTENT_PATH)
        except Exception as exc:
            print(f"[WARN] FinBERT intent model unavailable: {exc}")
            self.intent_model = None

        try:
            self.fallback_intent_model = FinancialPragmaticTransformer()
            self.fallback_intent_model.load_state_dict(
                torch.load(FALLBACK_MODEL_PATH, map_location=device)
            )
            self.fallback_intent_model.to(device)
            self.fallback_intent_model.eval()
        except Exception as exc:
            print(f"[WARN] Fallback intent model unavailable: {exc}")
            self.fallback_intent_model = None

        self.conversation_model = load_conversation_attention_model(
            model_path=CONVERSATION_ATTENTION_PATH,
            input_size=771,
            device=device,
        )
        if self.conversation_model is None:
            print("[WARN] Conversation attention model not found. Using rule-based fallback.")

    def predict_intent(self, text, speaker):
        if self.intent_model is not None:
            output = self.intent_model.predict(text)
            cls_embedding = output["embedding"].float()
            final_embedding = torch.cat([cls_embedding, _speaker_vector(speaker)], dim=-1)
            return {
                "intent": output["intent"],
                "logits": output["logits"],
                "embedding": final_embedding,
            }

        if self.fallback_intent_model is not None:
            intent = self.fallback_intent_model.predict(
                text,
                speaker=speaker,
                target_device=device,
            )
        else:
            intent = "GENERAL_UPDATE"

        fallback_embedding = torch.cat(
            [torch.zeros(768, dtype=torch.float32), _speaker_vector(speaker)],
            dim=-1,
        )
        return {
            "intent": intent,
            "logits": torch.zeros(4, dtype=torch.float32),
            "embedding": fallback_embedding,
        }

    def predict_conversation_signal(self, intents):
        if (
            self.conversation_model is not None
            and len(intents) > 0
            and len(self._last_embeddings) == len(intents)
        ):
            sequence = torch.stack(self._last_embeddings).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = self.conversation_model(sequence)
            pred = int(torch.argmax(logits, dim=-1).item())
            return INDEX_TO_SIGNAL[pred]

        score = compute_risk_score(intents)
        return derive_signal(score)

    def analyze(self, raw_text):
        print("🔥 NEW VERSION LOADED 🔥")
        segments = parse_transcript(raw_text)
        results = []
        embeddings = []

        for seg in segments:
            prediction = self.predict_intent(seg["text"], seg["speaker"])
            intent = prediction["intent"]
            text_lower = seg["text"].lower()

            print("MODEL INTENT:", intent)  # DEBUG

            # TEMP: DISABLE HEURISTIC OVERRIDE
            # if intent in ["GENERAL_UPDATE", "EXECUTIVE"]:
            #     ...

            results.append({
                "speaker": seg["speaker"],
                "text": seg["text"],
                "intent": intent,
            })
            embeddings.append(prediction["embedding"])

        # results = smooth_intents(results)
        self._last_embeddings = embeddings[: len(results)]

        print("[DEBUG] SAMPLE OUTPUT:")
        for result in results[:5]:
            print(result)

        return results
