from pathlib import Path
import re

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
            loaded = self.intent_model.load_weights(FINBERT_INTENT_PATH)
            if not loaded:
                print(
                    f"[WARN] FinBERT intent checkpoint not found at {FINBERT_INTENT_PATH}. "
                    "Falling back to pragmatic transformer."
                )
                self.intent_model = None
            else:
                print(f"[INFO] Loaded FinBERT intent weights from: {FINBERT_INTENT_PATH}")
        except RuntimeError as exc:
            print(f"[ERROR] FinBERT intent classifier load failed: {exc}")
            raise
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

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        cleaned = TranscriptAnalyzer._clean_text(text)
        if not cleaned:
            return []

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
            if sentence.strip()
        ]

        if len(sentences) <= 1:
            comma_split = [
                part.strip()
                for part in re.split(r"[;:]\s+|,\s+(?=[A-Z])", cleaned)
                if part.strip()
            ]
            if len(comma_split) > 1:
                sentences = comma_split

        if len(sentences) <= 1:
            words = cleaned.split()
            if len(words) > 45:
                sentences = [
                    " ".join(words[index : index + 24])
                    for index in range(0, len(words), 24)
                ]

        return sentences if sentences else [cleaned]

    @staticmethod
    def _chunk_sentences(sentences: list[str]) -> list[str]:
        if not sentences:
            return []

        n_sentences = len(sentences)
        if n_sentences <= 3:
            chunk_size = 1
        elif n_sentences <= 9:
            chunk_size = 2
        else:
            chunk_size = 3

        chunks = []
        for index in range(0, n_sentences, chunk_size):
            chunk = " ".join(sentences[index : index + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _segment_with_speaker_cues(self, raw_text: str) -> list[dict]:
        text = self._clean_text(raw_text)
        if not text:
            return []

        pattern = re.compile(r"(CEO|CFO|ANALYST|EXECUTIVE|OPERATOR)\s*:\s*", re.IGNORECASE)
        parts = pattern.split(text)
        if len(parts) < 3:
            return []

        segments: list[dict] = []
        preface = parts[0].strip()
        if preface:
            for chunk in self._chunk_sentences(self._split_sentences(preface)):
                segments.append({"speaker": "EXECUTIVE", "text": chunk})

        for idx in range(1, len(parts), 2):
            if idx + 1 >= len(parts):
                break
            speaker = parts[idx].upper()
            block_text = parts[idx + 1].strip()
            if not block_text:
                continue
            block_sentences = self._split_sentences(block_text)
            for chunk in self._chunk_sentences(block_sentences):
                segments.append({"speaker": speaker, "text": chunk})

        return segments

    def _build_segments(self, raw_text: str) -> list[dict]:
        segments = self._segment_with_speaker_cues(raw_text)

        if not segments:
            parsed = parse_transcript(raw_text)
            for parsed_segment in parsed:
                speaker = str(parsed_segment.get("speaker", "EXECUTIVE")).upper()
                text = str(parsed_segment.get("text", "")).strip()
                if not text:
                    continue

                sentences = self._split_sentences(text)
                for chunk in self._chunk_sentences(sentences):
                    segments.append({"speaker": speaker, "text": chunk})

        segments = [
            segment
            for segment in segments
            if segment["text"].strip() and len(segment["text"].strip()) >= 10
        ]

        if len(segments) <= 1:
            text = self._clean_text(raw_text)
            base_sentences = self._split_sentences(text)
            fallback_chunks = self._chunk_sentences(base_sentences)
            if len(fallback_chunks) > 1:
                segments = [{"speaker": "EXECUTIVE", "text": chunk} for chunk in fallback_chunks]
            elif fallback_chunks:
                segments = [{"speaker": "EXECUTIVE", "text": fallback_chunks[0]}]

        return segments

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
            "logits": torch.zeros(3, dtype=torch.float32),
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
        segments = self._build_segments(raw_text)
        print(f"[DEBUG] Parsed {len(segments)} segments")
        results = []
        embeddings = []

        for seg in segments:
            prediction = self.predict_intent(seg["text"], seg["speaker"])
            intent = prediction["intent"]
            print("MODEL INTENT:", intent)  # DEBUG

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
