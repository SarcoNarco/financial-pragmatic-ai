from financial_pragmatic_ai.analysis.financial_insight_generator import generate_insight
from financial_pragmatic_ai.analysis.timeline_signal_analyzer import TimelineSignalAnalyzer
from financial_pragmatic_ai.analysis.transcript_analyzer import TranscriptAnalyzer


class EarningsCallAnalyzer:

    def __init__(self, transcript_analyzer=None):
        self.transcript_analyzer = transcript_analyzer or TranscriptAnalyzer()
        self.timeline_analyzer = TimelineSignalAnalyzer(self.transcript_analyzer)

    def aggregate_signals(self, timeline_signals, fallback_signal="neutral"):

        signal_counts = {
            "risk": 0,
            "growth": 0,
            "neutral": 0
        }

        for item in timeline_signals:
            signal = item.get("signal", "neutral")
            if signal in signal_counts:
                signal_counts[signal] += 1

        if timeline_signals:
            dominant_signal = max(signal_counts, key=signal_counts.get)
        else:
            dominant_signal = fallback_signal if fallback_signal in signal_counts else "neutral"

        return {
            "dominant_signal": dominant_signal,
            "signal_counts": signal_counts
        }

    def analyze(self, transcript):

        base_analysis = self.transcript_analyzer.analyze(transcript)
        segments = base_analysis["segments"]

        timeline_signals = self.timeline_analyzer.analyze_timeline(segments)
        aggregation = self.aggregate_signals(
            timeline_signals,
            fallback_signal=base_analysis["financial_signal"]
        )

        insight = generate_insight(aggregation["dominant_signal"])

        return {
            "segments": segments,
            "timeline_signals": timeline_signals,
            "aggregation": aggregation,
            "insight": insight
        }
