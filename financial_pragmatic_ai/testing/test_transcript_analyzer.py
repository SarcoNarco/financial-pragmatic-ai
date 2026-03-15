from financial_pragmatic_ai.analysis.transcript_analyzer import TranscriptAnalyzer
from financial_pragmatic_ai.analysis.financial_signal_engine import FinancialSignalEngine

analyzer = TranscriptAnalyzer()

transcript = """
CEO: We plan to expand operations in Asia next quarter.
CFO: Margins may compress due to supply chain costs.
Analyst: How will this impact profitability going forward?
"""

results = analyzer.analyze(transcript)

for r in results:
    print(r)

engine = FinancialSignalEngine()

signals = engine.analyze(results)

print("\nDetected financial signals:\n")
print(signals)