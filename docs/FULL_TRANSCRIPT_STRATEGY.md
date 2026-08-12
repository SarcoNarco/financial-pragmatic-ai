# Smart Full-Transcript Analysis

## Purpose

The hosted Financial Pragmatic AI demo accepts long earnings-call transcripts without attempting exhaustive synchronous inference on every segment. This is a performance strategy for the Railway demo, not a claim of full-document analysis.

## Analysis Modes

| Mode | When used | Behavior |
| --- | --- | --- |
| `standard` | Transcript is at or below `MAX_DIRECT_TRANSCRIPT_CHARS`. | Existing full analysis path runs on all parsed segments. |
| `sampled_full_transcript` | Transcript is above `MAX_DIRECT_TRANSCRIPT_CHARS`. | The full transcript is segmented, then up to the configured representative segment budget is analyzed. If the transcript is already within budget, every parsed segment is analyzed and `sampled` is `false`. |

The default direct-analysis threshold is 20,000 characters and the default sampled budget is 32 segments. An absolute `MAX_FULL_TRANSCRIPT_CHARS` safety cap, defaulting to 250,000 characters, still protects the synchronous hosted endpoint.

## Representative Sampling

For sampled mode, the backend:

1. Parses the complete transcript with the existing segmentation logic.
2. Reserves roughly one third of the segment budget for early, middle, and late transcript coverage.
3. Fills the remaining budget with deterministic keyword-ranked segments covering growth, risk, finance, and Q&A language.
4. Removes duplicates and restores original transcript order before batched inference.

The score, signal, drivers, distributions, and timeline therefore describe the analyzed representative segments, not every segment in the source document.

## Response Metadata

`POST /analyze` and `POST /upload` include these backward-compatible fields:

| Field | Meaning |
| --- | --- |
| `analysis_mode` | `standard` or `sampled_full_transcript`. |
| `sampled` | Whether fewer segments were analyzed than were parsed. |
| `segments_total` | Segments parsed from the complete transcript. |
| `segments_analyzed` | Segments sent to inference. |
| `segment_budget` | Configured budget for the long-transcript path, otherwise `null`. |
| `sampling_note` | Honest explanation when representative sampling was used. |
| `transcript_chars` | Original transcript character count. |

The Vercel frontend shows a calm result note only when `sampled` is true:

> Sampled full-transcript analysis: analyzed X of Y transcript segments for hosted-demo performance.

The current Supabase history schema intentionally stores the existing core result fields only. Sampling metadata is visible for the immediate result but is not retained when an older history item is re-opened; this avoids a schema migration and keeps deployed history compatible.

## Document Uploads

`POST /upload` converts supported documents to plain transcript text before using the exact same analysis path as `POST /analyze`. Supported formats are TXT, PDF, and DOCX. PDF extraction uses the existing text parser and DOCX extraction reads non-empty paragraphs in source order; neither path performs OCR.

Legacy Microsoft Word `.doc` files are intentionally unsupported. Converting them reliably needs platform-specific tooling that is unsuitable for the Railway deployment. The upload response includes the extracted `transcript` so the frontend can preserve its existing Supabase history behavior without a schema change.

## Configuration

```text
MAX_DIRECT_TRANSCRIPT_CHARS=20000
FULL_TRANSCRIPT_SEGMENT_BUDGET=32
MAX_FULL_TRANSCRIPT_CHARS=250000
```

`MAX_TRANSCRIPT_CHARS` remains a legacy alias for `MAX_DIRECT_TRANSCRIPT_CHARS`. It no longer rejects that threshold; it selects the sampled path. Railway should keep the default budget modest and avoid raising it without measuring one representative request.

## Benchmark A Transcript File

Run one request locally or against a deliberate deployment test target:

```bash
BASE_URL=http://localhost:8000 \
python backend/scripts/benchmark_transcript_file.py path/to/transcript.txt
```

The script prints file size, approximate word count, HTTP status, elapsed time, core prediction fields, and the sampling metadata. It never retries. Do not repeatedly benchmark huge transcripts against Railway because CPU inference remains costly.

## Limitation And Future Path

This is not exhaustive full-transcript analysis. A production workflow that needs every segment should use asynchronous jobs, batch processing, progress status, persisted full results, and potentially RAG-backed citations rather than extending a synchronous hosted-demo request.
