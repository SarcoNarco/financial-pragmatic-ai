import re


SPEAKERS = ["CEO", "CFO", "ANALYST"]


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speaker_segments(text: str):
    """
    Extract structured segments from transcript.
    Handles:
    - CEO:
    - CFO:
    - Analyst:
    """
    pattern = r"(CEO|CFO|Analyst)[\s,:-]+(.+?)(?=(CEO|CFO|Analyst|$))"
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

    segments = []

    for match in matches:
        speaker = match[0].upper()
        content = match[1].strip()

        if len(content) > 20:
            segments.append({
                "speaker": speaker,
                "text": content,
            })

    return segments


def fallback_chunking(text: str):
    """
    If no speakers detected, split into meaningful chunks.
    """
    sentences = re.split(r"[.!?]", text)

    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 30:
            chunks.append({
                "speaker": "UNKNOWN",
                "text": sentence,
            })

    return chunks


def parse_transcript(text: str):
    text = clean_text(text)

    segments = extract_speaker_segments(text)

    if len(segments) == 0:
        segments = fallback_chunking(text)

    return segments
