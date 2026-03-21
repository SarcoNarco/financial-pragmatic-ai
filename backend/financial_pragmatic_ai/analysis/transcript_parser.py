import re


COMMON_EXECUTIVE_KEYWORDS = [
    "revenue", "growth", "strategy", "market", "expansion", "expand",
    "performance", "guidance"
]

CFO_KEYWORDS = [
    "margin", "cost", "expense", "ebitda", "operating income",
    "cash flow", "balance sheet"
]

ANALYST_KEYWORDS = [
    "question", "how", "what", "why", "could you", "can you"
]


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_speaker_blocks(text: str):
    """
    Extract blocks using NAME: pattern
    Works for:
    Doug McMillon:
    John Smith:
    Operator:
    """
    pattern = (
        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}(?:\s[A-Z]{2,})*):\s*(.+?)"
        r"(?=(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|[A-Z]{2,}(?:\s[A-Z]{2,})*):|$)"
    )

    matches = re.findall(pattern, text, re.DOTALL)

    blocks = []

    for name, content in matches:
        content = content.strip()

        if len(content) < 40:
            continue

        blocks.append({
            "name": name.strip(),
            "text": content
        })

    return blocks


def infer_role(name: str, text: str):
    text_lower = text.lower()
    name_lower = name.lower()

    if "operator" in name_lower:
        return "OPERATOR"

    if "analyst" in name_lower:
        return "ANALYST"

    # Analyst detection (questions dominate Q&A)
    if any(q in text_lower for q in ANALYST_KEYWORDS):
        return "ANALYST"

    # CFO detection (financial-heavy language)
    if any(k in text_lower for k in CFO_KEYWORDS):
        return "CFO"

    # CEO / Exec detection (strategy-heavy language)
    if any(k in text_lower for k in COMMON_EXECUTIVE_KEYWORDS):
        return "CEO"

    return "EXECUTIVE"


def fallback_chunking(text: str):
    """
    If no speaker blocks found -> split into semantic chunks
    """
    sentences = re.split(r"[.!?]", text)

    chunks = []
    buffer = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        buffer += sentence + ". "

        if len(buffer) > 120:
            chunks.append({
                "name": "UNKNOWN",
                "text": buffer.strip()
            })
            buffer = ""

    if len(buffer.strip()) > 30:
        chunks.append({
            "name": "UNKNOWN",
            "text": buffer.strip()
        })

    return chunks


def parse_transcript(text: str):
    text = clean_text(text)

    blocks = extract_speaker_blocks(text)

    if len(blocks) == 0:
        blocks = fallback_chunking(text)

    segments = []

    for block in blocks:
        role = infer_role(block["name"], block["text"])

        segments.append({
            "speaker": role,
            "name": block["name"],
            "text": block["text"]
        })

    print(f"[DEBUG] Parsed {len(segments)} segments")

    return segments
