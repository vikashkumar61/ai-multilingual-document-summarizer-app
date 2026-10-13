"""Small offline extractive fallback used only when the Transformer cannot load."""
import math
import re
from collections import Counter

from preprocessing import count_words

_STOPWORDS = set("""
a an the and or but if then than so because as at by for from in into of on onto
to up with without is are was were be been being am do does did has have had can could
should would may might will this that these those it its they them their he she we you
i me my our your not no yes also very more most such which who whom what when where why how
""".split())

def extractive_summary(text, target_words=100):
    """Select informative original sentences; never invent factual claims."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        raise ValueError("No text is available to summarize.")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।。！？])\s+", text) if s.strip()]
    if not sentences:
        return text[:max(200, target_words * 8)]
    words = re.findall(r"[^\W_]+", text.lower(), flags=re.UNICODE)
    counts = Counter(w for w in words if len(w) > 2 and w not in _STOPWORDS)
    max_count = max(counts.values(), default=1)
    scored = []
    for i, sentence in enumerate(sentences):
        tokens = re.findall(r"[^\W_]+", sentence.lower(), flags=re.UNICODE)
        if not tokens:
            continue
        score = sum(counts.get(w, 0) / max_count for w in tokens if len(w) > 2)
        score /= math.sqrt(len(tokens))
        score *= 1.0 + 0.12 / (i + 1)
        scored.append((score, i, sentence))
    if not scored:
        return " ".join(sentences[:3])
    chosen, total = [], 0
    for score, i, sentence in sorted(scored, reverse=True):
        n = count_words(sentence)
        if chosen and total + n > target_words:
            continue
        chosen.append((i, sentence))
        total += n
        if total >= target_words:
            break
    if not chosen:
        chosen = [(scored[0][1], scored[0][2])]
    return " ".join(sentence for _, sentence in sorted(chosen))
