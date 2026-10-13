"""Offline smoke test: verify the bundled sample file produces a non-empty summary."""
from pathlib import Path
from preprocessing import clean_text, count_words
from utils.fallback_summary import extractive_summary

source = Path(__file__).with_name("sample_test_document.txt")
text = clean_text(source.read_text(encoding="utf-8"))
summary = extractive_summary(text, target_words=100)
assert count_words(text) >= 30, "Sample input unexpectedly too short"
assert summary.strip(), "Summary is empty"
assert summary.strip() != text.strip(), "Summary is identical to source text"
print(f"PASS: {count_words(text)} input words -> {count_words(summary)} summary words")
print("\nGenerated summary:\n")
print(summary)
