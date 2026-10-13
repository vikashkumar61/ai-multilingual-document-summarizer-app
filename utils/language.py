"""Language detection with a safe fallback when langdetect is unavailable."""
def detect_language(text):
    sample = (text or "").strip()
    if not sample:
        return "Unknown"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        code = detect(sample)
        names = {
            "en": "English", "hi": "Hindi", "ar": "Arabic", "ur": "Urdu",
            "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese",
            "ru": "Russian", "zh-cn": "Chinese", "zh-tw": "Chinese",
            "ja": "Japanese", "ko": "Korean", "bn": "Bengali", "pa": "Punjabi",
            "ta": "Tamil", "te": "Telugu", "mr": "Marathi", "gu": "Gujarati",
            "it": "Italian", "tr": "Turkish", "nl": "Dutch",
        }
        return names.get(code.lower(), code.upper())
    except Exception:
        # Lightweight script-based fallback; summarization still uses the input text.
        if any("\\u0600" <= ch <= "\\u06ff" for ch in sample):
            return "Arabic"
        if any("\\u0900" <= ch <= "\\u097f" for ch in sample):
            return "Hindi"
        if any("\\u0980" <= ch <= "\\u09ff" for ch in sample):
            return "Bengali"
        if any("\\u4e00" <= ch <= "\\u9fff" for ch in sample):
            return "Chinese"
        if any("\\u3040" <= ch <= "\\u30ff" for ch in sample):
            return "Japanese"
        return "English (fallback)"
