# AI-Based Multilingual Document Summarizer

Streamlit app for PDF, DOCX, TXT, and pasted-text summarization using
`csebuetnlp/mT5_multilingual_XLSum` from Hugging Face.

## Windows setup (PowerShell)

Open PowerShell in this project folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run this in that terminal first:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Then open the `Local URL` shown by Streamlit, normally `http://localhost:8501`.

## First run

The first AI summary request downloads the multilingual model from Hugging Face.
This requires a working internet connection and enough disk space/RAM; model
initialization can take several minutes on a laptop. Subsequent requests reuse
the cached model. Do not start the app with `python app.py`; always use
`python -m streamlit run app.py`.

If the AI model cannot load or inference fails, the app explicitly warns you
and returns an offline extractive fallback summary. That fallback is labelled
as non-Transformer output and does not pretend the AI model worked.

## Supported files

- PDF (text-based PDFs; scanned/image-only PDFs need OCR)
- DOCX (paragraphs and tables)
- TXT (UTF-8)
- Pasted text

## Important

This package has been syntax-checked and its document extraction/fallback path
can be tested locally. The actual Hugging Face Transformer inference requires
installing the dependencies and downloading the model; it cannot be claimed
as tested in an environment where those dependencies/model weights are absent.
