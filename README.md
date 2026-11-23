# FakeID Detection System — Team Byte Force

## Gemini screening

Gemini provides an optional visual screening check. When `GEMINI_API_KEY` is
configured, each uploaded ID image is sent to Google's Gemini API. The model
only classifies whether the image appears to show an identity document; it is
not an authenticity check. Without the key, the app uses its existing local
checks.

Create a `.env` file in the project directory (it is ignored by Git) and add:

```text
GEMINI_API_KEY=your_rotated_api_key
GEMINI_MODEL=gemini-2.5-flash
```

Restart the app after changing `.env`. Never commit the key or paste it into
source code. If a key has been shared publicly or in a chat, revoke it and use
a newly generated key.

## OCR setup

The document text check requires the Tesseract OCR engine in addition to the
Python dependencies. On Windows, the app detects the default installation at
`C:\Program Files\Tesseract-OCR\tesseract.exe`. If Tesseract is installed
elsewhere or is not on `PATH`, set `TESSERACT_CMD` to the full executable path.
