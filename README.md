# Secret Sauce

Streamlit app for OTDR duplicate classification. Upload `.zip`, `.sor`, or
`.json` files and download a PDF report.

## Local

```bash
pip install -r requirements.txt
# macOS: brew install pango   (needed by WeasyPrint)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # set your password
./run.sh
```

## Streamlit Cloud

1. Push this repo to GitHub.
2. On <https://share.streamlit.io>, point a new app at `app.py`.
3. In the app's **Settings → Secrets**, set:
   ```toml
   app_password = "your-password"
   ```
4. The `packages.txt` and `requirements.txt` install WeasyPrint's native
   deps automatically — no browser is required for PDF rendering.
