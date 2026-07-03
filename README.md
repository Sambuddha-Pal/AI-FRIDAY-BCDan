# AI-Powered Customer Complaint Summary Generator

TCS AI Friday Season 2 — Manufacturing complaint triage tool. Users paste (or bulk-upload)
raw customer complaint text and get an AI-generated summary, category, severity, sentiment,
confidence score, and recommended action — powered by your GenAI Lab model endpoint.

## Features

- **Complaint Analysis** — paste a single complaint, get an instant structured AI summary you can edit and save.
- **Bulk Upload** — upload a CSV/JSON of complaints, pick the text column, and batch-analyze with a downloadable results CSV.
- **Analytics** — live charts (category breakdown, severity pie) built from whatever you've analyzed so far, plus sample data if you haven't analyzed anything yet.
- **History** — session log of every complaint you've analyzed, exportable to CSV.
- **Settings** — configure your GenAI Lab endpoint, API key, model name, and temperature, with a one-click "Test Connection" button.

## 1. Setup

```bash
cd complaint-ai
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configure your API credentials

You have two options — use whichever is easier:

**Option A — .env file (recommended)**

```bash
cp .env.example .env
```

Then edit `.env`:

```
GENAI_API_ENDPOINT=https://genailab.your-company.com/v1/chat/completions
GENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
GENAI_MODEL=gpt-4o-mini
```

**Option B — in-app Settings page**

Just run the app and fill in the endpoint/key/model on the **Settings** page. These are
stored for the session (not persisted to disk).

> The app expects an **OpenAI-compatible `chat/completions` endpoint**: it POSTs
> `{"model", "messages", "temperature"}` and sends the key as both `Authorization: Bearer <key>`
> and `api-key: <key>` headers (covers both OpenAI-style and Azure-style gateways). If your
> GenAI Lab gateway uses a different request/response shape, tell me the exact request format
> (sample curl or Postman call) and I'll adjust `call_genai_api()` in `app.py` to match.

## 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

## 4. Test your connection

Go to **Settings → Test Connection** and click "Send Test Complaint". If it's configured
correctly you'll see a parsed JSON response. If it fails, the exact error (auth, timeout,
bad URL, bad response shape) is shown so you can fix the endpoint/key.

## How the AI call works

`app.py` sends a system prompt instructing the model to return strict JSON with these fields:

```json
{
  "summary": "...",
  "category": "Battery | Display | Audio | Camera | Build Quality | Software/Firmware | Performance | Shipping/Packaging | Customer Service | Other",
  "severity": "Low | Medium | High | Critical",
  "sentiment": "Positive | Neutral | Negative | Very Negative",
  "confidence": 0-100,
  "recommended_action": "...",
  "keywords": ["..."]
}
```

The app parses this JSON (stripping markdown fences if the model adds them) and renders it
in the UI. If the API call fails for any reason, the app shows the error and falls back to a
clearly-labeled placeholder result so a demo never hard-crashes mid-presentation.

## Notes / things to double check before your demo

- **Never commit `.env`** or your real API key — `.env.example` is the template only.
- If your GenAI Lab endpoint truncates or rate-limits, the "Max rows to analyze" slider on
  Bulk Upload lets you cap API usage.
- Data privacy: the problem statement calls for anonymizing customer PII before sending text
  to the model. This starter doesn't include a PII-scrubbing step — let me know if you want
  one added (e.g. regex-based redaction of names/emails/phone numbers before the API call).
