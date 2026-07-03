import os
import json
import time
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="AI Customer Complaint Summary Generator",
    page_icon="🏭",
    layout="wide"
)

# ----------------------------------------
# Session State Defaults (API config, history, etc.)
# ----------------------------------------
if "api_endpoint" not in st.session_state:
    st.session_state.api_endpoint = os.getenv("GENAI_API_ENDPOINT", "https://genailab.../chat/completions")

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GENAI_API_KEY", "")

if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("GENAI_MODEL", "gpt-4o-mini")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: complaint, result

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None


# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("🏭 Manufacturing AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Complaint Analysis",
        "Bulk Upload",
        "Analytics",
        "History",
        "Settings"
    ]
)

st.sidebar.markdown("---")

if st.session_state.api_key:
    st.sidebar.success("✅ AI Model Connected")
else:
    st.sidebar.warning("⚠️ No API Key set — go to Settings")

st.sidebar.info(
    """
    **TCS AI Friday 2026**

    AI-Powered Customer Complaint
    Summary Generator
    """
)

# ----------------------------------------
# Core AI Function — calls the GenAI Lab endpoint
# ----------------------------------------

SYSTEM_PROMPT = """You are an expert quality-assurance analyst for a manufacturing company.
You read raw, unstructured customer complaint text (which may be messy, informal, or contain
typos) and produce a structured analysis to help Quality and Customer Service teams triage issues
quickly.

Always respond with STRICT JSON ONLY — no markdown fences, no commentary — matching this schema:

{
  "summary": "one or two sentence concise summary of the complaint",
  "category": "one of: Battery, Display, Audio, Camera, Build Quality, Software/Firmware, Performance, Shipping/Packaging, Customer Service, Other",
  "severity": "one of: Low, Medium, High, Critical",
  "sentiment": "one of: Positive, Neutral, Negative, Very Negative",
  "confidence": "integer percentage 0-100 representing your confidence in this classification",
  "recommended_action": "a specific, actionable recommendation for the internal team (e.g. which team to escalate to, whether a replacement/refund is warranted, whether it's a safety issue)",
  "keywords": ["short", "list", "of", "key", "defect", "terms"]
}
"""


def call_genai_api(complaint_text: str, timeout: int = 30) -> dict:
    """Calls the configured GenAI Lab chat-completions endpoint and parses a
    structured JSON result out of the model's reply. Raises on failure so the
    caller can decide how to surface the error."""

    endpoint = st.session_state.api_endpoint.strip()
    api_key = st.session_state.api_key.strip()
    model = st.session_state.model_name.strip()
    temperature = st.session_state.temperature

    if not endpoint or not api_key:
        raise ValueError("API endpoint or API key is not configured. Set them in Settings.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Some internal gateways (e.g. Azure-style) expect this header instead of/alongside Bearer.
        "api-key": api_key,
    }

    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this customer complaint:\n\n{complaint_text}"}
        ],
    }

    response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    # Standard OpenAI-compatible response shape
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        # Fall back: some gateways return {"content": "..."} or {"result": "..."}
        content = data.get("content") or data.get("result") or json.dumps(data)

    content = content.strip()
    # Strip markdown code fences if the model added them anyway
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Last resort: try to locate the first {...} block
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end + 1])
        else:
            raise ValueError(f"Could not parse AI response as JSON:\n{content}")

    return parsed


def analyze_complaint(text: str) -> dict:
    """Wrapper used by the UI. Calls the real API, and falls back to a clearly
    labeled dummy result if the call fails, so the demo never hard-crashes."""
    try:
        result = call_genai_api(text)
        return {
            "Summary": result.get("summary", "N/A"),
            "Category": result.get("category", "Other"),
            "Severity": result.get("severity", "Medium"),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Confidence": f'{result.get("confidence", "N/A")}%',
            "Recommended Action": result.get("recommended_action", "N/A"),
            "Keywords": result.get("keywords", []),
            "source": "ai"
        }
    except Exception as e:
        st.error(f"⚠️ AI call failed, showing fallback result. Error: {e}")
        return {
            "Summary": "(fallback) Could not reach AI model — please check API settings.",
            "Category": "Other",
            "Severity": "Medium",
            "Sentiment": "Neutral",
            "Confidence": "0%",
            "Recommended Action": "Check API endpoint/key in Settings and retry.",
            "Keywords": [],
            "source": "fallback"
        }


# ========================================
# Complaint Analysis
# ========================================

if page == "Complaint Analysis":

    st.title("🏭 AI Complaint Summary Generator")

    col1, col2 = st.columns([2, 1])

    with col1:
        complaint = st.text_area(
            "Customer Complaint",
            height=250,
            placeholder="Paste customer complaint here..."
        )

    with col2:
        total = len(st.session_state.history)
        high_sev = sum(1 for h in st.session_state.history if h["result"]["Severity"] in ("High", "Critical"))
        st.metric("Complaints Analyzed", total)
        st.metric("High/Critical Severity", high_sev)
        st.metric("Model", st.session_state.model_name)

    if st.button("🚀 Generate AI Summary", use_container_width=True):

        if complaint.strip():

            with st.spinner("Analyzing complaint with AI..."):
                result = analyze_complaint(complaint)

            if result["source"] == "ai":
                st.success("Analysis Complete")
            st.session_state.history.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "complaint": complaint,
                "result": result
            })

            st.subheader("📋 AI Summary")
            st.write(result["Summary"])

            c1, c2, c3 = st.columns(3)
            c1.metric("Category", result["Category"])
            c2.metric("Severity", result["Severity"])
            c3.metric("Sentiment", result["Sentiment"])

            st.metric("Confidence", result["Confidence"])

            if result["Keywords"]:
                st.write("**Keywords:** " + ", ".join(result["Keywords"]))

            st.subheader("Recommended Action")
            st.info(result["Recommended Action"])

            st.subheader("Edit Summary")
            edited = st.text_area(
                "",
                value=result["Summary"],
                height=120,
                key="edited_summary"
            )

            if st.button("Save Edited Summary"):
                st.session_state.history[-1]["result"]["Summary"] = edited
                st.success("Saved.")
        else:
            st.warning("Please paste a complaint before generating a summary.")

# ========================================
# Bulk Upload
# ========================================

elif page == "Bulk Upload":

    st.title("📂 Bulk Complaint Upload")

    st.caption("Upload a CSV or JSON file with a column/field named **complaint** (or similar text field).")

    uploaded = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"]
    )

    if uploaded:

        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_json(uploaded)

        st.session_state.bulk_df = df
        st.success(f"File uploaded — {len(df)} rows")
        st.dataframe(df, use_container_width=True)

        # Try to guess which column holds complaint text
        text_col_candidates = [c for c in df.columns if c.lower() in ("complaint", "text", "complaint_text", "description")]
        text_col = st.selectbox(
            "Which column contains the complaint text?",
            options=list(df.columns),
            index=(df.columns.get_loc(text_col_candidates[0]) if text_col_candidates else 0)
        )

        max_rows = st.slider("Max rows to analyze (to control API usage)", 1, min(len(df), 200), min(len(df), 20))

        if st.button("Analyze Complaints"):
            progress = st.progress(0)
            results = []
            subset = df.head(max_rows)
            for i, row in enumerate(subset.itertuples(index=False)):
                text = str(getattr(row, text_col) if hasattr(row, text_col) else subset.iloc[i][text_col])
                res = analyze_complaint(text)
                results.append({
                    "Complaint": text,
                    "Summary": res["Summary"],
                    "Category": res["Category"],
                    "Severity": res["Severity"],
                    "Sentiment": res["Sentiment"],
                    "Confidence": res["Confidence"],
                    "Recommended Action": res["Recommended Action"]
                })
                progress.progress((i + 1) / len(subset))

            result_df = pd.DataFrame(results)
            st.session_state.bulk_results = result_df
            st.success("Processing complete")

        if st.session_state.bulk_results is not None:
            st.subheader("Results")
            st.dataframe(st.session_state.bulk_results, use_container_width=True)

            csv_bytes = st.session_state.bulk_results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_bytes,
                file_name="complaint_analysis_results.csv",
                mime="text/csv"
            )

# ========================================
# Analytics
# ========================================

elif page == "Analytics":

    st.title("📊 Complaint Analytics")

    # Prefer real history/bulk data if available, else show sample data
    records = []
    for h in st.session_state.history:
        records.append(h["result"])
    if st.session_state.bulk_results is not None:
        records.extend(st.session_state.bulk_results.to_dict("records"))

    if records:
        df = pd.DataFrame(records)
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
    else:
        st.info("No analyzed complaints yet — showing sample data. Analyze some complaints to see live analytics.")
        cat_counts = pd.DataFrame({
            "Category": ["Battery", "Display", "Audio", "Battery", "Camera"],
            "Count": [40, 20, 12, 15, 18]
        }).groupby("Category", as_index=False).sum()

    fig = px.bar(
        cat_counts,
        x="Category",
        y="Count",
        color="Category",
        title="Complaint Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPIs")
    c1, c2, c3 = st.columns(3)
    total_complaints = len(records) if records else 1520
    critical = sum(1 for r in records if r.get("Severity") in ("High", "Critical")) if records else 32
    resolved = total_complaints - critical if records else 1425
    c1.metric("Complaints", total_complaints)
    c2.metric("Resolved (est.)", resolved)
    c3.metric("Critical", critical)

    if records:
        sev_counts = df["Severity"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        fig2 = px.pie(sev_counts, names="Severity", values="Count", title="Severity Breakdown")
        st.plotly_chart(fig2, use_container_width=True)

# ========================================
# History
# ========================================

elif page == "History":

    st.title("📜 Complaint History")

    if st.session_state.history:
        hist_rows = [{
            "Timestamp": h["timestamp"],
            "Complaint": h["complaint"][:80] + ("..." if len(h["complaint"]) > 80 else ""),
            "Category": h["result"]["Category"],
            "Severity": h["result"]["Severity"],
            "Sentiment": h["result"]["Sentiment"],
        } for h in reversed(st.session_state.history)]

        history_df = pd.DataFrame(hist_rows)
        st.dataframe(history_df, use_container_width=True)

        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download History as CSV", data=csv_bytes, file_name="complaint_history.csv", mime="text/csv")

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No complaints analyzed yet in this session. Go to 'Complaint Analysis' to get started.")

# ========================================
# Settings
# ========================================

elif page == "Settings":

    st.title("⚙️ Settings")

    st.markdown("Configure the connection to your GenAI Lab API endpoint below.")

    st.session_state.api_endpoint = st.text_input(
        "GenAI API Endpoint",
        value=st.session_state.api_endpoint,
        help="Full URL of the chat/completions endpoint, e.g. https://genailab.example.com/v1/chat/completions"
    )

    st.session_state.api_key = st.text_input(
        "API Key",
        value=st.session_state.api_key,
        type="password"
    )

    st.session_state.model_name = st.text_input(
        "Model Name",
        value=st.session_state.model_name,
        help="The model identifier expected by your GenAI Lab gateway."
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.temperature
    )

    if st.button("💾 Save Settings"):
        st.success("Settings saved for this session.")

    st.markdown("---")
    st.subheader("🔌 Test Connection")

    if st.button("Send Test Complaint"):
        with st.spinner("Testing connection..."):
            try:
                result = call_genai_api("The screen flickers randomly and the battery drains within two hours.")
                st.success("Connection successful! Sample response:")
                st.json(result)
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("---")
    st.caption(
        "Tip: instead of typing the API key here every time, you can set environment variables "
        "`GENAI_API_ENDPOINT`, `GENAI_API_KEY`, and `GENAI_MODEL` (e.g. in a `.env` file) before "
        "launching the app, and they'll be used as defaults."
    )
