import os
import json
import time
import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import io

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
    # NOTE: this should be the BASE url (no /chat/completions suffix) —
    # langchain_openai.ChatOpenAI appends the path itself.
    st.session_state.api_endpoint = os.getenv("GENAI_API_ENDPOINT", "https://genailab.tcs.in/")

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GENAI_API_KEY", "sk-bgfhTgskBe3Skyz9ae5mhw")

if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("GENAI_MODEL", "genailab-maas-gpt-35-turbo")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: complaint, result





# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("🏭 Manufacturing AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Complaint Analysis",
        "History"
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



KNOWLEDGE_BASE = [
    {
        "id": "KB001",
        "category": "Battery",
        "title": "Rapid battery drain (first 48 hours)",
        "content": "Units from batch Q1-2026 have a known firmware bug in the power "
                    "management module causing battery drain to 0% within 2-3 hours of "
                    "normal use. Fix: update firmware to v2.3.1+ via Settings > System "
                    "Update. If already updated and issue persists, escalate to Hardware "
                    "QA for battery cell inspection. Replacement is approved if device is "
                    "under 12 months old and battery health reports below 80%."
    },
    {
        "id": "KB002",
        "category": "Battery",
        "title": "Device won't hold charge / dies overnight",
        "content": "Typically a swollen or degraded battery cell rather than software. "
                    "Advise customer to stop using the device immediately (swelling is a "
                    "fire/safety risk) and bring it in for inspection. This is a Critical "
                    "severity, safety-flagged issue — escalate to Safety/Field Quality team, "
                    "not standard support queue."
    },
    {
        "id": "KB003",
        "category": "Display",
        "title": "Screen flickering or random black flashes",
        "content": "Common cause is a loose display ribbon cable from drop/impact, or a "
                    "known driver IC defect in units manufactured before March 2026. "
                    "Resolution: run built-in diagnostic (Settings > Device Health > "
                    "Display Test). If flicker reproduces, it's a hardware replacement "
                    "under warranty, not a software fix."
    },
    {
        "id": "KB004",
        "category": "Display",
        "title": "Dead pixels or discoloration",
        "content": "Company policy: 3 or more dead/stuck pixels, or any visible "
                    "discoloration/banding, qualifies for free screen replacement or unit "
                    "exchange within the first 90 days, no diagnostic required."
    },
    {
        "id": "KB005",
        "category": "Audio",
        "title": "Crackling, distorted, or muffled speaker sound",
        "content": "Usually debris in the speaker mesh or a loose speaker connector after "
                    "drop. First response: advise a compressed-air clean of the speaker "
                    "grille. If unresolved, this is a low-cost repair (speaker module swap), "
                    "not a full replacement — route to standard repair queue."
    },
    {
        "id": "KB006",
        "category": "Camera",
        "title": "Blurry photos / camera fails to focus",
        "content": "Check for a protective film left on the lens (very common false "
                    "report) before escalating. If confirmed hardware issue, it typically "
                    "traces to a misaligned autofocus motor and requires a camera module "
                    "replacement under warranty."
    },
    {
        "id": "KB007",
        "category": "Build Quality",
        "title": "Cracked casing / hinge failure without drop",
        "content": "If the customer states no drop or impact occurred, treat as a "
                    "manufacturing defect (materials fatigue), not accidental damage. "
                    "This is covered under standard warranty — do not apply accidental "
                    "damage exclusions or charge a fee."
    },
    {
        "id": "KB008",
        "category": "Software/Firmware",
        "title": "Random reboots, freezing, or app crashes",
        "content": "First step is always a full firmware update plus factory reset with "
                    "backup, which resolves the majority of reported cases. If it "
                    "persists after a clean reset, escalate to the Firmware Engineering "
                    "team with the device's crash logs."
    },
    {
        "id": "KB009",
        "category": "Performance",
        "title": "Device runs slow / overheats under normal use",
        "content": "Ask which apps are installed — a small number of known third-party "
                    "apps have memory leaks that cause thermal throttling. If overheating "
                    "occurs even in a clean/reset state, escalate as a thermal-management "
                    "hardware defect (High severity)."
    },
    {
        "id": "KB010",
        "category": "Shipping/Packaging",
        "title": "Item arrived damaged or box tampered with",
        "content": "Do not treat as a product defect. Route directly to the Logistics/"
                    "Carrier claims team with photos of the packaging. Customer is "
                    "entitled to an immediate replacement shipment while the carrier claim "
                    "is processed separately."
    },
    {
        "id": "KB011",
        "category": "Customer Service",
        "title": "Complaint about support responsiveness or rude staff",
        "content": "Not a product/technical issue. This should be tagged for the Customer "
                    "Experience team for service-quality review, and typically warrants a "
                    "goodwill gesture (discount/voucher) independent of any technical "
                    "resolution."
    },
]


@st.cache_resource(show_spinner="Building knowledge base TF-IDF index...")
def build_knowledge_base_index():
    """RAG STEP 1: Knowledge Data -> TF-IDF vectors -> local vector store.
    Fits a TfidfVectorizer on all KB entries once and caches both the
    vectorizer and the resulting matrix. Purely local — no model download,
    no network call of any kind. Cached with st.cache_resource so this only
    runs once per app session, not on every complaint analyzed."""
    texts = [f"{kb['category']} - {kb['title']}: {kb['content']}" for kb in KNOWLEDGE_BASE]

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_context(query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
    """RAG STEP 3: Retriever function.
    Query -> TF-IDF vector (using the vectorizer fit on the KB) -> cosine
    similarity search against the KB matrix -> returns the top-k most
    relevant knowledge base entries (filtered by a minimum similarity score
    so irrelevant complaints don't drag in noise).

    Note: min_score is lower than a typical embedding-similarity threshold
    because TF-IDF cosine scores run lower than dense-embedding scores for
    short queries. Tune this if you find matches too loose/strict.

    Retrieval failures are caught here and surfaced as a warning rather than
    raised, so the LLM call can still proceed without RAG context instead of
    hard-failing."""
    try:
        vectorizer, matrix = build_knowledge_base_index()

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, matrix)[0]

        ranked_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in ranked_indices:
            score = scores[idx]
            if score < min_score:
                continue
            entry = dict(KNOWLEDGE_BASE[idx])
            entry["score"] = float(score)
            results.append(entry)
        return results
    except Exception as e:
        st.warning(f"⚠️ Knowledge base retrieval unavailable, continuing without it. Error: {e}")
        return []


# ----------------------------------------
# Core AI Function — calls the GenAI Lab endpoint via langchain_openai
# ----------------------------------------

SYSTEM_PROMPT = """You are an expert quality-assurance analyst for a manufacturing company.
You read raw, unstructured customer complaint text (which may be messy, informal, or contain
typos) and produce a structured analysis to help Quality and Customer Service teams triage issues
quickly.

You will sometimes be given "Relevant internal knowledge base context" retrieved from the
company's QA/engineering knowledge base above the complaint. If any of it applies to this
complaint, use it to ground your category, severity, and especially your recommended_action
(e.g. known root causes, escalation paths, warranty/replacement policy). If none of the
retrieved context is relevant, ignore it and rely on your own judgment — never force a fit.

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


def get_llm() -> ChatOpenAI:
    """Builds a ChatOpenAI client pointed at the configured GenAI Lab endpoint.
    Uses a plain httpx.Client with SSL verification disabled (needed for some
    internal/self-signed corporate gateways). This is unrelated to the RAG
    change above — it's specifically for reaching your GenAI Lab endpoint."""

    endpoint = st.session_state.api_endpoint.strip()
    api_key = st.session_state.api_key.strip()
    model = st.session_state.model_name.strip()
    temperature = st.session_state.temperature

    if not endpoint or not api_key:
        raise ValueError("API endpoint or API key is not configured. Set them in Settings.")

    http_client = httpx.Client(verify=False)

    return ChatOpenAI(
        base_url=endpoint,
        model=model,
        api_key=api_key,
        temperature=temperature,
        http_client=http_client,
    )


def build_rag_prompt(complaint_text: str, retrieved: list[dict]) -> str:
    """RAG STEP 4: Inject retrieved knowledge base context into the LLM prompt."""
    if retrieved:
        context_block = "\n\n".join(
            f"[{r['id']} | {r['category']} | match {r['score']:.2f}] {r['title']}\n{r['content']}"
            for r in retrieved
        )
    else:
        context_block = "(No sufficiently relevant knowledge base entries were found for this complaint.)"

    return (
        "Relevant internal knowledge base context:\n\n"
        f"{context_block}\n\n"
        "---\n\n"
        f"Analyze this customer complaint:\n\n{complaint_text}"
    )


def call_genai_api(complaint_text: str, timeout: int = 30) -> tuple[dict, list[dict]]:
    """Calls the configured GenAI Lab endpoint via langchain_openai and parses
    a structured JSON result out of the model's reply. Raises on failure so
    the caller can decide how to surface the error.

    RAG pipeline: TF-IDF-vectorize the complaint -> cosine similarity search
    against the local KB matrix -> inject the top matches into the prompt ->
    LLM generates the final, grounded answer. Returns (parsed_result,
    retrieved_kb_entries) so the UI can show which knowledge base entries
    informed the answer."""

    llm = get_llm()

    # STEP 3 (retrieval) + STEP 4 (injection)
    retrieved = retrieve_context(complaint_text, k=3)
    user_prompt = build_rag_prompt(complaint_text, retrieved)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    content = (response.content or "").strip()

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

    return parsed, retrieved


def analyze_complaint(text: str) -> dict:
    """Wrapper used by the UI. Calls the real API (with RAG retrieval baked in),
    and falls back to a clearly labeled dummy result if the call fails, so the
    demo never hard-crashes."""
    try:
        result, retrieved = call_genai_api(text)
        return {
            "Summary": result.get("summary", "N/A"),
            "Category": result.get("category", "Other"),
            "Severity": result.get("severity", "Medium"),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Confidence": f'{result.get("confidence", "N/A")}%',
            "Recommended Action": result.get("recommended_action", "N/A"),
            "Keywords": result.get("keywords", []),
            "RetrievedContext": retrieved,
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
            "RetrievedContext": [],
            "source": "fallback"
        }

def generate_pdf(result_dict, complaint_text):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("AI Complaint Analysis Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>Complaint:</b><br/>{complaint_text}", styles["BodyText"]))
    story.append(Spacer(1, 12))

    for key in ["Summary", "Category", "Severity", "Sentiment", "Confidence", "Recommended Action"]:
        story.append(Paragraph(f"<b>{key}:</b> {result_dict.get(key)}", styles["BodyText"]))
        story.append(Spacer(1, 6))

    keywords = result_dict.get("Keywords", [])
    story.append(Paragraph(f"<b>Keywords:</b> {', '.join(keywords)}", styles["BodyText"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

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

            pdf_buffer = generate_pdf(result, complaint)

            st.download_button(
            label="📄 Download Report as PDF",
            data=pdf_buffer,
            file_name=f"complaint_report_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
)

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

            retrieved = result.get("RetrievedContext", [])
            with st.expander(f"📚 Knowledge Base Context Used ({len(retrieved)} match(es))"):
                if retrieved:
                    for r in retrieved:
                        st.markdown(f"**[{r['id']}] {r['title']}** · *{r['category']}* · similarity {r['score']:.2f}")
                        st.caption(r["content"])
                else:
                    st.caption("No relevant knowledge base entries were retrieved for this complaint.")

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
# History
# ========================================


if page == "History":

    st.title("📜 Complaint History")

    if not st.session_state.history:
        st.info("No reports generated yet.")
    else:
        for i, item in enumerate(reversed(st.session_state.history)):
            result = item["result"]
            with st.expander(f"Report {len(st.session_state.history)-i} • {item['timestamp']}"):
                st.write("### Original Complaint")
                st.write(item["complaint"])

                st.write("### AI Summary")
                st.write(result["Summary"])

                c1, c2, c3 = st.columns(3)
                c1.metric("Category", result["Category"])
                c2.metric("Severity", result["Severity"])
                c3.metric("Sentiment", result["Sentiment"])

                st.metric("Confidence", result["Confidence"])
                st.info(result["Recommended Action"])

                if result.get("Keywords"):
                    st.write("**Keywords:** " + ", ".join(result["Keywords"]))

                pdf = generate_pdf(result, item["complaint"])

                st.download_button(
                    "📄 Download PDF Report",
                    data=pdf,
                    file_name=f"Complaint_Report_{len(st.session_state.history)-i}.pdf",
                    mime="application/pdf",
                    key=f"history_pdf_{i}"
                )
            st.warning("Please paste a complaint before generating a summary.")

