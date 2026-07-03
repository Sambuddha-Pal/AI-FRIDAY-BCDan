import os
import json
import time
import httpx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="QualiSummarize — AI Complaint Insights",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------
# Theme CSS — restyles Streamlit's defaults to match the QualiSummarize
# dashboard mock (purple accent, white cards, rounded corners). This is
# pure presentation: no functionality below depends on it.
# ----------------------------------------
PURPLE = "#7C5CFC"
PURPLE_DARK = "#5B3FE0"

st.markdown(f"""
<style>
    .stApp {{
        background-color: #F8F7FC;
    }}
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF;
        border-right: 1px solid #EEECF6;
    }}
    section[data-testid="stSidebar"] .stRadio > label {{
        display: none;
    }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid #EEECF6;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }}
    div[data-testid="stMetricValue"] {{
        color: #1E1B2E;
        font-weight: 700;
    }}
    div[data-testid="stMetricLabel"] {{
        color: #8A8798;
    }}
    .stButton > button, .stDownloadButton > button {{
        background-color: {PURPLE};
        color: white;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        padding: 0.55rem 1.2rem;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        background-color: {PURPLE_DARK};
        color: white;
    }}
    .qs-card {{
        background: #FFFFFF;
        border: 1px solid #EEECF6;
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04);
    }}
    .qs-badge {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }}
    .qs-badge-Low {{ background:#F1F5F9; color:#475569; }}
    .qs-badge-Medium {{ background:#FEF3E2; color:#B45309; }}
    .qs-badge-High {{ background:#FFEDD5; color:#C2410C; }}
    .qs-badge-Critical {{ background:#FEE2E2; color:#B91C1C; }}
    .qs-badge-category {{ background:#EFEBFF; color:#5B3FE0; }}
    .qs-eyebrow {{
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        color: #8A8798;
        margin-bottom: 4px;
    }}
    h1, h2, h3 {{
        color: #1E1B2E !important;
    }}
    .qs-sidebar-title {{
        font-weight: 800;
        font-size: 18px;
        color: #1E1B2E;
    }}
    .qs-sidebar-sub {{
        font-size: 11px;
        color: #8A8798;
        margin-top: -6px;
    }}
</style>
""", unsafe_allow_html=True)


def badge(text, css_class):
    return f'<span class="qs-badge {css_class}">{text}</span>'


def severity_badge(sev):
    return badge(sev, f"qs-badge-{sev}")


def category_badge(cat):
    return badge(cat, "qs-badge-category")


# ----------------------------------------
# Session State Defaults (API config, history, etc.)
# ----------------------------------------
if "api_endpoint" not in st.session_state:
    # NOTE: this should be the BASE url (no /chat/completions suffix) —
    # langchain_openai.ChatOpenAI appends the path itself.
    st.session_state.api_endpoint = os.getenv("GENAI_API_ENDPOINT", "https://genailab.tcs.in/")

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GENAI_API_KEY", "sk")

if "model_name" not in st.session_state:
    st.session_state.model_name = os.getenv("GENAI_MODEL", "genailab-maas-gpt-35-turbo")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, complaint, result

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None

if "category_notes" not in st.session_state:
    st.session_state.category_notes = {}  # user renames / merge notes, keyed by category name


# ----------------------------------------
# Sidebar
# ----------------------------------------
with st.sidebar:
    top_l, top_r = st.columns([1, 4])
    with top_l:
        st.markdown(
            f"<div style='width:38px;height:38px;border-radius:12px;"
            f"background:linear-gradient(135deg,{PURPLE},#4F46E5);display:flex;"
            f"align-items:center;justify-content:center;color:white;font-size:18px;'>✦</div>",
            unsafe_allow_html=True,
        )
    with top_r:
        st.markdown("<div class='qs-sidebar-title'>QualiSummarize</div>", unsafe_allow_html=True)
        st.markdown("<div class='qs-sidebar-sub'>AI-Powered Complaint Insights</div>", unsafe_allow_html=True)

    st.write("")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "📝 New Analysis",
            "📂 Bulk Upload",
            "💬 Complaints (History)",
            "📊 Analytics",
            "🏷 Categories",
            "📄 Reports",
            "⚙️ Settings",
        ],
        label_visibility="collapsed",
    )
    page = page.split(" ", 1)[1]

    st.markdown("---")

    if st.session_state.api_key and st.session_state.api_key != "sk":
        st.success("✅ AI Model Connected")
    else:
        st.warning("⚠️ No API Key set — go to Settings")

    st.markdown(
        f"""
        <div class="qs-card" style="margin-top:10px;">
            <div class="qs-eyebrow">AI Model Status</div>
            <div style="font-weight:600;color:#1E1B2E;">🟢 Online</div>
            <div style="font-size:12px;color:#8A8798;margin-top:4px;">Model: {st.session_state.model_name}</div>
            <div style="font-size:12px;color:#8A8798;">Endpoint: {st.session_state.api_endpoint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("TCS AI Friday 2026 · AI-Powered Customer Complaint Summary Generator")


# ----------------------------------------
# RAG — STEP 1 & 2: Sample Knowledge Base + local TF-IDF Vector Store
# ----------------------------------------
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
    """RAG STEP 1: Knowledge Data -> TF-IDF vectors -> local vector store."""
    texts = [f"{kb['category']} - {kb['title']}: {kb['content']}" for kb in KNOWLEDGE_BASE]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def retrieve_context(query: str, k: int = 3, min_score: float = 0.05) -> list:
    """RAG STEP 3: Retriever function — TF-IDF cosine similarity against the KB."""
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
    """Builds a ChatOpenAI client pointed at the configured GenAI Lab endpoint."""
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


def build_rag_prompt(complaint_text: str, retrieved: list) -> str:
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


def call_genai_api(complaint_text: str, timeout: int = 30):
    """RAG pipeline: TF-IDF-vectorize -> cosine similarity search -> inject
    top matches into prompt -> LLM generates the grounded answer."""
    llm = get_llm()

    retrieved = retrieve_context(complaint_text, k=3)
    user_prompt = build_rag_prompt(complaint_text, retrieved)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    content = (response.content or "").strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            parsed = json.loads(content[start:end + 1])
        else:
            raise ValueError(f"Could not parse AI response as JSON:\n{content}")

    return parsed, retrieved


def analyze_complaint(text: str) -> dict:
    """Calls the real API (with RAG retrieval baked in), and falls back to a
    clearly labeled dummy result if the call fails."""
    try:
        result, retrieved = call_genai_api(text)
        return {
            "Summary": result.get("summary", "N/A"),
            "Category": result.get("category", "Other"),
            "Severity": result.get("severity", "Medium"),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Confidence": result.get("confidence", 0),
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
            "Confidence": 0,
            "Recommended Action": "Check API endpoint/key in Settings and retry.",
            "Keywords": [],
            "RetrievedContext": [],
            "source": "fallback"
        }


def history_df() -> pd.DataFrame:
    """Flattens st.session_state.history into a DataFrame used across
    Complaints, Analytics, Categories, and Reports pages."""
    if not st.session_state.history:
        return pd.DataFrame(columns=[
            "Timestamp", "Complaint", "Summary", "Category", "Severity",
            "Sentiment", "Confidence", "Recommended Action", "Keywords", "Source"
        ])
    rows = []
    for h in st.session_state.history:
        r = h["result"]
        rows.append({
            "Timestamp": h["timestamp"],
            "Complaint": h["complaint"],
            "Summary": r.get("Summary", ""),
            "Category": r.get("Category", "Other"),
            "Severity": r.get("Severity", "Medium"),
            "Sentiment": r.get("Sentiment", "Neutral"),
            "Confidence": r.get("Confidence", 0),
            "Recommended Action": r.get("Recommended Action", ""),
            "Keywords": ", ".join(r.get("Keywords", []) or []),
            "Source": r.get("source", "ai"),
        })
    return pd.DataFrame(rows)


# ========================================
# 🏠 Dashboard
# ========================================
if page == "Dashboard":
    st.title("👋 Welcome back, Quality Team")
    st.caption("Transform complaints into actionable insights")

    df = history_df()
    total = len(df)
    high_crit = int((df["Severity"].isin(["High", "Critical"])).sum()) if total else 0
    avg_conf = f'{df["Confidence"].mean():.0f}%' if total else "—"
    top_category = df["Category"].mode()[0] if total else "—"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Complaints Processed", total)
    c2.metric("Avg. Confidence", avg_conf)
    c3.metric("High / Critical Severity", high_crit)
    c4.metric("Top Category", top_category)

    st.write("")
    left, right = st.columns([2, 1])

    with left:
        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("📝 Quick Complaint Analysis")
        quick_text = st.text_area("Paste a complaint to analyze it now", height=120, key="quick_complaint")
        if st.button("🚀 Generate AI Summary", key="quick_generate"):
            if quick_text.strip():
                with st.spinner("Analyzing complaint with AI..."):
                    result = analyze_complaint(quick_text)
                st.session_state.history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "complaint": quick_text,
                    "result": result
                })
                st.success("Analysis complete — see it in Complaints (History).")
                st.rerun()
            else:
                st.warning("Please paste a complaint before generating a summary.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("📂 Recent Analyses")
        if total:
            recent = df.sort_values("Timestamp", ascending=False).head(5)
            for _, row in recent.iterrows():
                st.markdown(
                    f"**{row['Timestamp']}** — {row['Complaint'][:80]}{'…' if len(row['Complaint']) > 80 else ''}"
                    f"<br>{category_badge(row['Category'])}{severity_badge(row['Severity'])}",
                    unsafe_allow_html=True,
                )
                st.markdown("<hr style='margin:8px 0;border-color:#F1EFFB;'>", unsafe_allow_html=True)
        else:
            st.caption("No complaints analyzed yet — try the quick analysis box above.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("🏷 Top Categories")
        if total:
            cat_counts = df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.pie(cat_counts, names="Category", values="Count", hole=0.55,
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280,
                               legend=dict(orientation="h", y=-0.15))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Category breakdown will appear here once you analyze complaints.")
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# 📝 New Analysis
# ========================================
elif page == "New Analysis":
    st.title("📝 New Complaint Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        complaint = st.text_area(
            "Customer Complaint",
            height=250,
            placeholder="Paste customer complaint here..."
        )

    with col2:
        df = history_df()
        st.metric("Complaints Analyzed", len(df))
        st.metric("High/Critical Severity", int((df["Severity"].isin(["High", "Critical"])).sum()) if len(df) else 0)
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
            st.metric("Confidence", f'{result["Confidence"]}%')

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
            edited = st.text_area("", value=result["Summary"], height=120, key="edited_summary")
            if st.button("Save Edited Summary"):
                st.session_state.history[-1]["result"]["Summary"] = edited
                st.success("Saved.")
        else:
            st.warning("Please paste a complaint before generating a summary.")


# ========================================
# 📂 Bulk Upload
# ========================================
elif page == "Bulk Upload":
    st.title("📂 Bulk Upload")
    st.caption("Upload a .csv, .json, or .txt file and run every complaint through the same AI + RAG pipeline.")

    uploaded = st.file_uploader("Upload complaints file", type=["csv", "json", "txt"])

    if uploaded is not None:
        try:
            if uploaded.name.endswith(".csv"):
                df_in = pd.read_csv(uploaded)
                text_col = st.selectbox("Which column contains the complaint text?", df_in.columns)
                complaints_list = df_in[text_col].dropna().astype(str).tolist()
            elif uploaded.name.endswith(".json"):
                raw = json.load(uploaded)
                if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                    keys = list(raw[0].keys())
                    text_key = st.selectbox("Which field contains the complaint text?", keys)
                    complaints_list = [str(r.get(text_key, "")) for r in raw if r.get(text_key)]
                elif isinstance(raw, list):
                    complaints_list = [str(x) for x in raw]
                else:
                    complaints_list = [json.dumps(raw)]
            else:  # .txt — one complaint per line
                complaints_list = [
                    line.decode("utf-8").strip() if isinstance(line, bytes) else line.strip()
                    for line in uploaded.readlines()
                ]
                complaints_list = [c for c in complaints_list if c]

            st.session_state.bulk_df = complaints_list
            st.success(f"Loaded {len(complaints_list)} complaints from {uploaded.name}.")
            st.dataframe(pd.DataFrame({"Complaint": complaints_list}).head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.session_state.bulk_df = None

    if st.session_state.bulk_df:
        st.write("")
        if st.button(f"🚀 Analyze all {len(st.session_state.bulk_df)} complaints", use_container_width=True):
            results = []
            progress = st.progress(0, text="Starting batch analysis...")
            total_n = len(st.session_state.bulk_df)
            for i, text in enumerate(st.session_state.bulk_df):
                result = analyze_complaint(text)
                st.session_state.history.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "complaint": text,
                    "result": result
                })
                results.append({"Complaint": text, **{k: v for k, v in result.items() if k != "RetrievedContext"}})
                progress.progress((i + 1) / total_n, text=f"Analyzed {i + 1} of {total_n}")
            st.session_state.bulk_results = pd.DataFrame(results)
            progress.empty()
            st.success(f"Batch analysis complete — {total_n} complaints processed and added to History.")

    if st.session_state.bulk_results is not None:
        st.subheader("Batch Results")
        st.dataframe(st.session_state.bulk_results, use_container_width=True)
        csv = st.session_state.bulk_results.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download results as CSV", csv, "bulk_results.csv", "text/csv")


# ========================================
# 💬 Complaints (History)
# ========================================
elif page == "Complaints (History)":
    st.title("💬 Complaints")

    df = history_df()
    st.caption(f"{len(df)} complaints analyzed so far")

    if df.empty:
        st.info("No complaints yet. Analyze one from New Analysis or Bulk Upload.")
    else:
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            query = st.text_input("🔍 Search complaints")
        with f2:
            cat_filter = st.selectbox("Category", ["All"] + sorted(df["Category"].unique().tolist()))
        with f3:
            sev_filter = st.selectbox("Severity", ["All", "Low", "Medium", "High", "Critical"])

        filtered = df.copy()
        if query:
            filtered = filtered[filtered["Complaint"].str.contains(query, case=False, na=False)]
        if cat_filter != "All":
            filtered = filtered[filtered["Category"] == cat_filter]
        if sev_filter != "All":
            filtered = filtered[filtered["Severity"] == sev_filter]

        st.caption(f"Showing {len(filtered)} of {len(df)}")

        for idx, row in filtered.sort_values("Timestamp", ascending=False).iterrows():
            st.markdown('<div class="qs-card">', unsafe_allow_html=True)
            st.markdown(
                f"**{row['Timestamp']}**  \n{row['Complaint']}"
                f"<br><br>{category_badge(row['Category'])}{severity_badge(row['Severity'])}"
                f"<span style='color:#8A8798;font-size:12px;'>Confidence: {row['Confidence']}%</span>",
                unsafe_allow_html=True,
            )
            with st.expander("View AI summary & recommended action"):
                st.write(row["Summary"])
                st.info(row["Recommended Action"])
                if row["Keywords"]:
                    st.caption(f"Keywords: {row['Keywords']}")
            col_a, col_b = st.columns([1, 6])
            with col_a:
                if st.button("🗑 Delete", key=f"del_{idx}"):
                    del st.session_state.history[idx]
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# 📊 Analytics
# ========================================
elif page == "Analytics":
    st.title("📊 Analytics")
    df = history_df()

    if df.empty:
        st.info("Analyze some complaints first to see analytics here.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="qs-card">', unsafe_allow_html=True)
            st.subheader("Complaints by Category")
            cat_counts = df["Category"].value_counts().reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig = px.bar(cat_counts, x="Category", y="Count", color="Category",
                         color_discrete_sequence=px.colors.qualitative.Bold)
            fig.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="qs-card">', unsafe_allow_html=True)
            st.subheader("Severity Distribution")
            sev_counts = df["Severity"].value_counts().reindex(["Low", "Medium", "High", "Critical"]).fillna(0).reset_index()
            sev_counts.columns = ["Severity", "Count"]
            fig2 = px.bar(sev_counts, x="Severity", y="Count", color="Severity",
                          color_discrete_map={"Low": "#94A3B8", "Medium": "#F5A623", "High": "#FB923C", "Critical": "#EF4444"})
            fig2.update_layout(showlegend=False, height=320)
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("Complaint Volume Over Time")
        df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
        daily = df.groupby("Date").size().reset_index(name="Count")
        fig3 = px.line(daily, x="Date", y="Count", markers=True)
        fig3.update_traces(line_color=PURPLE)
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("Frequent Keywords")
        all_kw = ", ".join(df["Keywords"].tolist()).split(", ")
        all_kw = [k.strip() for k in all_kw if k.strip()]
        if all_kw:
            kw_counts = pd.Series(all_kw).value_counts().head(15).reset_index()
            kw_counts.columns = ["Keyword", "Count"]
            fig4 = px.bar(kw_counts, x="Count", y="Keyword", orientation="h",
                         color_discrete_sequence=[PURPLE])
            fig4.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.caption("No keywords recorded yet.")
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# 🏷 Categories
# ========================================
elif page == "Categories":
    st.title("🏷 Categories")
    df = history_df()
    counts = df["Category"].value_counts().to_dict() if not df.empty else {}

    all_categories = sorted(set([kb["category"] for kb in KNOWLEDGE_BASE]) | set(counts.keys()) |
                             {"Battery", "Display", "Audio", "Camera", "Build Quality",
                              "Software/Firmware", "Performance", "Shipping/Packaging",
                              "Customer Service", "Other"})

    cols = st.columns(3)
    for i, cat in enumerate(all_categories):
        with cols[i % 3]:
            display_name = st.session_state.category_notes.get(cat, cat)
            st.markdown('<div class="qs-card">', unsafe_allow_html=True)
            st.markdown(f"**{display_name}**")
            st.markdown(f"<div style='font-size:24px;font-weight:700;color:#1E1B2E;'>{counts.get(cat, 0)}</div>", unsafe_allow_html=True)
            kb_entries = [kb for kb in KNOWLEDGE_BASE if kb["category"] == cat]
            if kb_entries:
                st.caption(f"{len(kb_entries)} knowledge base entr{'y' if len(kb_entries)==1 else 'ies'} linked")
            with st.expander("Manage"):
                new_name = st.text_input("Rename category", value=display_name, key=f"rename_{cat}")
                if st.button("Save name", key=f"save_{cat}"):
                    st.session_state.category_notes[cat] = new_name
                    st.success(f"Renamed to {new_name}")
                    st.rerun()
                if not df.empty:
                    matching = df[df["Category"] == cat]
                    for _, row in matching.head(3).iterrows():
                        st.caption(f"• {row['Complaint'][:60]}")
            st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# 📄 Reports
# ========================================
elif page == "Reports":
    st.title("📄 Reports")
    df = history_df()

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("Generate Report")
        report_type = st.radio("Report Type", ["Daily", "Weekly", "Monthly", "Custom Date Range"])
        if report_type == "Custom Date Range" and not df.empty:
            df["Date"] = pd.to_datetime(df["Timestamp"]).dt.date
            date_range = st.date_input("Range", [df["Date"].min(), df["Date"].max()])
        kpis = st.multiselect(
            "KPIs to include",
            ["Total complaints", "Complaints by category", "Complaint trends", "AI accuracy",
             "Most affected products", "Frequent keywords", "Severity distribution"],
            default=["Total complaints", "Complaints by category", "Severity distribution"]
        )
        generate = st.button("📄 Generate Report", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="qs-card">', unsafe_allow_html=True)
        st.subheader("Preview")
        if generate:
            if df.empty:
                st.warning("No data yet — analyze some complaints first.")
            else:
                st.markdown(f"**{report_type} Report** · {len(kpis)} KPIs included")
                if "Total complaints" in kpis:
                    st.metric("Total Complaints", len(df))
                if "Severity distribution" in kpis:
                    st.write(df["Severity"].value_counts())
                if "Complaints by category" in kpis:
                    st.write(df["Category"].value_counts())
                if "AI accuracy" in kpis:
                    st.metric("Avg. AI Confidence", f'{df["Confidence"].mean():.0f}%')

                st.write("**Executive Summary**")
                top_cat = df["Category"].mode()[0]
                st.write(
                    f"Over the selected period, {len(df)} complaints were processed. "
                    f"The most common category was **{top_cat}**, with "
                    f"{int((df['Severity'].isin(['High','Critical'])).sum())} flagged High or Critical severity."
                )

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Export CSV", csv, "complaint_report.csv", "text/csv")

                report_md = df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)
                st.download_button("🖨 Download printable report (Markdown)", report_md, "complaint_report.md", "text/markdown")
                st.caption("PDF export isn't wired up yet — download the Markdown/CSV and print-to-PDF from your browser, or ask me to add a real PDF export.")
        else:
            st.caption("Choose a report type and KPIs, then click Generate Report.")
        st.markdown('</div>', unsafe_allow_html=True)


# ========================================
# ⚙️ Settings
# ========================================
elif page == "Settings":
    st.title("⚙️ Settings")

    st.markdown('<div class="qs-card">', unsafe_allow_html=True)
    st.subheader("AI Model Settings")
    st.session_state.api_endpoint = st.text_input("GenAI API Endpoint (base URL)", value=st.session_state.api_endpoint)
    st.session_state.api_key = st.text_input("API Key", value=st.session_state.api_key, type="password")
    st.session_state.model_name = st.text_input("Model name", value=st.session_state.model_name)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, float(st.session_state.temperature), 0.1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="qs-card">', unsafe_allow_html=True)
    st.subheader("Data Retention")
    if st.button("🗑 Clear all history"):
        st.session_state.history = []
        st.success("History cleared.")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="qs-card">', unsafe_allow_html=True)
    st.subheader("Connection Test")
    if st.button("🔌 Test connection to GenAI endpoint"):
        try:
            llm = get_llm()
            resp = llm.invoke([HumanMessage(content="Reply with the single word: OK")])
            st.success(f"Connected. Model replied: {resp.content.strip()}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    st.markdown('</div>', unsafe_allow_html=True)
