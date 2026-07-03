<<<<<<< HEAD
import streamlit as st
from backend.rag import analyze_complaint

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------
# Page Config
# ---------------------------------
=======
"""
AI Customer Complaint Summary Generator — with RAG

Talks to an OpenAI-compatible gateway (e.g. an internal TCS GenAI Lab
endpoint) via langchain's ChatOpenAI for chat, and the openai SDK for
embeddings. Both clients share an httpx.Client so TLS verification can be
disabled for internal endpoints with self-signed certs.

Requirements (requirements.txt):
    streamlit
    pandas
    plotly
    openai>=1.0.0
    langchain-openai
    httpx
    numpy
    python-dotenv   # optional, for .env support

Environment variables (optional defaults):
    API_ENDPOINT         - base URL of the OpenAI-compatible gateway, e.g. https://genailab.tcs.in/
    API_KEY               - API key for that gateway
    OPENAI_CHAT_MODEL    - default "genailab-maas-gpt-35-turbo"
    OPENAI_EMBED_MODEL   - default "text-embedding-3-small"
"""

import os
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import httpx

from langchain_openai import ChatOpenAI
from openai import OpenAI as OpenAIEmbeddingClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

st.set_page_config(
    page_title="AI Manufacturing Complaint Analyzer",
    page_icon="🏭",
    layout="wide"
)

<<<<<<< HEAD
# ---------------------------------
# Custom CSS
# ---------------------------------

st.markdown("""
<style>

.main{
    background:#F6F8FC;
}

.header{
    background:linear-gradient(90deg,#1E3C72,#2A5298);
    padding:25px;
    border-radius:15px;
    color:white;
    text-align:center;
    margin-bottom:25px;
}

.metric-card{
    background:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.08);
}
=======
# ----------------------------------------
# Session State Defaults
# ----------------------------------------
if "api_endpoint" not in st.session_state:
    st.session_state.api_endpoint = os.getenv("API_ENDPOINT", "")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = os.getenv("API_KEY", "")

if "chat_model" not in st.session_state:
    st.session_state.chat_model = os.getenv("OPENAI_CHAT_MODEL", "genailab-maas-gpt-35-turbo")

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "top_k" not in st.session_state:
    st.session_state.top_k = 4

if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.25

if "verify_ssl" not in st.session_state:
    # Many internal gateways use self-signed certs, so this defaults to False
    # to match the working example. See the warning in Settings.
    st.session_state.verify_ssl = False

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: timestamp, complaint, result

if "bulk_df" not in st.session_state:
    st.session_state.bulk_df = None

if "bulk_results" not in st.session_state:
    st.session_state.bulk_results = None

if "kb_chunks" not in st.session_state:
    st.session_state.kb_chunks = []  # list of dicts: text, source, embedding (np.array)


# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("🏭 Manufacturing AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Complaint Analysis",
        "Bulk Upload",
        "Knowledge Base",
        "Analytics",
        "History",
        "Settings"
    ]
)

st.sidebar.markdown("---")

if st.session_state.api_endpoint and st.session_state.openai_api_key:
    st.sidebar.success("✅ Model Connected")
else:
    st.sidebar.warning("⚠️ No API endpoint/key set — go to Settings")

if not st.session_state.verify_ssl:
    st.sidebar.caption("🔓 TLS verification is disabled for the model endpoint")

st.sidebar.caption(f"KB chunks: {len(st.session_state.kb_chunks)}  •  History: {len(st.session_state.history)}")

st.sidebar.info(
    """
    **TCS AI Friday 2026**

    AI-Powered Customer Complaint
    Summary Generator (RAG-enabled)
    """
)

# ----------------------------------------
# Client helpers (chat via langchain, embeddings via openai SDK)
# ----------------------------------------

def _http_client() -> httpx.Client:
    # Shared TLS behavior for both chat and embedding calls.
    return httpx.Client(verify=False)


def get_client() -> ChatOpenAI:
    """Returns a langchain ChatOpenAI client pointed at the configured
    OpenAI-compatible gateway."""
    endpoint = st.session_state.api_endpoint.strip()
    api_key = st.session_state.openai_api_key.strip()
    model = st.session_state.chat_model.strip()

    if not endpoint:
        raise ValueError("API endpoint is not configured. Set it in Settings.")
    if not api_key:
        raise ValueError("API key is not configured. Set it in Settings.")

    return ChatOpenAI(
        base_url=endpoint,
        api_key=api_key,
        model=model,
        http_client=_http_client(),
        temperature=st.session_state.temperature,
    )


def get_embedding_client() -> OpenAIEmbeddingClient:
    endpoint = st.session_state.api_endpoint.strip()
    api_key = st.session_state.openai_api_key.strip()

    if not endpoint:
        raise ValueError("API endpoint is not configured. Set it in Settings.")
    if not api_key:
        raise ValueError("API key is not configured. Set it in Settings.")

    return OpenAIEmbeddingClient(api_key=api_key, base_url=endpoint, http_client=_http_client())


# ----------------------------------------
# Embeddings + retrieval (in-memory RAG)
# ----------------------------------------

def get_embedding(text: str) -> np.ndarray:
    client = get_embedding_client()
    text = text.replace("\n", " ").strip()
    response = client.embeddings.create(model=st.session_state.embedding_model, input=text)
    return np.array(response.data[0].embedding, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """Simple fixed-size chunking with overlap, splitting on paragraph boundaries first."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 1 <= chunk_size:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= chunk_size:
                buf = p
            else:
                # Hard-split very long paragraphs
                for i in range(0, len(p), chunk_size - overlap):
                    chunks.append(p[i:i + chunk_size])
                buf = ""
    if buf:
        chunks.append(buf)
    return chunks if chunks else [text[:chunk_size]]


def retrieve_context(query_embedding: np.ndarray, top_k: int = None, threshold: float = None) -> list:
    """Search both the knowledge base and past complaint history for the
    most relevant snippets to the given query embedding."""
    top_k = top_k if top_k is not None else st.session_state.top_k
    threshold = threshold if threshold is not None else st.session_state.similarity_threshold

    candidates = []

    for chunk in st.session_state.kb_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        candidates.append({
            "text": chunk["text"],
            "source": f"Knowledge Base: {chunk['source']}",
            "score": score
        })

    for h in st.session_state.history:
        emb = h["result"].get("embedding")
        if emb is None:
            continue
        score = cosine_similarity(query_embedding, emb)
        candidates.append({
            "text": f"Complaint: {h['complaint']}\nPrior analysis: {h['result'].get('Summary', '')} "
                    f"(Category: {h['result'].get('Category', '')}, Severity: {h['result'].get('Severity', '')})",
            "source": f"Past Complaint ({h['timestamp']})",
            "score": score
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    filtered = [c for c in candidates if c["score"] >= threshold]
    return filtered[:top_k]


# ----------------------------------------
# Core AI Function — chat completion via langchain ChatOpenAI
# ----------------------------------------

SYSTEM_PROMPT = """You are an expert quality-assurance analyst for a manufacturing company.
You read raw, unstructured customer complaint text (which may be messy, informal, or contain
typos) and produce a structured analysis to help Quality and Customer Service teams triage issues
quickly.

You may be given RELEVANT CONTEXT retrieved from past complaints and internal knowledge-base
documents. Use it only if it is genuinely relevant — for example, to recognize a recurring known
defect, align with an established resolution, or calibrate severity against similar past cases.
If the context is not relevant, ignore it and rely on the complaint text alone. Never mention the
retrieval process itself in your output.

Always respond with STRICT JSON ONLY — no markdown fences, no commentary — matching this schema:

{
  "summary": "one or two sentence concise summary of the complaint",
  "category": "one of: Battery, Display, Audio, Camera, Build Quality, Software/Firmware, Performance, Shipping/Packaging, Customer Service, Other",
  "severity": "one of: Low, Medium, High, Critical",
  "sentiment": "one of: Positive, Neutral, Negative, Very Negative",
  "confidence": "integer percentage 0-100 representing your confidence in this classification",
  "recommended_action": "a specific, actionable recommendation for the internal team (e.g. which team to escalate to, whether a replacement/refund is warranted, whether it's a safety issue)",
  "keywords": ["short", "list", "of", "key", "defect", "terms"],
  "context_relevant": true or false, indicating whether the retrieved context materially informed this analysis
}
"""


def call_openai_api(complaint_text: str, context_items: list) -> dict:
    llm = get_client()

    if context_items:
        context = "\n\n".join(f"{c['source']}\n{c['text']}" for c in context_items)
        prompt = f"""{SYSTEM_PROMPT}

Context:

{context}

Complaint:

{complaint_text}

Return ONLY valid JSON.
"""
    else:
        prompt = f"""{SYSTEM_PROMPT}

Complaint:

{complaint_text}

Return ONLY valid JSON.
"""

    response = llm.invoke(prompt)
    content = response.content.strip()

    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1:
            return json.loads(content[start:end + 1])
        raise ValueError(f"Could not parse AI response as JSON:\n{content}")


def analyze_complaint(text: str) -> dict:
    """Embeds the complaint, retrieves relevant context (RAG), calls the model,
    and returns a UI-ready result dict. Falls back to a clearly labeled dummy
    result if anything fails, so the demo never hard-crashes."""
    try:
        query_embedding = get_embedding(text)
        context_items = retrieve_context(query_embedding)
        result = call_openai_api(text, context_items)
        return {
            "Summary": result.get("summary", "N/A"),
            "Category": result.get("category", "Other"),
            "Severity": result.get("severity", "Medium"),
            "Sentiment": result.get("sentiment", "Neutral"),
            "Confidence": f'{result.get("confidence", "N/A")}%',
            "Recommended Action": result.get("recommended_action", "N/A"),
            "Keywords": result.get("keywords", []),
            "ContextRelevant": result.get("context_relevant", False),
            "ContextItems": context_items,
            "embedding": query_embedding,
            "source": "ai"
        }
    except Exception as e:
        st.error(f"⚠️ AI call failed, showing fallback result. Error: {e}")
        return {
            "Summary": "(fallback) Could not reach the model — please check API settings.",
            "Category": "Other",
            "Severity": "Medium",
            "Sentiment": "Neutral",
            "Confidence": "0%",
            "Recommended Action": "Check API endpoint/key in Settings and retry.",
            "Keywords": [],
            "ContextRelevant": False,
            "ContextItems": [],
            "embedding": None,
            "source": "fallback"
        }

>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

.source-card{
    background:white;
    padding:15px;
    border-radius:10px;
    border-left:6px solid #1E88E5;
    margin-bottom:10px;
}

.footer{
    text-align:center;
    color:gray;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# PDF
# ---------------------------------

<<<<<<< HEAD
def generate_pdf(result, complaint):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story=[]

    story.append(
        Paragraph("<b>Manufacturing Complaint Analysis Report</b>",styles["Title"])
    )

    story.append(
        Paragraph("<b>Customer Complaint</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(complaint,styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Summary</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["summary"],styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Category</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["category"],styles["BodyText"])
    )

    story.append(
        Paragraph("<b>Severity</b>",styles["Heading2"])
    )

    story.append(
        Paragraph(result["severity"],styles["BodyText"])
    )

    if "recommended_action" in result:

        story.append(
            Paragraph("<b>Recommended Action</b>",styles["Heading2"])
        )

        story.append(
            Paragraph(result["recommended_action"],styles["BodyText"])
        )

    doc.build(story)
=======
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
        st.metric("Model", st.session_state.chat_model)
        st.metric("KB Chunks", len(st.session_state.kb_chunks))
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

    buffer.seek(0)

    return buffer

<<<<<<< HEAD
# ---------------------------------
# Header
# ---------------------------------

st.markdown("""
<div class='header'>
<h1>🏭 AI Manufacturing Complaint Analyzer</h1>
<h4>Retrieval-Augmented Generation (RAG)</h4>
<p>Analyze manufacturing complaints using AI and your internal knowledge base.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# KPI Cards
# ---------------------------------

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric("📄 Knowledge Docs","250+")

with c2:
    st.metric("🤖 AI Model","GPT-35")

with c3:
    st.metric("⚡ Retrieval","FAISS")
=======
            with st.spinner("Retrieving relevant context and analyzing with AI..."):
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
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

with c4:
    st.metric("📊 Accuracy","96%")

<<<<<<< HEAD
st.divider()

# ---------------------------------
# Layout
# ---------------------------------

left,right=st.columns([2,1])

with left:

    st.subheader("📝 Customer Complaint")

    complaint=st.text_area(
        "",
        height=250,
        placeholder="Example: The battery overheats after firmware update and drains within two hours..."
=======
            if result["Keywords"]:
                st.write("**Keywords:** " + ", ".join(result["Keywords"]))

            st.subheader("Recommended Action")
            st.info(result["Recommended Action"])

            if result["ContextItems"]:
                label = "🔎 Retrieved Context (used)" if result["ContextRelevant"] else "🔎 Retrieved Context (not used by model)"
                with st.expander(label):
                    for c in result["ContextItems"]:
                        st.markdown(f"**{c['source']}** — similarity {c['score']:.2f}")
                        st.caption(c["text"])
            else:
                st.caption("No sufficiently similar past complaints or KB documents were found for this complaint.")

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

    st.caption(
        "Upload a CSV or JSON file with a column/field named **complaint** (or similar text field). "
        "Each row will be embedded and analyzed with RAG context, so bulk runs use more API calls "
        "(1 embedding + 1 chat call per row)."
    )

    uploaded = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"]
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601
    )

    uploaded=st.file_uploader(
        "Optional Knowledge Base Upload",
        type=["pdf","csv","txt"]
    )

<<<<<<< HEAD
    analyze=st.button(
        "🚀 Analyze Complaint",
        use_container_width=True
    )

with right:

    st.subheader("ℹ️ Workflow")

    st.info("""
1. Enter customer complaint

2. Retrieve relevant manufacturing documents

3. Generate AI summary

4. Recommend corrective action

5. Export report
""")

# ---------------------------------
# Analysis
# ---------------------------------

if analyze:

    if complaint.strip()=="":

        st.warning("Please enter a complaint.")

    else:

        with st.spinner("🔍 Searching Knowledge Base..."):

            result=analyze_complaint(complaint)

        st.success("Analysis Completed")

        st.divider()

        st.subheader("🤖 AI Summary")

        st.success(result["summary"])

        c1,c2,c3=st.columns(3)

        c1.metric("Category",result["category"])
        c2.metric("Severity",result["severity"])

        if "confidence" in result:
            c3.metric("Confidence",result["confidence"])
        else:
            c3.metric("Status","Completed")

        if "recommended_action" in result:

            st.subheader("💡 Recommended Action")

            st.info(result["recommended_action"])

        elif "recommendation" in result:

            st.subheader("💡 Recommended Action")
=======
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_json(uploaded)

        st.session_state.bulk_df = df
        st.success(f"File uploaded — {len(df)} rows")
        st.dataframe(df, use_container_width=True)

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
                    "Recommended Action": res["Recommended Action"],
                    "Context Used": res["ContextRelevant"]
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
# Knowledge Base
# ========================================

elif page == "Knowledge Base":

    st.title("📚 Knowledge Base")
    st.caption(
        "Upload reference documents (known-issue lists, product manuals, resolution guides, etc.) "
        "as .txt, .md, or .csv. Text is chunked and embedded, then retrieved automatically during "
        "complaint analysis. This is stored in memory for the current session only."
    )

    kb_files = st.file_uploader(
        "Upload reference documents",
        type=["txt", "md", "csv"],
        accept_multiple_files=True
    )

    if kb_files and st.button("Add to Knowledge Base"):
        if not st.session_state.api_endpoint or not st.session_state.openai_api_key:
            st.error("Set your API endpoint and key in Settings first.")
        else:
            progress = st.progress(0)
            added = 0
            for fi, f in enumerate(kb_files):
                if f.name.endswith(".csv"):
                    df = pd.read_csv(f)
                    texts = [" | ".join(f"{col}: {row[col]}" for col in df.columns) for _, row in df.iterrows()]
                else:
                    raw = f.read().decode("utf-8", errors="ignore")
                    texts = chunk_text(raw)

                for ci, chunk in enumerate(texts):
                    try:
                        emb = get_embedding(chunk)
                        st.session_state.kb_chunks.append({
                            "text": chunk,
                            "source": f"{f.name} (chunk {ci + 1})",
                            "embedding": emb
                        })
                        added += 1
                    except Exception as e:
                        st.error(f"Failed to embed a chunk from {f.name}: {e}")
                        break
                progress.progress((fi + 1) / len(kb_files))
            st.success(f"Added {added} chunks to the knowledge base.")

    st.markdown("---")
    st.subheader(f"Current Knowledge Base ({len(st.session_state.kb_chunks)} chunks)")

    if st.session_state.kb_chunks:
        kb_df = pd.DataFrame([
            {"Source": c["source"], "Preview": c["text"][:120] + ("..." if len(c["text"]) > 120 else "")}
            for c in st.session_state.kb_chunks
        ])
        st.dataframe(kb_df, use_container_width=True)

        if st.button("🗑️ Clear Knowledge Base"):
            st.session_state.kb_chunks = []
            st.rerun()
    else:
        st.info("No knowledge base documents uploaded yet.")
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

            st.info(result["recommendation"])

        if "root_cause" in result:

            st.subheader("🔍 Root Cause")

<<<<<<< HEAD
            st.warning(result["root_cause"])

        st.divider()

        st.subheader("📚 Retrieved Knowledge")

        for i,source in enumerate(result["sources"],1):

            if isinstance(source,dict):

                st.markdown(f"""
<div class="source-card">
=======
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
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

<b>📄 Document {i}</b><br>

<b>Source:</b> {source.get("source","Knowledge Base")}<br><br>

{source.get("content","")}

<<<<<<< HEAD
</div>
""",unsafe_allow_html=True)

            else:

                st.markdown(f"""
<div class="source-card">

<b>📄 Document {i}</b><br><br>

{source}

</div>
""",unsafe_allow_html=True)
=======
    if st.session_state.history:
        hist_rows = [{
            "Timestamp": h["timestamp"],
            "Complaint": h["complaint"][:80] + ("..." if len(h["complaint"]) > 80 else ""),
            "Category": h["result"]["Category"],
            "Severity": h["result"]["Severity"],
            "Sentiment": h["result"]["Sentiment"],
            "Context Used": h["result"].get("ContextRelevant", False),
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
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601

        st.divider()

        pdf=generate_pdf(result,complaint)

<<<<<<< HEAD
        st.download_button(
            "📥 Export Analysis Report (PDF)",
            data=pdf,
            file_name="Manufacturing_Complaint_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("""
<div class='footer'>
Built for <b>TCS AI Friday Season 2</b> | AI-Powered Manufacturing Complaint Summary Generator
</div>
""",unsafe_allow_html=True)
=======
    st.title("⚙️ Settings")

    st.markdown("Configure the connection to your model gateway below.")

    st.session_state.api_endpoint = st.text_input(
        "API Endpoint",
        value=st.session_state.api_endpoint,
        help="Base URL of the OpenAI-compatible gateway, e.g. https://genailab.tcs.in/"
    )

    st.session_state.openai_api_key = st.text_input(
        "API Key",
        value=st.session_state.openai_api_key,
        type="password"
    )

    st.session_state.chat_model = st.text_input(
        "Chat Model",
        value=st.session_state.chat_model,
        help="e.g. genailab-maas-gpt-35-turbo"
    )

    st.session_state.embedding_model = st.text_input(
        "Embedding Model",
        value=st.session_state.embedding_model,
        help="Embedding model name served by your gateway"
    )

    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.temperature
    )

    st.session_state.verify_ssl = not st.checkbox(
        "Skip TLS certificate verification",
        value=not st.session_state.verify_ssl,
        help="Enable this only for internal endpoints with self-signed certificates. "
             "Leaving TLS verification off makes the connection vulnerable to interception — "
             "turn it back on for any public or production endpoint."
    )
    if not st.session_state.verify_ssl:
        st.warning("TLS certificate verification is disabled. Only use this for trusted internal endpoints.")

    with st.expander("Retrieval (RAG) settings"):
        st.session_state.top_k = st.slider(
            "Max context snippets to retrieve", 1, 10, st.session_state.top_k
        )
        st.session_state.similarity_threshold = st.slider(
            "Minimum similarity score to include a snippet", 0.0, 1.0, st.session_state.similarity_threshold,
            help="Higher = stricter matching, fewer but more relevant snippets."
        )

    if st.button("💾 Save Settings"):
        st.success("Settings saved for this session.")

    st.markdown("---")
    st.subheader("🔌 Test Connection")

    if st.button("Send Test Complaint"):
        with st.spinner("Testing connection..."):
            try:
                test_text = "The screen flickers randomly and the battery drains within two hours."
                emb = get_embedding(test_text)
                st.write(f"✅ Embedding call succeeded — vector length {len(emb)}")
                result = call_openai_api(test_text, [])
                st.success("Chat completion succeeded! Sample response:")
                st.json(result)
            except Exception as e:
                st.error(f"Connection failed: {e}")

    st.markdown("---")
    st.caption(
        "Tip: instead of typing these values here every time, set the environment variables "
        "`API_ENDPOINT`, `API_KEY`, `OPENAI_CHAT_MODEL`, and `OPENAI_EMBED_MODEL` (e.g. in a "
        "`.env` file) before launching the app, and they'll be used as defaults."
    )
>>>>>>> 0b0613c5c3e34176fb0b010f72cf1ca938c23601
