"""
AI-Powered Customer Complaint Summary Generator for Manufacturing
Backend API (FastAPI)

Run:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""
import json
from datetime import datetime, timezone
from io import StringIO

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import db
from utils.anonymize import preprocess
from utils.summarizer import analyze_complaint, CATEGORIES, SEVERITIES
from utils.export import to_csv, to_pdf

app = FastAPI(title="Complaint Summary Generator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only — restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

db.init_db()


# ---------- Schemas ----------

class ComplaintIn(BaseModel):
    text: str


class BulkComplaintsIn(BaseModel):
    texts: list[str]


class EditIn(BaseModel):
    summary: str
    category: str
    severity: str


# ---------- Health / meta ----------

@app.get("/api/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "severities": SEVERITIES}


# ---------- Ingest ----------

@app.post("/api/complaints")
def create_complaint(payload: ComplaintIn):
    """Add one complaint (does NOT auto-analyze; call /analyze after)."""
    anonymized = preprocess(payload.text)
    new_id = db.insert_complaint(payload.text, anonymized)
    return db.get_complaint(new_id)


@app.post("/api/complaints/bulk")
def create_complaints_bulk(payload: BulkComplaintsIn):
    """Add many complaints at once (e.g. pasted list, one per line)."""
    created = []
    for text in payload.texts:
        text = text.strip()
        if not text:
            continue
        anonymized = preprocess(text)
        new_id = db.insert_complaint(text, anonymized)
        created.append(db.get_complaint(new_id))
    return created


@app.post("/api/complaints/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Accepts a .csv (one complaint per row, first column) or .json
    (list of strings, or list of {"text": "..."} objects) or .txt
    (one complaint per line).
    """
    content = (await file.read()).decode("utf-8", errors="ignore")
    texts: list[str] = []

    if file.filename.endswith(".json"):
        data = json.loads(content)
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and "text" in item:
                texts.append(item["text"])
    elif file.filename.endswith(".csv"):
        import csv as csv_mod
        reader = csv_mod.reader(StringIO(content))
        rows = list(reader)
        # skip header if first cell looks like a header label
        start = 1 if rows and rows[0] and rows[0][0].lower() in ("text", "complaint", "complaint_text") else 0
        for row in rows[start:]:
            if row and row[0].strip():
                texts.append(row[0].strip())
    else:  # treat as plain text, one complaint per line
        texts = [line.strip() for line in content.splitlines() if line.strip()]

    created = []
    for text in texts:
        anonymized = preprocess(text)
        new_id = db.insert_complaint(text, anonymized)
        created.append(db.get_complaint(new_id))
    return {"imported": len(created), "complaints": created}


# ---------- Analysis ----------

@app.post("/api/complaints/{complaint_id}/analyze")
def analyze_one(complaint_id: int):
    record = db.get_complaint(complaint_id)
    if not record:
        raise HTTPException(404, "Complaint not found")
    try:
        analysis = analyze_complaint(record["anonymized_text"])
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    db.update_analysis(complaint_id, analysis)
    return db.get_complaint(complaint_id)


@app.post("/api/complaints/analyze-all")
def analyze_all():
    """Analyze every complaint currently in 'pending' status."""
    results = []
    errors = []
    for record in db.list_complaints():
        if record["status"] == "pending":
            try:
                analysis = analyze_complaint(record["anonymized_text"])
                db.update_analysis(record["id"], analysis)
                results.append(db.get_complaint(record["id"]))
            except RuntimeError as e:
                errors.append({"id": record["id"], "error": str(e)})
    return {"analyzed": len(results), "errors": errors, "complaints": results}


# ---------- Read / edit / delete ----------

@app.get("/api/complaints")
def list_all():
    return db.list_complaints()


@app.get("/api/complaints/{complaint_id}")
def get_one(complaint_id: int):
    record = db.get_complaint(complaint_id)
    if not record:
        raise HTTPException(404, "Complaint not found")
    return record


@app.put("/api/complaints/{complaint_id}")
def edit_one(complaint_id: int, payload: EditIn):
    record = db.get_complaint(complaint_id)
    if not record:
        raise HTTPException(404, "Complaint not found")
    db.update_summary_edit(complaint_id, payload.summary, payload.category, payload.severity)
    return db.get_complaint(complaint_id)


@app.delete("/api/complaints/{complaint_id}")
def delete_one(complaint_id: int):
    record = db.get_complaint(complaint_id)
    if not record:
        raise HTTPException(404, "Complaint not found")
    db.delete_complaint(complaint_id)
    return {"deleted": complaint_id}


# ---------- KPIs ----------

@app.get("/api/kpis")
def kpis():
    records = db.list_complaints()
    total = len(records)
    analyzed = sum(1 for r in records if r["status"] == "analyzed")
    edited = sum(1 for r in records if r["edited"])
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for r in records:
        if r["category"]:
            by_category[r["category"]] = by_category.get(r["category"], 0) + 1
        if r["severity"]:
            by_severity[r["severity"]] = by_severity.get(r["severity"], 0) + 1

    # Rough "time saved" estimate: assume 4 minutes of manual triage per
    # complaint vs ~10 seconds of review time once AI-summarized.
    minutes_saved = round(analyzed * (4 - (10 / 60)), 1)
    accuracy_proxy = round(100 * (1 - (edited / analyzed)), 1) if analyzed else None

    return {
        "total_complaints": total,
        "analyzed": analyzed,
        "pending": total - analyzed,
        "edited_after_review": edited,
        "estimated_minutes_saved": minutes_saved,
        "unedited_rate_pct": accuracy_proxy,  # proxy for summarization "accuracy"
        "by_category": by_category,
        "by_severity": by_severity,
    }


# ---------- Export ----------

@app.get("/api/export/csv")
def export_csv():
    records = db.list_complaints()
    csv_bytes = to_csv(records)
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=complaint_summaries.csv"},
    )


@app.get("/api/export/pdf")
def export_pdf():
    records = db.list_complaints()
    pdf_bytes = to_pdf(records)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=complaint_summaries.pdf"},
    )
