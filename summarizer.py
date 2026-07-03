"""
Calls the Anthropic API to generate a concise, standardized summary and a
category + severity classification for a manufacturing customer complaint.
"""
import json
import os
from anthropic import Anthropic

MODEL = "claude-sonnet-4-6"

CATEGORIES = [
    "Product Defect",
    "Performance Issue",
    "Packaging / Shipping Damage",
    "Delivery Delay",
    "Safety Concern",
    "Billing / Order Issue",
    "Customer Service",
    "Other",
]

SEVERITIES = ["Low", "Medium", "High", "Critical"]

SYSTEM_PROMPT = f"""You are a quality-analysis assistant for a manufacturing company.
You read a single customer complaint (already anonymized) and produce a standardized,
structured analysis for the Quality and Customer Service teams.

Respond ONLY with valid JSON, no preamble, no markdown fences, matching this schema exactly:
{{
  "summary": "1-2 sentence concise summary of the complaint in plain business language",
  "category": "one of: {', '.join(CATEGORIES)}",
  "severity": "one of: {', '.join(SEVERITIES)}",
  "root_cause_hypothesis": "short phrase, best guess at underlying cause, or 'Unclear' if not determinable",
  "suggested_action": "short, actionable recommendation for the quality/service team",
  "sentiment": "one of: Frustrated, Neutral, Angry, Disappointed, Satisfied"
}}

Be concise, consistent, and factual. Do not invent details not present or implied in the complaint."""


def _client() -> Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it before starting the server, e.g. export ANTHROPIC_API_KEY=sk-ant-..."
        )
    return Anthropic(api_key=api_key)


def analyze_complaint(anonymized_text: str) -> dict:
    """Send one complaint to Claude and return structured JSON analysis."""
    client = _client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Complaint text:\n\n{anonymized_text}"}],
    )
    raw = "".join(block.text for block in response.content if block.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "summary": raw[:300] if raw else "Could not generate summary.",
            "category": "Other",
            "severity": "Medium",
            "root_cause_hypothesis": "Unclear",
            "suggested_action": "Manual review recommended.",
            "sentiment": "Neutral",
        }
    data.setdefault("category", "Other")
    if data["category"] not in CATEGORIES:
        data["category"] = "Other"
    data.setdefault("severity", "Medium")
    if data["severity"] not in SEVERITIES:
        data["severity"] = "Medium"
    return data
