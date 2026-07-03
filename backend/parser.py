import json
import re


def clean_json(response: str) -> str:
    """
    Removes markdown code blocks if present.
    """

    response = response.strip()

    response = re.sub(r"```json", "", response, flags=re.IGNORECASE)
    response = re.sub(r"```", "", response)

    return response.strip()


def parse_response(response: str):
    """
    Parse LLM JSON response safely.
    """

    response = clean_json(response)

    try:
        return json.loads(response)

    except Exception:

        return {
            "summary": response,
            "category": "Unknown",
            "severity": "Unknown",
            "sentiment": "Unknown",
            "root_cause": "Unknown",
            "recommended_action": "Manual Review Required"
        }