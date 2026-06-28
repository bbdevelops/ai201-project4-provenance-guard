"""Central configuration for Provenance Guard.

Loads secrets from .env (never committed) and exposes the small set of values
the rest of the system needs: the Groq API key, the model name, the audit-log
database path, and a factory for the Groq client.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Secrets / tunables -----------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Signal 1 model. Free-tier Groq model specified in the project stack.
MODEL = "llama-3.3-70b-versatile"

# SQLite audit log lives next to the code, in the repo root.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_log.db")


def get_groq_client():
    """Return a Groq client. Raises a clear error if the key is missing."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file (see README)."
        )
    return Groq(api_key=GROQ_API_KEY)
