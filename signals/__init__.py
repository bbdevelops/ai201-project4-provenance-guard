"""Detection signals for Provenance Guard.

Each signal exposes a function that takes the submitted text and returns the
standardized signal contract: a dict ``{"score": float|None, "status": str}``
(plus any signal-specific extras). The isolated confidence scorer (M4) consumes
these objects; the Flask routes never do scoring math themselves.
"""
