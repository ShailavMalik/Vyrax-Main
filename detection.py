"""Compatibility wrapper for the emotion engine.

The real implementation now lives in emotion_engine.py. This file remains so
older imports keep working without changes.
"""

from emotion_engine import *  # noqa: F401,F403
