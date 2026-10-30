"""Shared pytest settings."""

import matplotlib

# Draw into memory only, so tests never open a window.
matplotlib.use("Agg")
