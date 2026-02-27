#!/usr/bin/env python3
"""CLI wrapper to run the field model comparison example."""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from examples.compare_field_models import main
main()
