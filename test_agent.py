#!/usr/bin/env python3
"""
Test Agent CLI Runner
=====================
Command-line runner to test that the LLM and Hugging Face token for BERT scoring
are working properly.

Usage:
    python test_agent.py
    python test_agent.py --json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Prevent OpenBLAS/MKL thread allocation issues on Windows Python 3.13
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.agentic_graphrag.agents.verification_agent import VerificationAgent


def main():
    parser = argparse.ArgumentParser(description="Test LLM Gateway & HF Token BERT Scoring Agent")
    parser.add_argument("--json", action="store_true", help="Output full report in JSON format")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs")
    args = parser.parse_args()

    agent = VerificationAgent(verbose=not args.quiet and not args.json)
    report = agent.run_all()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))

    # Exit code: 0 if healthy/degraded, 1 if critical
    sys.exit(0 if report.overall_status in ("HEALTHY", "DEGRADED") else 1)


if __name__ == "__main__":
    main()
