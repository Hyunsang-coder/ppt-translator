#!/usr/bin/env python
"""Run iterative color matching verification against real LLM providers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.models import DEFAULT_LIGHT_MODEL
from src.utils.color_match_verifier import run_color_matching_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify PPT multi-color text matching over one or more iterations.",
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=("anthropic", "openai"),
        help="Provider to test. Repeat for multiple providers. Defaults to both.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of verification iterations per provider.",
    )
    parser.add_argument(
        "--anthropic-model",
        default=DEFAULT_LIGHT_MODEL["anthropic"],
        help="Anthropic model for color distribution.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_LIGHT_MODEL["openai"],
        help="OpenAI model for color distribution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated input/output PPTX artifacts. Omit to use temporary files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    providers = args.provider or ["anthropic", "openai"]
    models = {
        "anthropic": args.anthropic_model,
        "openai": args.openai_model,
    }

    results = run_color_matching_loop(
        providers,
        iterations=max(1, args.iterations),
        models=models,
        output_dir=args.output_dir,
    )

    failures = 0
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"[{status}] {result.provider} / {result.model} "
            f"iteration {result.iteration}"
        )
        if result.output_path is not None:
            print(f"  output: {result.output_path}")

        for case in result.cases:
            case_status = "PASS" if case.passed else "FAIL"
            print(f"  - [{case_status}] {case.name}: {case.output_text}")
            for error in case.errors:
                print(f"      {error}")

        if not result.passed:
            failures += 1

    if failures:
        print(f"\nColor matching verification failed: {failures} iteration(s)")
        return 1

    print(f"\nColor matching verification passed: {len(results)} iteration(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
