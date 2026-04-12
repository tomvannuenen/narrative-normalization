"""
Run protocol robustness test: Compare baseline (user prompt) vs system prompt placement.

Based on "The Fragility of Moral Judgment in LLMs" finding that protocol
variations cause larger verdict shifts than content changes.

Tests whether prompt effects are stable across different protocol designs.

Usage:
    python scripts/run_protocol_robustness_test.py --prompt generic --dry-run
    python scripts/run_protocol_robustness_test.py --prompt generic
    python scripts/run_protocol_robustness_test.py --prompt all

Author: Claude
Date: 2026-03-08
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import REWRITE_MODELS, PROMPT_CONDITIONS, REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS

# Convert REWRITE_MODELS list to dict format
MODELS = {model['label']: {'name': model['model'], 'provider': model['provider']}
          for model in REWRITE_MODELS}

# Paths
DATA_DIR = Path("data")
SAMPLE_FILE = DATA_DIR / "robustness_tests/test2_protocol_sample.csv"
OUTPUT_DIR = DATA_DIR / "robustness_tests/protocol"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stories_for_protocol_test():
    """Load the 100 stories selected for protocol testing."""
    sample_ids = pd.read_csv(SAMPLE_FILE)
    print(f"Loaded {len(sample_ids)} stories for protocol test")
    print(f"\nDistribution by source:")
    print(sample_ids['data_source'].value_counts())

    # Load full sample data
    full_sample = pd.read_parquet(DATA_DIR / "processed/sample.parquet")

    # Filter to protocol test IDs
    test_stories = full_sample[full_sample['id'].isin(sample_ids['story_id'])]

    print(f"\nLoaded {len(test_stories)} stories with full text")
    return test_stories


def call_api_with_system_prompt(
    text: str,
    instruction: str,
    model_key: str,
    model_name: str,
    temperature: float = 0.7
) -> str:
    """
    Call API with instruction in SYSTEM message and text in USER message.

    This is the protocol variant we're testing against the baseline
    (where both instruction and text are in the user message).

    Args:
        text: The story text to rewrite
        instruction: The rewriting instruction (e.g., "Please improve...")
        model_key: 'gpt54', 'claude_sonnet', 'gemini_31_pro'
        model_name: Actual model name for API
        temperature: Sampling temperature

    Returns:
        Rewritten text
    """
    if model_key.startswith('gpt'):
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": text}
            ],
            max_completion_tokens=REWRITE_MAX_TOKENS,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    elif model_key == 'claude_sonnet':
        from anthropic import Anthropic
        client = Anthropic()

        response = client.messages.create(
            model=model_name,
            max_tokens=REWRITE_MAX_TOKENS,
            temperature=temperature,
            system=instruction,  # Anthropic uses 'system' parameter
            messages=[{"role": "user", "content": text}],
        )
        return response.content[0].text.strip()

    elif model_key.startswith('gemini'):
        import time
        from google import genai
        client = genai.Client()

        # Google doesn't have a clear system message concept,
        # but we can prepend instruction with special formatting
        full_prompt = f"INSTRUCTION: {instruction}\n\nTEXT TO PROCESS:\n{text}"

        # Rate limiting for Gemini (free tier: 15 RPM = 1 per 4 seconds)
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config={
                        "max_output_tokens": REWRITE_MAX_TOKENS,
                        "temperature": temperature,
                    },
                )
                result = response.text.strip()
                time.sleep(4)  # Rate limit
                return result
            except Exception as e:
                if '429' in str(e) or 'TooManyRequests' in str(e):
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"  [Rate limit] Waiting {wait_time}s before retry...")
                        time.sleep(wait_time)
                    else:
                        raise
                else:
                    raise

    else:
        raise ValueError(f"Unknown model key: {model_key}")


def run_protocol_test(prompt_condition: str, protocol_variant: str, dry_run: bool = False):
    """
    Run protocol robustness test.

    Args:
        prompt_condition: 'generic', 'voice_preserving', or 'rewrite_only'
        protocol_variant: 'baseline' (user msg) or 'system' (system msg)
        dry_run: If True, only print what would be done
    """
    print("\n" + "="*80)
    print(f"PROTOCOL ROBUSTNESS TEST: {prompt_condition.upper()} - {protocol_variant.upper()}")
    print("="*80)

    # Load stories
    stories = load_stories_for_protocol_test()

    # Get prompt config
    if prompt_condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown prompt: {prompt_condition}")

    prompt_config = PROMPT_CONDITIONS[prompt_condition]
    prompt_template = prompt_config['prompt']

    # Extract instruction and format
    # The template is like: "Please improve the following story...\n\n{text}"
    # Split on {text} to get instruction vs placeholder
    if '{text}' in prompt_template:
        instruction = prompt_template.replace('{text}', '').strip()
    else:
        raise ValueError("Prompt template must contain {text} placeholder")

    print(f"\nInstruction: {instruction[:100]}...")
    print(f"Protocol: {protocol_variant}")
    print(f"Temperature: {REWRITE_TEMPERATURE}")
    print(f"Number of stories: {len(stories)}")
    print(f"Models: {list(MODELS.keys())}")

    if dry_run:
        print("\n[DRY RUN] Would rewrite:")
        print(f"  - {len(stories)} stories")
        print(f"  - {len(MODELS)} models")
        print(f"  - 1 prompt condition ({prompt_condition})")
        print(f"  - 1 protocol variant ({protocol_variant})")
        print(f"  - Total: {len(stories) * len(MODELS)} rewrites")
        return

    # Prepare data
    story_ids = stories['id'].tolist()
    story_texts = stories['story'].tolist()

    print(f"\nStarting rewrites...")

    # Check for existing checkpoint and resume if possible
    output_file = OUTPUT_DIR / f"protocol_{prompt_condition}_{protocol_variant}.parquet"

    if output_file.exists():
        print(f"\n✓ Found checkpoint: {output_file}")
        results = pd.read_parquet(output_file)
        print(f"  Loaded {len(results)} stories from checkpoint")
    else:
        results = stories.copy()

    for model_key, model_config in MODELS.items():
        col_name = f"rewrite_{model_key}_{protocol_variant}"

        # Check if this model is already complete
        if col_name in results.columns and results[col_name].notna().all():
            print(f"\n{'='*60}")
            print(f"Model: {model_key}")
            print(f"{'='*60}")
            print(f"✓ Already completed - skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")
        print(f"{'='*60}")

        model_name = model_config['name']
        rewrites = []

        for i, (story_id, text) in enumerate(zip(story_ids, story_texts)):
            try:
                if protocol_variant == 'baseline':
                    # Baseline: instruction + text in user message (existing behavior)
                    # This would use the normal rewriter, but for consistency let's implement here
                    full_prompt = prompt_template.format(text=text)

                    if model_key.startswith('gpt'):
                        from openai import OpenAI
                        client = OpenAI()
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[{"role": "user", "content": full_prompt}],
                            max_completion_tokens=REWRITE_MAX_TOKENS,
                            temperature=REWRITE_TEMPERATURE,
                        )
                        rewrite = response.choices[0].message.content.strip()

                    elif model_key == 'claude_sonnet':
                        from anthropic import Anthropic
                        client = Anthropic()
                        response = client.messages.create(
                            model=model_name,
                            max_tokens=REWRITE_MAX_TOKENS,
                            temperature=REWRITE_TEMPERATURE,
                            messages=[{"role": "user", "content": full_prompt}],
                        )
                        rewrite = response.content[0].text.strip()

                    elif model_key.startswith('gemini'):
                        import time
                        from google import genai
                        client = genai.Client()

                        # Rate limiting for Gemini (free tier: 15 RPM = 1 per 4 seconds)
                        max_retries = 3
                        retry_delay = 5

                        for attempt in range(max_retries):
                            try:
                                response = client.models.generate_content(
                                    model=model_name,
                                    contents=full_prompt,
                                    config={
                                        "max_output_tokens": REWRITE_MAX_TOKENS,
                                        "temperature": REWRITE_TEMPERATURE,
                                    },
                                )
                                rewrite = response.text.strip()
                                time.sleep(4)  # Rate limit
                                break
                            except Exception as e:
                                if '429' in str(e) or 'TooManyRequests' in str(e):
                                    if attempt < max_retries - 1:
                                        wait_time = retry_delay * (attempt + 1)
                                        print(f"  [Rate limit] Waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                                        time.sleep(wait_time)
                                    else:
                                        print(f"  [!] Rate limit exceeded after {max_retries} retries")
                                        raise
                                else:
                                    raise

                elif protocol_variant == 'system':
                    # System variant: instruction in system, text in user
                    rewrite = call_api_with_system_prompt(
                        text=text,
                        instruction=instruction,
                        model_key=model_key,
                        model_name=model_name,
                        temperature=REWRITE_TEMPERATURE
                    )
                else:
                    raise ValueError(f"Unknown protocol variant: {protocol_variant}")

                rewrites.append(rewrite)

                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(story_ids)}")

            except Exception as e:
                print(f"  [!] Error on story {story_id}: {e}")
                rewrites.append(None)

        # Add to results
        col_name = f"rewrite_{model_key}_{protocol_variant}"
        results[col_name] = rewrites

        # Report success rate
        n_success = sum(1 for r in rewrites if r is not None)
        print(f"✓ {n_success}/{len(story_ids)} rewrites successful ({n_success/len(story_ids)*100:.1f}%)")

        # SAVE CHECKPOINT AFTER EACH MODEL (critical for resume!)
        results.to_parquet(output_file)
        print(f"💾 Checkpoint saved: {output_file}")

    # Final save
    results.to_parquet(output_file)
    print(f"\n✓ All models complete. Final save: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run protocol robustness test")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        choices=['generic', 'voice_preserving', 'rewrite_only', 'all'],
        help="Which prompt condition to run (or 'all')"
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default=None,
        choices=['baseline', 'system'],
        help="Which protocol variant (baseline=user msg, system=system msg). If not specified, runs both."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running"
    )

    args = parser.parse_args()

    # Determine which prompts to run
    if args.prompt == 'all':
        prompts = ['generic', 'voice_preserving', 'rewrite_only']
    else:
        prompts = [args.prompt]

    # Determine which protocol variants to execute
    if args.protocol is not None:
        protocols = [args.protocol]
    else:
        protocols = ['baseline', 'system']

    # Execute all combinations
    for prompt in prompts:
        for protocol in protocols:
            run_protocol_test(
                prompt_condition=prompt,
                protocol_variant=protocol,
                dry_run=args.dry_run
            )

    if not args.dry_run:
        print("\n" + "="*80)
        print("PROTOCOL ROBUSTNESS TEST COMPLETE")
        print("="*80)
        print(f"\nCompleted:")
        print(f"  Prompts: {prompts}")
        print(f"  Protocols: {protocols}")
        print(f"  Total combinations: {len(prompts) * len(protocols)}")
        print(f"\nNext step: python scripts/analyze_protocol_robustness.py")


if __name__ == "__main__":
    main()
