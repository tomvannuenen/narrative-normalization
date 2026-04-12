"""
Run self-consistency test: 3 independent runs of same story/model/prompt combination.

Tests whether observed prompt effects exceed baseline stochastic variation.
Critical for interpreting main study findings.

Usage:
    python scripts/run_self_consistency_test.py --prompt generic --dry-run
    python scripts/run_self_consistency_test.py --prompt generic
    python scripts/run_self_consistency_test.py --prompt all  # all 3 prompts

Author: Claude
Date: 2026-03-08
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import REWRITE_MODELS, PROMPT_CONDITIONS, REWRITE_TEMPERATURE, REWRITE_MAX_TOKENS

# Convert REWRITE_MODELS list to dict format
MODELS = {model['label']: {'name': model['model'], 'provider': model['provider']}
          for model in REWRITE_MODELS}

# Paths
DATA_DIR = Path("data")
SAMPLE_FILE = DATA_DIR / "robustness_tests/test1_self_consistency_sample.csv"
OUTPUT_DIR = DATA_DIR / "robustness_tests/self_consistency"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_stories_for_consistency_test():
    """Load the 50 stories selected for consistency testing."""
    # Load sample IDs
    sample_ids = pd.read_csv(SAMPLE_FILE)
    print(f"Loaded {len(sample_ids)} stories for consistency test")
    print(f"\nDistribution by source:")
    print(sample_ids['data_source'].value_counts())

    # Load full sample data
    full_sample = pd.read_parquet(DATA_DIR / "processed/sample.parquet")

    # Filter to consistency test IDs
    test_stories = full_sample[full_sample['id'].isin(sample_ids['story_id'])]

    print(f"\nLoaded {len(test_stories)} stories with full text")
    return test_stories


def run_consistency_test(prompt_condition: str, run_number: int, dry_run: bool = False):
    """
    Run one iteration of the consistency test.

    Args:
        prompt_condition: 'generic', 'voice_preserving', or 'rewrite_only'
        run_number: 1, 2, or 3
        dry_run: If True, only print what would be done
    """
    print("\n" + "="*80)
    print(f"SELF-CONSISTENCY TEST: {prompt_condition.upper()} - Run {run_number}/3")
    print("="*80)

    # Load stories
    stories = load_stories_for_consistency_test()

    # Get prompt config
    if prompt_condition not in PROMPT_CONDITIONS:
        raise ValueError(f"Unknown prompt: {prompt_condition}")

    prompt_config = PROMPT_CONDITIONS[prompt_condition]
    prompt_template = prompt_config['prompt']

    print(f"\nPrompt template: {prompt_template[:100]}...")
    print(f"Temperature: {REWRITE_TEMPERATURE}")
    print(f"Number of stories: {len(stories)}")
    print(f"Models: {list(MODELS.keys())}")

    if dry_run:
        print("\n[DRY RUN] Would rewrite:")
        print(f"  - {len(stories)} stories")
        print(f"  - {len(MODELS)} models")
        print(f"  - 1 prompt condition ({prompt_condition})")
        print(f"  - Total: {len(stories) * len(MODELS)} rewrites")
        return

    # Prepare data for rewriting
    texts_to_rewrite = stories['story'].tolist()
    story_ids = stories['id'].tolist()

    print(f"\nStarting rewrites...")

    # Check for existing checkpoint and resume if possible
    output_file = OUTPUT_DIR / f"consistency_{prompt_condition}_run{run_number}.parquet"

    if output_file.exists():
        print(f"\n✓ Found checkpoint: {output_file}")
        results = pd.read_parquet(output_file)
        print(f"  Loaded {len(results)} stories from checkpoint")
    else:
        results = stories.copy()

    for model_key, model_config in MODELS.items():
        col_name = f"rewrite_{model_key}_run{run_number}"

        # Check if this model is already complete
        if col_name in results.columns and results[col_name].notna().all():
            print(f"\n{'='*60}")
            print(f"Model: {model_key}")
            print(f"{'='*60}")
            print(f"✓ Already completed - skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_key}")

        # Show API key for Gemini to verify which one is being used
        if model_key.startswith('gemini'):
            import os
            api_key = os.getenv('GOOGLE_API_KEY', 'NOT SET')
            if api_key != 'NOT SET':
                print(f"API Key: ...{api_key[-8:]}")

        print(f"{'='*60}")

        model_name = model_config['name']
        rewrites = []

        # Track timing for progress feedback
        import time
        start_time = time.time()

        for i, (story_id, text) in enumerate(zip(story_ids, texts_to_rewrite)):
            story_start = time.time()
            try:
                # Format prompt
                full_prompt = prompt_template.format(text=text)

                # Call appropriate API based on model
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

                    # Retry logic for transient errors (no rate limiting delay)
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
                            break  # Success - no artificial delay

                        except Exception as e:
                            if '429' in str(e) or 'TooManyRequests' in str(e) or 'timeout' in str(e).lower():
                                if attempt < max_retries - 1:
                                    wait_time = retry_delay * (attempt + 1)
                                    print(f"  [Retry] Waiting {wait_time}s before retry {attempt+2}/{max_retries}...")
                                    time.sleep(wait_time)
                                else:
                                    print(f"  [!] Failed after {max_retries} retries")
                                    raise
                            else:
                                raise

                else:
                    raise ValueError(f"Unknown model key: {model_key}")

                rewrites.append(rewrite)

                # Enhanced progress feedback
                story_elapsed = time.time() - story_start
                if (i + 1) % 10 == 0:
                    total_elapsed = time.time() - start_time
                    avg_time = total_elapsed / (i + 1)
                    remaining = (len(story_ids) - (i + 1)) * avg_time
                    print(f"  Progress: {i+1}/{len(story_ids)} | "
                          f"Last story: {story_elapsed:.1f}s | "
                          f"Avg: {avg_time:.1f}s/story | "
                          f"ETA: {remaining/60:.1f} min")
                elif model_key.startswith('gemini'):
                    # Show every story for Gemini to see speed
                    print(f"  Story {i+1}/{len(story_ids)}: {story_elapsed:.1f}s")

            except Exception as e:
                story_elapsed = time.time() - story_start
                print(f"  [!] Error on story {story_id} after {story_elapsed:.1f}s: {e}")
                rewrites.append(None)

        # Add to results
        col_name = f"rewrite_{model_key}_run{run_number}"
        results[col_name] = rewrites

        # Report success rate
        n_success = sum(1 for r in rewrites if r is not None)
        print(f"✓ {n_success}/{len(story_ids)} rewrites successful ({n_success/len(story_ids)*100:.1f}%)")

        # SAVE CHECKPOINT AFTER EACH MODEL (critical for resume!)
        results.to_parquet(output_file)
        print(f"💾 Checkpoint saved: {output_file}")

    # Final save
    results.to_parquet(output_file)
    print(f"\n✓ All runs complete. Final save: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run self-consistency test")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        choices=['generic', 'voice_preserving', 'rewrite_only', 'all'],
        help="Which prompt condition to run (or 'all')"
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        choices=[1, 2, 3],
        help="Which run (1-3). If not specified, runs all 3."
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

    # Determine which runs to execute
    if args.run is not None:
        runs = [args.run]
    else:
        runs = [1, 2, 3]

    # Execute all combinations
    for prompt in prompts:
        for run_num in runs:
            run_consistency_test(
                prompt_condition=prompt,
                run_number=run_num,
                dry_run=args.dry_run
            )

    if not args.dry_run:
        print("\n" + "="*80)
        print("SELF-CONSISTENCY TEST COMPLETE")
        print("="*80)
        print(f"\nCompleted:")
        print(f"  Prompts: {prompts}")
        print(f"  Runs: {runs}")
        print(f"  Total combinations: {len(prompts) * len(runs)}")
        print(f"\nNext step: python scripts/analyze_self_consistency.py")


if __name__ == "__main__":
    main()
