"""
Run temperature sensitivity test: same stories at multiple temperatures.

Tests whether normalization patterns are stable across temperature settings,
or if they are artifacts of our default temperature (0.7).

Design:
- 50 stories (same subsample as self-consistency test for comparability)
- 3 models (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro)
- 3 temperatures: 0.0 (deterministic), 0.7 (default), 1.0 (high stochasticity)
- 1 prompt condition: generic (baseline)

API Mode (matches main rewriter exactly):
- OpenAI: Batch API (50% cheaper)
- Anthropic: Batch API (50% cheaper)
- Gemini: Sync mode (no batch support)

Total rewrites: 50 x 3 x 3 = 450

Usage:
    python scripts/run_temperature_sensitivity_test.py --dry-run
    python scripts/run_temperature_sensitivity_test.py
    python scripts/run_temperature_sensitivity_test.py --temp 0.0  # single temp
    python scripts/run_temperature_sensitivity_test.py --poll      # poll pending batches

Author: Claude
Date: 2026-03-31
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import REWRITE_MODELS, PROMPT_CONDITIONS, REWRITE_MAX_TOKENS

# Convert REWRITE_MODELS list to dict format
MODELS = {model['label']: {'name': model['model'], 'provider': model['provider']}
          for model in REWRITE_MODELS}

# Temperature values to test
TEMPERATURES = [0.0, 0.7, 1.0]

# Paths
DATA_DIR = Path("data")
SAMPLE_FILE = DATA_DIR / "robustness_tests/test1_self_consistency_sample.csv"
OUTPUT_DIR = DATA_DIR / "robustness_tests/temperature"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System message to prevent preamble/commentary (matches main rewriter exactly)
OUTPUT_FORMAT_SYSTEM_MESSAGE = (
    "You are a writing assistant. Return only the rewritten story. "
    "Do not include any commentary, preamble, multiple versions, headers, "
    "or explanation. Output the story text only."
)


def load_stories_for_temperature_test():
    """Load the 50 stories selected for testing (same as self-consistency)."""
    sample_ids = pd.read_csv(SAMPLE_FILE)
    print(f"Loaded {len(sample_ids)} stories for temperature test")
    print(f"\nDistribution by source:")
    print(sample_ids['data_source'].value_counts())

    full_sample = pd.read_parquet(DATA_DIR / "processed/sample.parquet")
    test_stories = full_sample[full_sample['id'].isin(sample_ids['story_id'])]

    print(f"\nLoaded {len(test_stories)} stories with full text")
    return test_stories


# ═══════════════════════════════════════════════════════════════════════════
# BATCH API — OpenAI (matches main rewriter)
# ═══════════════════════════════════════════════════════════════════════════

def _submit_openai_batch(
    stories: list[tuple[int, str]],
    model: str,
    label: str,
    prompt: str,
    temperature: float,
) -> str:
    """Create a .jsonl file of requests and submit an OpenAI batch job."""
    from openai import OpenAI

    client = OpenAI()
    jsonl_path = OUTPUT_DIR / f"batch_input_{label}.jsonl"

    with open(jsonl_path, "w") as f:
        for story_id, text in stories:
            request = {
                "custom_id": str(story_id),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": OUTPUT_FORMAT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt.format(text=text)},
                    ],
                    "max_completion_tokens": REWRITE_MAX_TOKENS,
                    "temperature": temperature,
                },
            }
            f.write(json.dumps(request) + "\n")

    # Upload the file
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    # Create the batch
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"  [OpenAI] Batch submitted: {batch.id}")
    meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"
    meta_path.write_text(json.dumps({
        "batch_id": batch.id,
        "label": label,
        "temperature": temperature,
        "model": model,
    }))
    return batch.id


def _poll_openai_batch(label: str, poll_interval: int = 60) -> dict[int, str]:
    """Poll until the OpenAI batch completes. Returns {story_id: rewrite}."""
    from openai import OpenAI

    client = OpenAI()
    meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"
    meta = json.loads(meta_path.read_text())
    batch_id = meta["batch_id"]

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  [OpenAI] Batch {batch_id}: {status}")

        if status == "completed":
            break
        elif status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"OpenAI batch {batch_id} ended with status: {status}")

        time.sleep(poll_interval)

    # Download results
    content = client.files.content(batch.output_file_id)
    results = {}
    for line in content.text.strip().split("\n"):
        obj = json.loads(line)
        story_id = int(obj["custom_id"])
        if obj.get("error"):
            print(f"  [!] story {story_id} failed: {obj['error']}")
            continue
        text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
        results[story_id] = text

    return results


# ═══════════════════════════════════════════════════════════════════════════
# BATCH API — Anthropic (matches main rewriter)
# ═══════════════════════════════════════════════════════════════════════════

def _submit_anthropic_batch(
    stories: list[tuple[int, str]],
    model: str,
    label: str,
    prompt: str,
    temperature: float,
) -> str:
    """Submit an Anthropic Message Batch. Returns the batch ID."""
    from anthropic import Anthropic

    client = Anthropic()

    requests = []
    for story_id, text in stories:
        requests.append({
            "custom_id": str(story_id),
            "params": {
                "model": model,
                "max_tokens": REWRITE_MAX_TOKENS,
                "temperature": temperature,
                "system": OUTPUT_FORMAT_SYSTEM_MESSAGE,
                "messages": [
                    {"role": "user", "content": prompt.format(text=text)}
                ],
            },
        })

    batch = client.messages.batches.create(requests=requests)
    print(f"  [Anthropic] Batch submitted: {batch.id}")

    meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"
    meta_path.write_text(json.dumps({
        "batch_id": batch.id,
        "label": label,
        "temperature": temperature,
        "model": model,
    }))
    return batch.id


def _poll_anthropic_batch(label: str, poll_interval: int = 60) -> dict[int, str]:
    """Poll until the Anthropic batch completes. Returns {story_id: rewrite}."""
    from anthropic import Anthropic

    client = Anthropic()
    meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"
    meta = json.loads(meta_path.read_text())
    batch_id = meta["batch_id"]

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        print(f"  [Anthropic] Batch {batch_id}: {status}")

        if status == "ended":
            break

        time.sleep(poll_interval)

    # Stream results
    results = {}
    for result in client.messages.batches.results(batch_id):
        story_id = int(result.custom_id)
        if result.result.type == "succeeded":
            try:
                content = result.result.message.content
                if content and len(content) > 0:
                    text = content[0].text.strip()
                    results[story_id] = text
                else:
                    print(f"  [!] story {story_id} failed: empty content")
            except (AttributeError, IndexError) as e:
                print(f"  [!] story {story_id} failed: {e}")
        else:
            print(f"  [!] story {story_id} failed: {result.result.type}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# SYNC API — Gemini (no batch support, matches main rewriter exactly)
# ═══════════════════════════════════════════════════════════════════════════

def _call_gemini_sync(text: str, model: str, prompt: str, temperature: float) -> str:
    """Call Google/Gemini API with system instruction (sync mode)."""
    from google import genai
    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt.format(text=text),
        config={
            "max_output_tokens": REWRITE_MAX_TOKENS,
            "temperature": temperature,
            "system_instruction": OUTPUT_FORMAT_SYSTEM_MESSAGE,
        },
    )
    return response.text.strip()


def _call_with_retry(fn, *args, max_retries: int = 3, base_wait: int = 60, **kwargs):
    """Call fn with exponential backoff on rate-limit errors.

    Matches main rewriter exactly:
    - Detects 429 / RESOURCE_EXHAUSTED responses
    - Parses retry-after hints from error messages
    - Exponential backoff: base_wait * 2^attempt (60→120→240)
    - Detects daily quota and raises immediately
    """
    import re
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()
            if not is_rate_limit or attempt == max_retries:
                raise

            # Check for daily quota — no point waiting
            if "per_day" in msg.lower() or "quota exceeded" in msg.lower():
                print(f"  [!] Daily quota exceeded — stopping sync run.")
                raise

            # Parse retry-after hint if present
            wait = base_wait * (2 ** attempt)
            match = re.search(r"retry[_ ]?(?:after|in)[:\s]+(\d+)s", msg, re.IGNORECASE)
            if match:
                wait = int(match.group(1)) + 5  # small buffer

            print(f"  [rate limit] attempt {attempt + 1}/{max_retries}, waiting {wait}s…")
            time.sleep(wait)


def _rewrite_gemini_sync(
    stories: list[tuple[int, str]],
    model: str,
    prompt: str,
    temperature: float,
    results_df: pd.DataFrame,
    col_name: str,
    output_file: Path,
) -> None:
    """Rewrite stories using Gemini sync mode with checkpointing.

    Matches main rewriter exactly:
    - Uses _call_with_retry for exponential backoff
    - 0.5s sleep between calls
    - Checkpoint every 10 stories
    - Daily quota detection
    """
    id_to_idx = {row['id']: idx for idx, row in results_df.iterrows()}

    completed = 0
    skipped = 0
    start_time = time.time()

    for story_id, text in stories:
        idx = id_to_idx.get(story_id)
        if idx is None:
            continue

        # Skip if already has a value
        if pd.notna(results_df.at[idx, col_name]):
            skipped += 1
            continue

        try:
            # Use same retry logic as main rewriter
            rewrite = _call_with_retry(
                _call_gemini_sync, text, model, prompt, temperature
            )
            results_df.at[idx, col_name] = rewrite
            completed += 1

        except Exception as e:
            print(f"  [!] id={story_id} failed: {e}")
            # Check for daily quota - save and exit
            if "per_day" in str(e).lower() or "quota exceeded" in str(e).lower():
                results_df.to_parquet(output_file)
                print(f"  [Gemini] Daily quota hit — checkpoint saved. Re-run tomorrow.")
                return

        # Rate limiting (matches main rewriter)
        time.sleep(0.5)

        # Progress and checkpoint every 10
        total_done = completed + skipped
        if total_done % 10 == 0:
            results_df.to_parquet(output_file)
            elapsed = time.time() - start_time
            avg = elapsed / completed if completed > 0 else 0
            remaining = (len(stories) - total_done) * avg
            print(f"  [{col_name}] checkpoint at {total_done}/{len(stories)} | "
                  f"Avg: {avg:.1f}s | ETA: {remaining/60:.1f}min")

    results_df.to_parquet(output_file)
    print(f"  ✓ Gemini: {completed} new, {skipped} from checkpoint")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def run_temperature_test(temperature: float, dry_run: bool = False, poll_only: bool = False):
    """
    Run rewrites at a specific temperature using batch mode (OpenAI/Anthropic)
    and sync mode (Gemini), exactly matching the main rewriter.
    """
    temp_str = str(temperature).replace('.', '_')
    print("\n" + "="*80)
    print(f"TEMPERATURE SENSITIVITY TEST: temp={temperature}")
    print("="*80)

    # Load stories
    stories = load_stories_for_temperature_test()
    prompt_template = PROMPT_CONDITIONS['generic']['prompt']

    print(f"\nPrompt: generic")
    print(f"Temperature: {temperature}")
    print(f"Stories: {len(stories)}")
    print(f"Mode: Batch (OpenAI/Anthropic) + Sync (Gemini)")
    print(f"System message: YES (matches main rewriter)")

    if dry_run:
        print("\n[DRY RUN] Would submit:")
        print(f"  - OpenAI batch: 50 stories")
        print(f"  - Anthropic batch: 50 stories")
        print(f"  - Gemini sync: 50 stories")
        print(f"  - Total: 150 rewrites at temp={temperature}")
        return

    # Prepare story list for batch submission
    story_list = [(int(row['id']), row['story']) for _, row in stories.iterrows()]

    # Output file
    output_file = OUTPUT_DIR / f"temperature_{temp_str}.parquet"

    # Load or create results DataFrame
    if output_file.exists():
        results = pd.read_parquet(output_file)
        print(f"\n✓ Loaded checkpoint: {len(results)} stories")
    else:
        results = stories.copy()

    # Ensure all model columns exist (handles partial checkpoints)
    for model_key in MODELS.keys():
        col = f"rewrite_{model_key}"
        if col not in results.columns:
            results[col] = None

    # ── BATCH MODELS (OpenAI, Anthropic) ──────────────────────────────────
    batch_models = [
        ("gpt54", "openai", MODELS["gpt54"]["name"]),
        ("claude_sonnet", "anthropic", MODELS["claude_sonnet"]["name"]),
    ]

    pending_batches = []

    for model_key, provider, model_name in batch_models:
        col_name = f"rewrite_{model_key}"
        label = f"temp_{temp_str}_{model_key}"
        meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"

        # Check if already complete
        if col_name in results.columns and results[col_name].notna().all():
            print(f"\n[{model_key}] ✓ Already complete")
            continue

        # Check if batch already submitted
        if meta_path.exists():
            print(f"\n[{model_key}] Batch already submitted, will poll")
            pending_batches.append((model_key, provider, label, col_name))
            continue

        if poll_only:
            print(f"\n[{model_key}] No pending batch (--poll mode)")
            continue

        # Submit new batch
        print(f"\n[{model_key}] Submitting batch...")
        if provider == "openai":
            _submit_openai_batch(story_list, model_name, label, prompt_template, temperature)
        else:
            _submit_anthropic_batch(story_list, model_name, label, prompt_template, temperature)

        pending_batches.append((model_key, provider, label, col_name))

    # Poll for batch results
    for model_key, provider, label, col_name in pending_batches:
        meta_path = OUTPUT_DIR / f"batch_meta_{label}.json"
        if not meta_path.exists():
            continue

        print(f"\n[{model_key}] Polling for results...")

        if provider == "openai":
            batch_results = _poll_openai_batch(label)
        else:
            batch_results = _poll_anthropic_batch(label)

        # Merge results
        id_to_idx = {row['id']: idx for idx, row in results.iterrows()}
        filled = 0
        for story_id, rewrite_text in batch_results.items():
            if story_id in id_to_idx:
                results.at[id_to_idx[story_id], col_name] = rewrite_text
                filled += 1

        print(f"[{model_key}] Got {filled}/{len(batch_results)} results")
        results.to_parquet(output_file)

        # Clean up batch metadata
        meta_path.unlink()
        jsonl_path = OUTPUT_DIR / f"batch_input_{label}.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()

    # ── SYNC MODEL (Gemini) ───────────────────────────────────────────────
    gemini_key = "gemini_31_pro"
    gemini_col = f"rewrite_{gemini_key}"
    gemini_model = MODELS[gemini_key]["name"]

    if gemini_col in results.columns and results[gemini_col].notna().all():
        print(f"\n[{gemini_key}] ✓ Already complete")
    elif not poll_only:
        print(f"\n[{gemini_key}] Running sync mode...")
        _rewrite_gemini_sync(
            story_list, gemini_model, prompt_template, temperature,
            results, gemini_col, output_file
        )

    # Final save
    results.to_parquet(output_file)
    print(f"\n✓ Temperature {temperature} saved: {output_file}")

    # Summary
    for model_key in MODELS.keys():
        col = f"rewrite_{model_key}"
        n = results[col].notna().sum() if col in results.columns else 0
        print(f"  {model_key}: {n}/{len(results)}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run temperature sensitivity test")
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        choices=[0.0, 0.7, 1.0],
        help="Which temperature to run. If not specified, runs all 3."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running"
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Only poll for pending batch results (don't submit new batches)"
    )

    args = parser.parse_args()

    temps = [args.temp] if args.temp is not None else TEMPERATURES

    for temp in temps:
        run_temperature_test(temp, dry_run=args.dry_run, poll_only=args.poll)

    if not args.dry_run:
        print("\n" + "="*80)
        print("TEMPERATURE SENSITIVITY TEST")
        print("="*80)
        print(f"\nTemperatures processed: {temps}")
        print(f"\nNote: Batch jobs (OpenAI/Anthropic) may take up to 24h.")
        print(f"Re-run with --poll to check status and collect results.")
        print(f"\nAfter all data collected:")
        print(f"  python scripts/analyze_temperature_sensitivity.py")


if __name__ == "__main__":
    main()
