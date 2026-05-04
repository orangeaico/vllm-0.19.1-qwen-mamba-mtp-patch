#!/usr/bin/env python3
"""Client-side multi-turn vLLM metrics probe.

Run this from the host after a vLLM server is reachable. This script sends
multi-turn chat requests and records per-turn local prefill tokens, cached
prompt tokens, generated tokens, and generation TPS from vLLM `/metrics`.

Example 10-turn latest-Mamba probe:

    cd /path/to/vllm-0.19.1-qwen-mamba-mtp-patch
    RUN_DIR="artifacts/mamba_probe_$(date +%Y%m%d_%H%M%S)"
    SALT="mamba_probe_$(date +%s)"
    mkdir -p "$RUN_DIR"
    printf '%s\n' "$SALT" > "$RUN_DIR/cache_salt.txt"
    python3 scripts/multiturn_vllm_metrics.py \
      --base-url http://127.0.0.1:3004 \
      --model qwen3 \
      --turns 10 \
      --input-tokens 400 \
      --output-tokens 200 \
      --min-output-tokens 200 \
      --temperature 0 \
      --cache-salt "$SALT" \
      --out-dir "$RUN_DIR" \
      --trace-json

Use `--base-url http://127.0.0.1:3003` when the server is directly on port
3003, or `--base-url http://127.0.0.1:3004` when the container maps host 3004
to container 3003. If `--dataset-file` is omitted, the script downloads a small
public WikiText test split into `.cache/datasets/`.
"""

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+0-9.eE]+)$')
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="([^"]*)"')
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/"
    "word_language_model/data/wikitext-2/test.txt"
)


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def parse_metrics(text):
    metrics = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue

        name, labels_text, value_text = match.groups()
        labels = dict(LABEL_RE.findall(labels_text or ""))
        key = (name, tuple(sorted(labels.items())))
        metrics[key] = float(value_text)
    return metrics


def metric_value(metrics, name, **labels):
    for (metric_name, metric_labels), value in metrics.items():
        if metric_name != name:
            continue
        label_dict = dict(metric_labels)
        if all(label_dict.get(k) == v for k, v in labels.items()):
            return value
    return 0.0


def snapshot(base_url, timeout):
    return parse_metrics(get_text(f"{base_url}/metrics", timeout))


def timed_snapshot(base_url, timeout):
    started = time.perf_counter()
    metrics = snapshot(base_url, timeout)
    return metrics, time.perf_counter() - started


def delta(after, before, name, **labels):
    return metric_value(after, name, **labels) - metric_value(before, name, **labels)


def prompt_cache_delta(after, before):
    cached = delta(after, before, "vllm:prompt_tokens_cached_total")
    if cached:
        return cached
    cached = delta(after, before, "vllm:prefix_cache_hits_total")
    if cached:
        return cached
    return delta(after, before, "vllm:gpu_prefix_cache_hits_total")


def computed_prefill_delta(after, before):
    local_compute = delta(
        after,
        before,
        "vllm:prompt_tokens_by_source_total",
        source="local_compute",
    )
    if local_compute:
        return local_compute

    prompt_total = delta(after, before, "vllm:prompt_tokens_total")
    return max(prompt_total - prompt_cache_delta(after, before), 0)


def wait_for_server(base_url, timeout):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"server did not become ready within {timeout}s: {last_error}")


def download_dataset(url, path, timeout):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "vllm-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        path.write_bytes(response.read())


def load_dataset_text(dataset_file, dataset_url, dataset_cache_dir, timeout):
    if dataset_file:
        path = Path(dataset_file)
    else:
        path = Path(dataset_cache_dir) / "wikitext-2-test.txt"
        if not path.exists():
            download_dataset(dataset_url, path, timeout)

    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("="):
            continue
        lines.append(line.replace("<unk>", "unknown"))
    return "\n".join(lines)


def make_dataset_turn_texts(
    base_url, model, dataset_text, input_tokens, turns, timeout
):
    if input_tokens <= 0:
        return [""] * turns

    tokenized = post_json(
        f"{base_url}/tokenize",
        {"model": model, "prompt": dataset_text},
        timeout,
    )
    tokens = tokenized["tokens"]
    needed = input_tokens * turns
    if len(tokens) < needed:
        raise RuntimeError(
            f"dataset has {len(tokens)} tokens, but {needed} are needed "
            f"for {turns} turns x {input_tokens} input tokens"
        )

    turn_texts = []
    for index in range(turns):
        start = index * input_tokens
        chunk_tokens = tokens[start : start + input_tokens]
        detokenized = post_json(
            f"{base_url}/detokenize",
            {"model": model, "tokens": chunk_tokens},
            timeout,
        )
        turn_texts.append(detokenized["prompt"].strip())
    return turn_texts


def common_prefix_len(left, right):
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def hash_previews(tokens, block_size=16):
    previews = []
    for end in range(block_size, len(tokens) + 1, block_size):
        digest = hashlib.sha256(
            json.dumps(tokens[:end], separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        previews.append(
            {
                "end": end,
                "hash": digest[:16],
            }
        )
    return previews


def write_csv(path, rows):
    fieldnames = [
        "turn",
        "latency_s",
        "prompt_tokens",
        "prefill_tokens",
        "cached_prompt_tokens",
        "generated_tokens",
        "generation_tps",
        "usage_completion_tokens",
        "ttft_avg_s_from_metrics",
        "e2e_avg_s_from_metrics",
        "finish_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-turn chat requests and collect per-turn vLLM "
            "/metrics deltas."
        )
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3003")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--input-tokens", type=int, default=1000)
    parser.add_argument("--output-tokens", type=int, default=10)
    parser.add_argument(
        "--min-output-tokens",
        type=int,
        default=None,
        help="Set vLLM min_tokens; use with --output-tokens to force long decodes.",
    )
    parser.add_argument(
        "--dataset-file",
        help="Plain-text dataset file to use for per-turn input chunks.",
    )
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--cache-salt",
        help="Optional vLLM cache_salt to isolate one diagnostic run.",
    )
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument(
        "--dataset-cache-dir",
        default=".cache/datasets",
        help="Directory for downloaded benchmark datasets.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help=(
            "Send each turn as a standalone request instead of preserving "
            "chat history."
        ),
    )
    parser.add_argument(
        "--trace-json",
        action="store_true",
        help="Write per-turn tokenization/common-prefix trace JSON.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    summary_path = out_dir / f"multiturn_metrics_{run_id}.csv"
    trace_path = out_dir / f"multiturn_trace_{run_id}.json"

    wait_for_server(base_url, args.timeout)
    dataset_text = load_dataset_text(
        args.dataset_file,
        args.dataset_url,
        args.dataset_cache_dir,
        args.timeout,
    )
    turn_texts = make_dataset_turn_texts(
        base_url,
        args.model,
        dataset_text,
        args.input_tokens,
        args.turns,
        args.timeout,
    )

    messages = []
    summary_rows = []
    trace_rows = []
    previous_prompt_tokens = None
    previous_sequence_tokens = None
    previous_sequence_prompt_len = 0
    previous_sequence_output_len = 0
    current_metrics, _ = timed_snapshot(base_url, args.timeout)

    for turn in range(1, args.turns + 1):
        if args.no_history:
            messages = []
        messages.append(
            {
                "role": "user",
                "content": turn_texts[turn - 1],
            }
        )
        prompt_token_ids = []
        prior_common_prefix = 0
        prior_sequence_common_prefix = 0
        if args.trace_json:
            tokenized_chat = post_json(
                f"{base_url}/tokenize",
                {
                    "model": args.model,
                    "messages": messages,
                    "add_generation_prompt": True,
                },
                args.timeout,
            )
            prompt_token_ids = tokenized_chat["tokens"]
            prior_common_prefix = (
                common_prefix_len(previous_prompt_tokens, prompt_token_ids)
                if previous_prompt_tokens is not None
                else 0
            )
            prior_sequence_common_prefix = (
                common_prefix_len(previous_sequence_tokens, prompt_token_ids)
                if previous_sequence_tokens is not None
                else 0
            )
            prior_decoded_common_prefix = min(
                max(prior_sequence_common_prefix - previous_sequence_prompt_len, 0),
                previous_sequence_output_len,
            )

        started = time.perf_counter()
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.output_tokens,
            "temperature": args.temperature,
        }
        if args.min_output_tokens is not None:
            payload["min_tokens"] = args.min_output_tokens
        if args.cache_salt:
            payload["cache_salt"] = args.cache_salt
        if args.trace_json:
            payload["return_token_ids"] = True
        response = post_json(
            f"{base_url}/v1/chat/completions",
            payload,
            args.timeout,
        )
        latency_s = time.perf_counter() - started

        next_metrics, _ = timed_snapshot(base_url, args.timeout)

        choice = response["choices"][0]
        assistant_message = choice["message"]
        assistant_content = assistant_message.get("content") or ""
        assistant_token_ids = choice.get("token_ids") or []
        messages.append(
            {
                "role": assistant_message["role"],
                "content": assistant_content,
            }
        )

        request_count_delta = delta(
            next_metrics,
            current_metrics,
            "vllm:e2e_request_latency_seconds_count",
        )
        ttft_delta = delta(
            next_metrics,
            current_metrics,
            "vllm:time_to_first_token_seconds_sum",
        )
        e2e_delta = delta(
            next_metrics,
            current_metrics,
            "vllm:e2e_request_latency_seconds_sum",
        )
        usage = response.get("usage") or {}

        row = {
            "turn": turn,
            "latency_s": round(latency_s, 2),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "prefill_tokens": int(
                computed_prefill_delta(next_metrics, current_metrics)
            ),
            "cached_prompt_tokens": int(
                prompt_cache_delta(next_metrics, current_metrics)
            ),
            "generated_tokens": int(
                delta(next_metrics, current_metrics, "vllm:generation_tokens_total")
            ),
            "generation_tps": 0.0,
            "usage_completion_tokens": int(usage.get("completion_tokens", 0)),
            "ttft_avg_s_from_metrics": round(ttft_delta / request_count_delta, 6)
            if request_count_delta
            else "",
            "e2e_avg_s_from_metrics": round(e2e_delta / request_count_delta, 6)
            if request_count_delta
            else "",
            "finish_reason": choice.get("finish_reason"),
        }
        row["generation_tps"] = (
            round(row["generated_tokens"] / latency_s, 2) if latency_s else 0.0
        )

        summary_rows.append(row)
        if args.trace_json:
            trace_rows.append(
                {
                    "turn": turn,
                    "request_id": response.get("id"),
                    "prompt_tokens_from_tokenize": len(prompt_token_ids),
                    "prompt_tokens_from_usage": row["prompt_tokens"],
                    "prior_common_prefix": prior_common_prefix,
                    "prior_common_prefix_rounded_16": prior_common_prefix
                    - prior_common_prefix % 16,
                    "prior_sequence_common_prefix": prior_sequence_common_prefix,
                    "prior_sequence_common_prefix_rounded_16": (
                        prior_sequence_common_prefix
                        - prior_sequence_common_prefix % 16
                    ),
                    "prior_sequence_prompt_len": previous_sequence_prompt_len,
                    "prior_sequence_output_len": previous_sequence_output_len,
                    "prior_decoded_common_prefix": prior_decoded_common_prefix,
                    "prior_decoded_common_prefix_rounded_16": (
                        prior_decoded_common_prefix
                        - prior_decoded_common_prefix % 16
                    ),
                    "cached_prompt_tokens": row["cached_prompt_tokens"],
                    "prefill_tokens": row["prefill_tokens"],
                    "assistant_role": assistant_message.get("role"),
                    "assistant_content_len": len(assistant_content),
                    "assistant_content_preview": assistant_content[:200],
                    "assistant_token_ids_len": len(assistant_token_ids),
                    "assistant_token_ids_tail": assistant_token_ids[-64:],
                    "assistant_message_keys": sorted(assistant_message.keys()),
                    "prompt_tail_token_ids": prompt_token_ids[-64:],
                    "prompt_hash_previews_tail": hash_previews(prompt_token_ids)[-16:],
                }
            )

        print(
            "turn={turn} latency_s={latency_s} prompt_tokens={prompt_tokens} "
            "prefill_tokens={prefill_tokens} "
            "cached_prompt_tokens={cached_prompt_tokens} "
            "generated_tokens={generated_tokens} "
            "generation_tps={generation_tps}".format(**row),
            flush=True,
        )

        current_metrics = next_metrics
        if args.trace_json:
            previous_prompt_tokens = prompt_token_ids
            previous_sequence_tokens = prompt_token_ids + assistant_token_ids
            previous_sequence_prompt_len = len(prompt_token_ids)
            previous_sequence_output_len = len(assistant_token_ids)

    write_csv(summary_path, summary_rows)
    print(f"summary_csv={summary_path}")
    if args.trace_json:
        trace_path.write_text(json.dumps(trace_rows, indent=2), encoding="utf-8")
        print(f"trace_json={trace_path}")


if __name__ == "__main__":
    main()
