"""Bounded cloud HTTP benchmark for raw OpenAI completions; no model imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def percentile(values, fraction):
    """Return an interpolated quantile of finite observations, or None for no data.

    Inputs: numeric sequence and fraction in [0,1]. Sorts a copy, without I/O.
    Invalid fractions or nonfinite observations raise ValueError.
    """
    if not 0 <= fraction <= 1 or any(not math.isfinite(value) for value in values):
        raise ValueError("Invalid percentile input")
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered)-1)*fraction
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def read_stream(lines, started, *, clock=time.perf_counter, timeout=120):
    """Parse SSE completion events and return client timings plus server token usage.

    Inputs: byte-line iterator, request start clock and injectable monotonic clock.
    Records first/last nonempty text event, never counts SSE chunks as tokens.
    Requires terminal DONE, finish reason and integer usage counts; malformed,
    incomplete or error streams raise ValueError. Does not save generated text.
    """
    first = last = None
    chunks = 0
    usage = None
    finished = done = False
    event = []
    for raw in lines:
        if clock()-started > timeout:
            raise TimeoutError("Streaming request exceeded wall limit")
        if len(raw) > 65536:
            raise ValueError("Oversized SSE line")
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            event.append(line[5:].lstrip())
            continue
        if line or not event:
            continue
        data = "\n".join(event)
        event.clear()
        if data == "[DONE]":
            done = True
            break
        item = json.loads(data)
        if item.get("error"):
            raise ValueError("Server returned a streaming error")
        if item.get("usage") is not None:
            usage = item["usage"]
        for choice in item.get("choices", []):
            if choice.get("index", 0) != 0:
                raise ValueError("Benchmark supports one completion per request")
            if choice.get("text"):
                last = clock()
                first = last if first is None else first
                chunks += 1
            if choice.get("finish_reason") is not None:
                finished = True
    elapsed = clock()-started
    if not done or not finished or not isinstance(usage, dict):
        raise ValueError("Incomplete stream or missing server usage")
    for key in ("prompt_tokens", "completion_tokens"):
        if type(usage.get(key)) is not int or usage[key] < 0:
            raise ValueError("Missing/invalid server token count")
    tokens = usage["completion_tokens"]
    if tokens and first is None:
        raise ValueError("Server reported tokens without streamed text")
    return dict(success=True, latency_seconds=elapsed,
                ttft_seconds=None if first is None else first-started,
                text_event_span_seconds=None if first is None else last-first,
                text_events=chunks, prompt_tokens=usage["prompt_tokens"],
                completion_tokens=tokens,
                output_token_time_estimate_seconds=(last-first)/(tokens-1)
                    if first is not None and tokens > 1 else None)


def request_completion(base_url, model_id, prompt, max_tokens, timeout):
    """Send one raw streaming completion and return metrics or a sanitized failure.

    Inputs: loopback endpoint, model ID, rendered prompt, generation cap and timeout.
    Uses include_usage and n=1; no chat template or retries. Network/parse failures
    become failed rows without response bodies, prompt text or credentials.
    """
    started = time.perf_counter()
    payload = dict(model=model_id, prompt=prompt, max_tokens=max_tokens,
                   temperature=0, seed=12345, stop=["</answer>"], include_stop_str_in_output=True,
                   n=1, stream=True, stream_options=dict(include_usage=True))
    request = Request(base_url.rstrip("/")+"/v1/completions",
                      data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            if "text/event-stream" not in response.headers.get("Content-Type", ""):
                raise ValueError("Expected SSE response")
            return read_stream(iter(lambda: response.readline(65537), b""), started, timeout=timeout)
    except Exception as error:
        return dict(success=False, error_type=type(error).__name__,
                    latency_seconds=time.perf_counter()-started)


def summarize(rows, elapsed):
    """Aggregate completed measurement rows over the full measurement wall interval.

    Inputs: per-request metric dictionaries and positive wall seconds. Returns
    request/token throughput, error counts and latency quantiles; failed requests
    stay in the denominator. Empty successful samples produce None quantiles.
    """
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("Positive finite elapsed time required")
    successful = [row for row in rows if row["success"]]
    result = dict(requests=len(rows), successful_requests=len(successful),
                  errors=len(rows)-len(successful), wall_seconds=elapsed,
                  successful_requests_per_second=len(successful)/elapsed,
                  generated_tokens=sum(row["completion_tokens"] for row in successful),
                  error_rate=(len(rows)-len(successful))/len(rows) if rows else None)
    result["generated_tokens_per_second"] = result["generated_tokens"]/elapsed
    for field in ("latency_seconds", "ttft_seconds", "output_token_time_estimate_seconds"):
        values = [row[field] for row in successful if row[field] is not None]
        result[field] = dict(samples=len(values), p50=percentile(values,.5), p95=percentile(values,.95))
    return result


def run(base_url, model_id, prompts, *, requests=16, concurrency=1, warmup=2,
        max_tokens=128, timeout=120, requester=request_completion):
    """Run bounded warmup then concurrent requests and return a JSON-safe report.

    Prompts are already rendered raw strings. Warmup failures abort measurement;
    measured failures remain in the report. Requester injection supports CPU unit
    tests without any model/network activity. No output files or cleanup side effects.
    """
    address = urlsplit(base_url)
    if address.scheme != "http" or address.hostname not in ("localhost", "127.0.0.1") or address.username or address.password or address.path not in ("", "/") or address.query or address.fragment:
        raise ValueError("Use the local HTTP port-forward endpoint")
    if not prompts or any(not isinstance(prompt,str) or not prompt.strip() for prompt in prompts):
        raise ValueError("Nonempty rendered prompts required")
    for value, low, high in ((requests,1,128),(concurrency,1,16),(warmup,0,8),(max_tokens,1,512),(timeout,1,180)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Benchmark setting outside bounded range")
    if concurrency > requests:
        raise ValueError("Concurrency cannot exceed measured requests")
    for index in range(warmup):
        if not requester(base_url, model_id, prompts[index%len(prompts)], max_tokens, timeout)["success"]:
            raise RuntimeError("Warmup failed; serving is not ready for measurement")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(requester, base_url, model_id, prompts[index%len(prompts)], max_tokens, timeout)
                   for index in range(requests)]
        rows = [dict(request_index=index, prompt_sha256=hashlib.sha256(prompts[index%len(prompts)].encode()).hexdigest(),
                     **future.result()) for index,future in enumerate(futures)]
    elapsed = time.perf_counter()-started
    return dict(settings=dict(model_id=model_id, requests=requests, concurrency=concurrency,
                              warmup=warmup, max_tokens=max_tokens, timeout_seconds=timeout,
                              temperature=0, seed=12345, stop=["</answer>"], include_stop_str_in_output=True,
                              stream=True, client="cloud VM loopback port-forward"),
                summary=summarize(rows,elapsed), requests=rows,
                limitations=["Client TTFT includes HTTP/queue/prefill and transport latency",
                             "Text-event span/token estimate is not engine inter-token latency",
                             "SSE text events may contain multiple tokens",
                             "Not a model-quality or multi-node scalability test"])


def main():
    """Run only on an explicitly selected cloud GPU host and save a fresh report.

    JSONL input rows contain a raw `prompt` string, prepared by the future E2E
    runner. Refuses overwrite before HTTP requests. No tokenizer/model is loaded.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cloud", action="store_true", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="part7-math-grpo")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name,default in (("requests",16),("concurrency",1),("warmup",2),("max-tokens",128),("timeout",120)):
        parser.add_argument("--"+name, type=int, default=default)
    args = parser.parse_args()
    if not shutil.which("nvidia-smi"):
        raise ValueError("Cloud GPU host required")
    subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, timeout=15)
    if args.output.exists():
        raise FileExistsError("Use a fresh benchmark evidence path")
    prompts = [json.loads(line)["prompt"] for line in args.prompts.read_text().splitlines() if line.strip()]
    report = run(args.base_url,args.model_id,prompts,requests=args.requests,concurrency=args.concurrency,
                 warmup=args.warmup,max_tokens=args.max_tokens,timeout=args.timeout)
    report["prompt_file_sha256"] = hashlib.sha256(args.prompts.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2)
        stream.write("\n")
    if report["summary"]["errors"]:
        raise SystemExit("Benchmark retained request errors; inspect report")


if __name__ == "__main__":
    main()
