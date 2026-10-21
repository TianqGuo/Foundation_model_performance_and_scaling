"""Synthetic SSE/benchmark tests; no network calls, checkpoints or GPU execution."""
from importlib import import_module
import json
import unittest
from unittest.mock import patch
module = import_module("infrastructure.7d_serve.benchmark")


def stream_events(items):
    """Encode synthetic server dictionaries/DONE as byte-line SSE frames."""
    for item in items:
        yield ("data: " + (item if isinstance(item,str) else json.dumps(item)) + "\n").encode()
        yield b"\n"


class BenchmarkTests(unittest.TestCase):
    def success(self, *args):
        """Return one synthetic request result for aggregation tests."""
        return dict(success=True,latency_seconds=.5,ttft_seconds=.1,
                    output_token_time_estimate_seconds=.02,completion_tokens=5)

    def test_usage_not_text_chunks_counts_tokens(self):
        """A single text event may contain several tokens and empty events are ignored."""
        events=[dict(choices=[dict(text="",finish_reason=None)]),
                dict(choices=[dict(text="several tokens",finish_reason=None)]),
                dict(choices=[dict(text="more",finish_reason="stop")]),
                dict(choices=[],usage=dict(prompt_tokens=10,completion_tokens=7)),"[DONE]"]
        with patch.object(module.time,"perf_counter",return_value=1):
            row=module.read_stream(stream_events(events),0,clock=lambda:1)
        self.assertEqual(row["completion_tokens"],7)
        self.assertEqual(row["text_events"],2)
        self.assertEqual(row["ttft_seconds"],1)

    def test_truncated_error_and_missing_usage_fail(self):
        """Do not classify invalid streams as successful requests."""
        for events in ([dict(error=dict(message="failure"))],
                       [dict(choices=[dict(text="x",finish_reason="stop")]),"[DONE]"],
                       [dict(usage=dict(prompt_tokens=1,completion_tokens=2))],
                       [dict(choices=[dict(text="x",finish_reason="stop")]),
                        dict(usage=dict(prompt_tokens=1,completion_tokens=True)),"[DONE]"]):
            with self.assertRaises(ValueError):
                module.read_stream(stream_events(events),0,clock=lambda:1)

    def test_zero_tokens_and_timeouts(self):
        """Zero-token responses have no TTFT; wall timeout rejects an active stream."""
        events=[dict(choices=[dict(text="",finish_reason="stop")],
                     usage=dict(prompt_tokens=1,completion_tokens=0)),"[DONE]"]
        row=module.read_stream(stream_events(events),0,clock=lambda:1)
        self.assertIsNone(row["ttft_seconds"])
        self.assertIsNone(row["output_token_time_estimate_seconds"])
        with self.assertRaises(TimeoutError):
            module.read_stream(stream_events(events),0,clock=lambda:121)

    def test_aggregation_counts_failures_and_wall_time(self):
        """Failures reduce request throughput and are retained in the error rate."""
        report=module.summarize([self.success(),dict(success=False)],2)
        self.assertEqual(report["error_rate"],.5)
        self.assertEqual(report["successful_requests_per_second"],.5)
        self.assertEqual(report["generated_tokens_per_second"],2.5)
        empty=module.summarize([dict(success=False)],1)
        self.assertIsNone(empty["ttft_seconds"]["p95"])
        self.assertEqual(module.percentile([1,3],.95),2.9)

    def test_warmup_excluded_from_measurements(self):
        """Only measured requests enter reports; prompt text is replaced by a hash."""
        calls=[]
        def requester(*args):
            """Record synthetic calls and return a completed request."""
            calls.append(args)
            return self.success()
        report=module.run("http://127.0.0.1:8000","model",["raw prompt"],
                          requests=4,concurrency=2,warmup=2,requester=requester)
        self.assertEqual(len(calls),6)
        self.assertEqual(report["summary"]["requests"],4)
        self.assertNotIn("raw prompt",json.dumps(report))
        self.assertEqual(len(report["requests"][0]["prompt_sha256"]),64)

    def test_invalid_settings_and_warmup_failure(self):
        """Reject unsafe endpoints/unbounded settings before scheduling requests."""
        for endpoint in ("https://127.0.0.1", "http://example.com", "http://user@localhost", "http://localhost/path"):
            with self.assertRaises(ValueError):
                module.run(endpoint,"model",["prompt"],requester=self.success)
        with self.assertRaises(ValueError):
            module.run("http://localhost","model",["prompt"],requests=129,requester=self.success)
        with self.assertRaises(RuntimeError):
            module.run("http://localhost","model",["prompt"],requester=lambda *args:dict(success=False))

    def test_request_parameters_and_sanitized_errors(self):
        """Send raw prompt/stop semantics and retain no response bodies on HTTP failure."""
        with patch.object(module,"urlopen",side_effect=ValueError("private response")) as opener:
            row=module.request_completion("http://localhost:8000","model","raw",32,10)
        payload=json.loads(opener.call_args.args[0].data)
        self.assertEqual(payload["prompt"],"raw")
        self.assertEqual(payload["stop"],["</answer>"])
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertTrue(payload["include_stop_str_in_output"])
        self.assertFalse(row["success"])
        self.assertNotIn("private response",json.dumps(row))
