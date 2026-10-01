"""Owned vLLM 0.19.1 server, HTTP generation and GPU/NCCL policy transfer.

No model/vLLM imports at module import time; HTTP/lifecycle contracts can be
tested on CPU. API reference: vLLM's versioned RLHF HTTP NCCL example.
"""

from __future__ import annotations

from concurrent.futures import Future
import threading
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


@dataclass
class Completion:
    text: str
    token_ids: list[int]
    finish_reason: str | None


@dataclass
class PromptOutput:
    # A deliberately small adapter matching the shared evaluation helper.
    outputs: list[Completion]
    prompt_token_ids: list[int] | None = None


def visible_gpu_id(device: str, visible_devices: str | None = None) -> str:
    """Map a trainer's logical cuda index to the child process visibility token."""
    if not device.startswith("cuda:") or not device[5:].isdigit():
        raise ValueError("Server devices must be explicit cuda:N indices")
    index = int(device[5:])
    if visible_devices is None:
        return str(index)
    tokens = [value.strip() for value in visible_devices.split(",") if value.strip()]
    if index >= len(tokens):
        raise ValueError(f"{device} is outside CUDA_VISIBLE_DEVICES={visible_devices!r}")
    return tokens[index]


def decode_choices(choices: list[dict], prompt_count: int, n: int) -> list[PromptOutput]:
    expected = prompt_count * n
    if len(choices) != expected:
        raise ValueError(f"Expected {expected} completions, received {len(choices)}")
    indices = [choice.get("index") for choice in choices]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(expected)):
        raise ValueError("Completion indices must uniquely cover prompt_count * n")
    ordered = sorted(choices, key=lambda choice: choice["index"])
    completions = []
    for choice in ordered:
        ids = choice.get("token_ids")
        if not isinstance(choice.get("text"), str) or not isinstance(ids, list):
            raise ValueError("Server response must include text and token_ids")
        if any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid generated token IDs")
        completions.append(Completion(choice["text"], ids, choice.get("finish_reason")))
    outputs = []
    for i in range(0, expected, n):
        prompt_ids = ordered[i].get("prompt_token_ids")
        if not isinstance(prompt_ids, list) or not prompt_ids or any(type(t) is not int or t < 0 for t in prompt_ids):
            raise ValueError("Response must include valid prompt_token_ids")
        if any(c.get("prompt_token_ids") != prompt_ids for c in ordered[i:i + n]):
            raise ValueError("Response group prompt token IDs disagree")
        outputs.append(PromptOutput(completions[i:i + n], prompt_ids))
    return outputs


class VLLMServerBackend:
    """Single inference GPU, single trainer; manages only its own subprocess."""

    def __init__(self, model_id: str, device: str, seed: int, log_dir: Path,
                 gpu_memory_utilization: float = .85, port: int = 0,
                 startup_timeout: float = 600, request_timeout: float = 300,
                 max_model_len: int = 4096, enforce_eager: bool = False):
        if not 0 < gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if startup_timeout <= 0 or request_timeout <= 0 or max_model_len <= 0:
            raise ValueError("Timeouts and max_model_len must be positive")
        if not 0 <= port <= 65535:
            raise ValueError("Invalid server port")
        self.model_id, self.device, self.seed = model_id, device, seed
        self.log_dir = Path(log_dir)
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port
        self.startup_timeout, self.request_timeout = startup_timeout, request_timeout
        self.max_model_len, self.enforce_eager = max_model_len, enforce_eager
        self.process = None
        self.watchdog = None
        self.weight_sync_group = None
        self._log_file = None
        self._broken = False
        self.weight_version = 0
        self.events: list[dict] = []
        self.base_url = ""

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.close()

    def _request(self, method: str, route: str, payload: dict | None = None,
                 timeout: float | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(self.base_url + route, data=data, method=method,
                                         headers={"Content-Type": "application/json"})
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=timeout or self.request_timeout) as response:
                raw = response.read()
            return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:1500]
            raise RuntimeError(f"vLLM {method} {route}: HTTP {error.code}: {detail}") from error

    def _event(self, kind: str, started: float, **fields) -> None:
        event = {"kind": kind, "seconds": time.perf_counter() - started,
                 "weight_version": self.weight_version, **fields}
        self.events.append(event)
        with (self.log_dir / "backend_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("Server already started")
        # Refuse occupied ports; never kill an unrelated user's process.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", self.port))
            self.port = probe.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpu_id(self.device, env.get("CUDA_VISIBLE_DEVICES"))
        env["VLLM_SERVER_DEV_MODE"] = "1"
        command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", self.model_id,
                   "--host", "127.0.0.1", "--port", str(self.port), "--served-model-name", "policy",
                   "--dtype", "bfloat16", "--enable-prefix-caching", "--seed", str(self.seed),
                   "--gpu-memory-utilization", str(self.gpu_memory_utilization),
                   "--tensor-parallel-size", "1", "--max-model-len", str(self.max_model_len),
                   "--generation-config", "vllm",
                   "--weight-transfer-config", json.dumps({"backend": "nccl"})]
        if self.enforce_eager:
            command.append("--enforce-eager")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = (self.log_dir / "vllm_server.log").open("x")
        started = time.perf_counter()
        try:
            self.process = subprocess.Popen(command, env=env, stdout=self._log_file,
                                            stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"vLLM exited with code {self.process.returncode}; see vllm_server.log")
                try:
                    self._request("GET", "/health", timeout=2)
                    # A separate watchdog removes the owned server if the trainer
                    # is killed while stuck inside a native NCCL call.
                    watchdog_code = """
import os, signal, sys, time
owner, server = map(int, sys.argv[1:])
while os.getppid() == owner:
    time.sleep(1)
try:
    os.killpg(server, signal.SIGTERM)
    time.sleep(10)
    os.killpg(server, signal.SIGKILL)
except ProcessLookupError:
    pass
"""
                    self.watchdog = subprocess.Popen(
                        [sys.executable, "-c", watchdog_code, str(os.getpid()), str(self.process.pid)],
                        stdout=self._log_file, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    self._event("startup", started, port=self.port)
                    return
                except (OSError, RuntimeError):
                    time.sleep(.5)
            raise TimeoutError("vLLM startup timed out; see vllm_server.log")
        except BaseException:
            self.close()
            raise

    def _receiving(self, route: str, payload: dict) -> Future:
        # HTTP runs concurrently with collective initialization/send. A daemon
        # thread cannot delay failure cleanup by waiting in executor.__exit__.
        future = Future()
        def receive():
            try:
                future.set_result(self._request("POST", route, payload))
            except BaseException as error:
                future.set_exception(error)
        threading.Thread(target=receive, daemon=True).start()
        return future

    def init_weight_sync(self, policy_device: str) -> None:
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
        from vllm.utils.network_utils import get_open_port

        started = time.perf_counter()
        if self.weight_sync_group is not None:
            raise RuntimeError("Weight transfer already initialized")
        if self._request("GET", "/get_world_size")["world_size"] != 1:
            raise ValueError("This backend supports exactly one inference worker")
        torch.cuda.set_device(torch.device(policy_device))
        info = {"master_address": "127.0.0.1", "master_port": get_open_port(), "world_size": 2}
        try:
            receiving = self._receiving("/init_weight_transfer_engine", {"init_info": {**info, "rank_offset": 1}})
            self.weight_sync_group = NCCLWeightTransferEngine.trainer_init(info)
            receiving.result(timeout=self.request_timeout)
            self._event("init_weight_sync", started)
        except BaseException:
            self._broken = True
            self.close()
            raise

    def sync_policy_weights(self, policy) -> None:
        import torch
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine,
        )

        if self._broken or self.weight_sync_group is None:
            raise RuntimeError("A healthy initialized weight-transfer group is required")
        weights = list(policy.named_parameters())
        if not weights or weights[0][1].device.type != "cuda" or any(p.device != weights[0][1].device for _, p in weights):
            raise ValueError("All policy parameters must be on one training GPU")
        torch.cuda.set_device(weights[0][1].device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        info = {"names": [name for name, _ in weights],
                "dtype_names": [str(p.dtype).split(".")[-1] for _, p in weights],
                "shapes": [list(p.shape) for _, p in weights], "packed": True}
        try:
            self._request("POST", "/pause")
            receiving = self._receiving("/update_weights", {"update_info": info})
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=iter(weights),
                trainer_args=NCCLTrainerSendWeightsArgs(group=self.weight_sync_group, packed=True),
            )
            torch.cuda.synchronize()
            receiving.result(timeout=self.request_timeout)
            self._request("POST", "/reset_prefix_cache")
            self._request("POST", "/resume")
            self.weight_version += 1
            self._event("weight_sync", started,
                        bytes=sum(p.numel() * p.element_size() for _, p in weights),
                        cache_reset=True, transport="nccl")
        except BaseException:
            # Never resume serving after a partial or uncertain weight update.
            self._broken = True
            self.close()
            raise

    def generate(self, prompts: list[str], sampling_params) -> list[PromptOutput]:
        if self._broken or self.process is None or self.process.poll() is not None:
            raise RuntimeError("Generation requires a healthy running server")
        if not prompts:
            return []
        def get(name, default=None):
            return sampling_params.get(name, default) if isinstance(sampling_params, dict) else getattr(sampling_params, name, default)
        n = get("n", 1)
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        payload = {"model": "policy", "prompt": prompts, "n": n,
                   "temperature": get("temperature", 1.), "max_tokens": get("max_tokens"),
                   "seed": get("seed"), "return_token_ids": True,
                   "add_special_tokens": True, "min_tokens": get("min_tokens", 0),
                   "top_p": get("top_p", 1.), "top_k": get("top_k", -1),
                   "repetition_penalty": get("repetition_penalty", 1.),
                   "ignore_eos": get("ignore_eos", False)}
        if get("stop"):
            payload.update(stop=get("stop"), include_stop_str_in_output=get("include_stop_str_in_output", False))
        started = time.perf_counter()
        response = self._request("POST", "/v1/completions", payload)
        outputs = decode_choices(response["choices"], len(prompts), n)
        tokens = sum(len(c.token_ids) for output in outputs for c in output.outputs)
        self._event("generation", started, prompts=len(prompts), completions=len(prompts) * n,
                    generated_tokens=tokens,
                    truncated=sum(c.finish_reason == "length" for o in outputs for c in o.outputs))
        return outputs

    def close(self) -> None:
        process = self.process
        self.process = None
        # PyNcclCommunicator is stateless, not torch's default process group.
        # vLLM 0.19.1's own shutdown drops its communicator reference.
        self.weight_sync_group = None
        started = time.perf_counter()
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)
        if process is not None:
            (self.log_dir / "server_shutdown.json").write_text(json.dumps({
                "pid": process.pid, "returncode": process.returncode,
                "stopped": process.poll() is not None, "seconds": time.perf_counter() - started,
            }, indent=2) + "\n")
        if self.watchdog is not None:
            self.watchdog.terminate()
            try:
                self.watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.watchdog.kill()
                self.watchdog.wait(timeout=5)
            self.watchdog = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
