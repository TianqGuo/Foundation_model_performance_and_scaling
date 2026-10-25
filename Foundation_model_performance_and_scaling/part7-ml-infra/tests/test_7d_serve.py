"""CPU contract/integrity tests with synthetic files; never load a real model."""
from dataclasses import replace
from importlib import import_module
import json
import os
from pathlib import Path
import tempfile
import unittest
config = import_module("infrastructure.7d_serve.config")
render = import_module("infrastructure.7d_serve.render")
model = import_module("infrastructure.7d_serve.prepare_model")
IMAGE = "registry.example/serving@sha256:" + "a"*64


class ServingPreparationTests(unittest.TestCase):
    def snapshot(self, root):
        """Write fake export assets for hash/layout tests, not actual tensor data."""
        for name,text in {"config.json":"{}","tokenizer_config.json":"{}", "tokenizer.json":"{}",
                          "model.safetensors":"synthetic non-model bytes"}.items():
            (root/name).write_text(text)

    def test_pins_and_configurability(self):
        """Reject mutable identity/unsupported settings and separate model snapshot paths."""
        base=config.ServingConfig()
        self.assertNotEqual(base.snapshot_path(),replace(base,model_revision="a"*40).snapshot_path())
        for update in (dict(model_revision="main"),dict(model_repo="../bad"),dict(model_id="../bad"),
                       dict(max_model_len=0),dict(max_num_seqs=0),dict(gpu_memory_utilization=1),
                       dict(tensor_parallel_size=2)):
            with self.assertRaises(ValueError):replace(base,**update).validate()
        native=base.llm_config()
        self.assertEqual(native["deployment_config"]["num_replicas"],1)
        self.assertFalse(native["engine_kwargs"]["enable_prefix_caching"])

    def test_manifest_security_and_single_gpu_contract(self):
        """Keep head CPU-only, worker one whole GPU, and credentials out of JSON."""
        doc=render.render_service(IMAGE,"part7-gpu")
        self.assertEqual(doc["kind"],"RayService")
        cluster=doc["spec"]["rayClusterConfig"]
        head=cluster["headGroupSpec"]["template"]["spec"]
        worker=cluster["workerGroupSpecs"][0]["template"]["spec"]
        self.assertNotIn("nvidia.com/gpu",head["containers"][0]["resources"]["limits"])
        self.assertEqual(worker["containers"][0]["resources"]["limits"]["nvidia.com/gpu"],1)
        for pod in (head,worker):
            self.assertEqual(pod["securityContext"]["runAsUser"],1000)
            self.assertFalse(pod["automountServiceAccountToken"])
            init=pod["initContainers"][0]
            self.assertIn("--cloud",init["args"])
            self.assertIn(config.DEFAULT_REVISION,init["args"])
            self.assertIn(dict(name="NVIDIA_VISIBLE_DEVICES",value="void"),init["env"])
            self.assertFalse(any("token" in arg.lower() for arg in init["args"]))
            mount=next(m for m in pod["containers"][0]["volumeMounts"] if m["name"]=="workspace")
            self.assertTrue(mount["readOnly"])
        probe=worker["containers"][0]["readinessProbe"]
        command=probe["exec"]["command"][-1]
        self.assertIn("local_raylet_healthz",command)
        self.assertNotIn("8000",command)
        self.assertGreater(probe["timeoutSeconds"],2)
        native=json.loads(doc["spec"]["serveConfigV2"])
        self.assertEqual(native["proxy_location"],"HeadOnly")
        self.assertEqual(native["applications"][0]["import_path"],"ray.serve.llm:build_openai_app")

    def test_bad_image_and_namespace_rejected(self):
        """Generated service must retain the tested 7C identifier/digest validation."""
        with self.assertRaises(ValueError):render.render_service("image:latest","node")
        with self.assertRaises(ValueError):render.render_service(IMAGE,"node",namespace="../bad")

    def test_snapshot_reuse_requires_exact_identity_and_hashes(self):
        """Verify actual hash reads and reject tampered synthetic exports."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.snapshot(root)
            settings=config.ServingConfig()
            receipt=dict(repo_id=settings.model_repo,revision=settings.model_revision,files=model.inventory(root))
            (root/model.RECEIPT).write_text(json.dumps(receipt))
            self.assertEqual(model.verify_saved(root,settings),receipt)
            (root/"tokenizer_config.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,"inventory changed"):model.verify_saved(root,settings)

    def test_index_requires_all_shards(self):
        """A present safetensors index cannot hide a missing shard or path escape."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.snapshot(root)
            index=root/"model.safetensors.index.json"
            for name in ("missing.safetensors","../outside.safetensors"):
                index.write_text(json.dumps(dict(weight_map={"weight":name})))
                with self.assertRaises(ValueError):model.inventory(root)

    def test_snapshot_symlinks_rejected(self):
        """Prevent inventory traversal through linked model/tokenizer files."""
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);self.snapshot(root)
            (root/"link.json").symlink_to(root/"config.json")
            with self.assertRaises(ValueError):model.inventory(root)

    def test_cloud_only_staging_guard(self):
        """Local paths must fail before any Hub credentials or download are requested."""
        with self.assertRaisesRegex(ValueError,"persistent workspace"):
            model.prepare(config.ServingConfig(),Path("/tmp/not-a-real-model"))

    @unittest.skipUnless(os.environ.get("KUBERAY_SERVICE_CRD"),"set KUBERAY_SERVICE_CRD for pinned schema validation")
    def test_pinned_rayservice_schema(self):
        """Validate rendered service against actual KubeRay 1.4.2 CRD without API calls."""
        import yaml
        import jsonschema
        crd=yaml.safe_load(Path(os.environ["KUBERAY_SERVICE_CRD"]).read_text())
        schema=next(v for v in crd["spec"]["versions"] if v["name"]=="v1")["schema"]["openAPIV3Schema"]
        errors=list(jsonschema.Draft7Validator(schema).iter_errors(render.render_service(IMAGE,"part7-gpu")))
        self.assertFalse(errors,[(list(e.path),e.message) for e in errors])


class ServingRuntimeGateTests(unittest.TestCase):
    def test_signature_gate_rejects_removed_argument(self):
        """A resolved package set must fail when the adapter passes a removed keyword."""
        gate=import_module("infrastructure.7d_serve.runtime_check")
        def constructor(vllm_config, *, log_stats=False):
            """Synthetic signature only; never instantiate an engine."""
            raise AssertionError("Must not execute constructor")
        good="engine = AsyncLLM(vllm_config=config, log_stats=True)"
        self.assertEqual(gate.validate_constructor_calls(good,constructor),1)
        with self.assertRaises(TypeError):
            gate.validate_constructor_calls("AsyncLLM(vllm_config=config, executor_class=executor)",constructor)

    def test_signature_gate_fails_closed_on_unreviewed_source(self):
        """Missing or dynamic constructor calls cannot accidentally pass the audit."""
        gate=import_module("infrastructure.7d_serve.runtime_check")
        for source in ("other_call()", "AsyncLLM(**kwargs)", "AsyncLLM(*args)"):
            with self.assertRaises(ValueError):gate.validate_constructor_calls(source,object)


class ModernServingInterfaceTests(unittest.TestCase):
    def test_reviewed_engine_alias_without_constructing_engine(self):
        """Support the exact Ray 2.59 alias while rejecting unknown/dynamic calls."""
        gate=import_module("infrastructure.7d_serve.runtime_check")
        def constructor(vllm_config,executor_class,log_stats,log_requests,stat_loggers=None):
            """Signature fixture only; execution would invalidate this test."""
            raise AssertionError("Do not create inference engines locally")
        source="engine_cls = AsyncLLM\nengine = engine_cls(vllm_config=config,executor_class=executor,log_stats=True,log_requests=False)"
        self.assertEqual(gate.validate_constructor_calls(source,constructor),1)
        for source in ("engine_cls = OtherEngine\nengine_cls()", "engine_cls = AsyncLLM\nengine_cls(**kwargs)"):
            with self.assertRaises(ValueError):gate.validate_constructor_calls(source,constructor)

    def test_runtime_rejects_accidental_vllm_version_drift(self):
        """CUDA local suffixes are allowed; unreviewed core package versions are not."""
        gate=import_module("infrastructure.7d_serve.runtime_check")
        installed=dict(config.CORE_VERSIONS)
        installed["torch"] += "+cu130"
        gate.validate_core_versions(installed)
        installed["vllm"]="0.13.0"
        with self.assertRaisesRegex(ValueError,"vllm"):gate.validate_core_versions(installed)

    def test_protocol_round_trip_allows_added_defaults(self):
        """Additional nested protocol defaults are valid; dropped request semantics are not."""
        gate=import_module("infrastructure.7d_serve.runtime_check")
        expected=dict(prompt="raw",stream_options=dict(include_usage=True))
        actual=dict(prompt="raw",stream_options=dict(include_usage=True,continuous_usage_stats=False))
        gate.require_payload_fields(expected,actual)
        actual["stream_options"]["include_usage"]=False
        with self.assertRaisesRegex(ValueError,"stream_options.include_usage"):
            gate.require_payload_fields(expected,actual)
