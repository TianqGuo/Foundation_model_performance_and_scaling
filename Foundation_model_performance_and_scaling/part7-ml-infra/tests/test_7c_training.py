"""CPU config/recovery checks; no Ray cluster, Torch import or checkpoint loading."""
from copy import deepcopy
from importlib import import_module
import unittest
runner = import_module("infrastructure.7c_kuberay.run_7c")
render = import_module("infrastructure.7c_kuberay.render")


class ExistingClusterTests(unittest.TestCase):
    def config(self):
        """Return minimal synthetic topology and local-Ray fields to be removed."""
        return dict(trainer=dict(nnodes=1, n_gpus_per_node=2),
            actor_rollout_ref=dict(actor=dict(strategy="fsdp2", fsdp_config=dict(fsdp_size=2)), model={}),
            ray_kwargs=dict(ray_init=dict(address="local", _temp_dir="/tmp/local", num_cpus=8,
                                         runtime_env=dict(env_vars=dict(PYTHONPATH="/opt/part7")))))

    def test_existing_cluster_and_interruption(self):
        """Reject local runtime options and enable bounded interruption hooks."""
        c=runner.configure_phase(self.config(), "interrupt", "/workspace/runs/stop")
        self.assertEqual(set(c["ray_kwargs"]["ray_init"]), {"address", "runtime_env"})
        self.assertEqual(c["ray_kwargs"]["ray_init"]["address"], "auto")
        self.assertEqual(c["trainer"]["save_freq"], 2)
        self.assertFalse(c["trainer"]["val_before_train"])
        self.assertEqual(c["ray_kwargs"]["ray_init"]["runtime_env"]["env_vars"]["PART7_INTERRUPT_STEP"], "2")

    def test_native_resume_and_bounds(self):
        """Require matching two-rank FSDP2 and more steps than the saved checkpoint."""
        manifest=dict(step=2, world_size=2)
        c=runner.configure_phase(self.config(), "resume", "/workspace/runs/resume", checkpoint="/workspace/saved/global_step_2", manifest=manifest)
        self.assertEqual(c["trainer"]["resume_mode"], "resume_path")
        for bad in (dict(step=3,world_size=2), dict(step=2,world_size=1)):
            with self.assertRaises(ValueError):
                runner.configure_phase(self.config(), "resume", "/workspace/resume", checkpoint="/workspace/saved", manifest=bad)
        with self.assertRaises(ValueError):
            runner.configure_phase(self.config(), "interrupt", "/workspace/stop", total_steps=2)

    def test_identity_changes_rejected_but_pods_can_change(self):
        """Allow a fresh pod only when source/model/data/environment remain identical."""
        identity=dict(verl_commit="pin", python="same",packages={},source_hashes={},model_files={"w":"hash"},
                      platform=dict(milestone="7c",pod="old"), data_manifest=dict(outputs={"train":dict(sha256="data")}))
        saved=dict(identity=deepcopy(identity))
        identity["platform"]["pod"]="new"
        runner.validate_identity(identity,saved)
        for key in ("model_files","source_hashes","packages","python"):
            changed=deepcopy(identity); changed[key]="different"
            with self.assertRaises(ValueError): runner.validate_identity(changed,saved)
        changed=deepcopy(identity);changed["data_manifest"]["outputs"]["train"]["sha256"]="changed"
        with self.assertRaises(ValueError): runner.validate_identity(changed,saved)

    def test_persistent_paths_required(self):
        """Reject ephemeral/head-only paths before any cloud model preparation."""
        runner.validate_shared_paths(["/workspace/runs/job", "/workspace/data/bundle"])
        with self.assertRaises(ValueError): runner.validate_shared_paths(["/tmp/checkpoint"])
        with self.assertRaises(ValueError): runner.validate_shared_paths(["/workspace/../tmp/checkpoint"])

    def test_resume_manifest_and_shell_quoting(self):
        """Require a PVC checkpoint and quote shell syntax as a literal argument."""
        args=("registry/image@sha256:"+"a"*64,"part7","resume","pvc","secret","node")
        with self.assertRaises(ValueError):render.render_job(*args,phase="resume")
        with self.assertRaises(ValueError):render.render_job(*args,phase="resume",checkpoint="/workspace/../etc/passwd")
        m=render.render_job(*args,phase="resume",checkpoint="/workspace/saved/a;echo injected")
        self.assertIn("'/workspace/saved/a;echo injected'",m["spec"]["entrypoint"])
        self.assertEqual(m["spec"]["activeDeadlineSeconds"],2700)
