"""CPU prerequisite tests with fabricated Kubernetes objects; no cluster calls."""
from copy import deepcopy
from decimal import Decimal
from importlib import import_module
import json
import unittest
from unittest.mock import patch
check=import_module("infrastructure.7c_kuberay.cluster_check")


class ClusterCheckTests(unittest.TestCase):
    def objects(self):
        """Return synthetic ready node, shared PVC, operator and established CRD."""
        node=dict(metadata=dict(name="gpu-node",labels={"kubernetes.io/hostname":"gpu-node"}),spec={},
                  status=dict(conditions=[dict(type="Ready",status="True")],allocatable={"cpu":"32000m","memory":"128Gi","nvidia.com/gpu":"2"}))
        pvc=dict(metadata=dict(name="workspace"),spec=dict(accessModes=["ReadWriteOnce"]),status=dict(phase="Bound",capacity=dict(storage="200Gi")))
        operator=dict(metadata=dict(generation=1),spec=dict(template=dict(spec=dict(containers=[dict(image="kuberay/operator:v1.4.2")]))),status=dict(observedGeneration=1,availableReplicas=1))
        crd=dict(spec=dict(versions=[dict(name="v1",served=True)]),status=dict(conditions=[dict(type="Established",status="True")]))
        return node,pvc,operator,crd

    def test_quantity_units(self):
        """Interpret CPU millicores and binary/decimal byte quantities accurately."""
        self.assertEqual(check.quantity("32000m"),Decimal(32))
        self.assertEqual(check.quantity("128Gi"),128*1024**3)
        self.assertEqual(check.quantity("2G"),2000000000)
        with self.assertRaises(ValueError):check.quantity("unknown")

    def test_shared_storage_ready_capacity(self):
        """Accept same-node RWO/RWX storage without claiming free resources."""
        values=self.objects();report=check.validate(*values)
        self.assertEqual(report["rayjob_api"],"ray.io/v1")
        self.assertIn("not free",report["scope"])
        values[1]["spec"]["accessModes"]=["ReadWriteMany"]
        check.validate(*values)

    def test_rejects_single_pod_or_unbound_storage(self):
        """Prevent using storage that cannot be shared by the head and worker."""
        for mode in ("ReadWriteOncePod","ReadOnlyMany"):
            values=self.objects();values[1]["spec"]["accessModes"]=[mode]
            with self.assertRaisesRegex(ValueError,"sharing"):check.validate(*values)
        values=self.objects();values[1]["status"]["phase"]="Pending"
        with self.assertRaisesRegex(ValueError,"Bound"):check.validate(*values)

    def test_node_capacity_readiness_and_taints(self):
        """Reject insufficient or unschedulable nodes before staging or GPU jobs."""
        for resource,value in (("cpu","24"),("memory","80Gi"),("nvidia.com/gpu","1")):
            values=self.objects();values[0]["status"]["allocatable"][resource]=value
            with self.assertRaisesRegex(ValueError,"capacity"):check.validate(*values)
        values=self.objects();values[0]["spec"]["taints"]=[dict(effect="NoSchedule",key="gpu")]
        with self.assertRaisesRegex(ValueError,"tolerations"):check.validate(*values)
        values=self.objects();values[0]["spec"]["unschedulable"]=True
        with self.assertRaisesRegex(ValueError,"schedulable"):check.validate(*values)

    def test_operator_pin_and_api_establishment(self):
        """Reject unavailable/wrong operators and an unserved/unestablished API."""
        values=self.objects();values[2]["status"]["availableReplicas"]=0
        with self.assertRaisesRegex(ValueError,"available"):check.validate(*values)
        values=self.objects();values[2]["spec"]["template"]["spec"]["containers"][0]["image"]="operator:latest"
        with self.assertRaisesRegex(ValueError,"identity"):check.validate(*values)
        values=self.objects();values[3]["spec"]["versions"][0]["served"]=False
        with self.assertRaisesRegex(ValueError,"served"):check.validate(*values)

    def test_inspection_is_get_only_without_secrets(self):
        """Verify explicit context and read-only metadata queries, never Secret access."""
        with patch.object(check.control,"kubectl",side_effect=[json.dumps(x) for x in self.objects()]) as api:
            report=check.inspect("explicit","part7","gpu-node","workspace","default","operator")
        self.assertEqual(report["node"],"gpu-node")
        self.assertTrue(all(call.args[0]=="explicit" and call.args[2][0]=="get" for call in api.call_args_list))
        self.assertFalse(any("secret" in call.args[2] for call in api.call_args_list))
