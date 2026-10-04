"""Cloud verl adapter preserving R1 stop behavior; generation stays upstream.

Register via rollout.agent.agent_loop_config_path with a Hydra list entry whose
name is r1_single_turn and _target_ is workloads.r1_agent.R1SingleTurnAgentLoop.
Set rollout.agent.default_agent_loop=r1_single_turn. CPU utilities remain in
math_workload.py so preparation does not import verl or any model libraries.
"""

from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

from workloads.math_workload import r1_sampling_params


class R1SingleTurnAgentLoop(SingleTurnAgentLoop):
    async def run(self, sampling_params, **kwargs):
        params = r1_sampling_params(sampling_params, kwargs["extra_info"])
        return await super().run(params, **kwargs)
