"""
ISAA Jobs - Persistent Scheduled Agent Tasks
=============================================

Usage:
    from toolboxv2.mods.isaa.extras.jobs import (
        JobDefinition, TriggerConfig, TriggerType,
        JobScheduler, TriggerRegistry, TriggerEvaluator,
    )

Custom trigger example:
    class MyTrigger:
        async def setup(self, job, scheduler): ...
        async def evaluate(self, job) -> bool: ...
        async def teardown(self, job): ...

    scheduler.trigger_registry.register("my_trigger", MyTrigger())
"""

from .job_manager import (
    JobDefinition,
    JobEventBus,
    JobScheduler,
    TriggerConfig,
    TriggerEvaluator,
    TriggerRegistry,
    TriggerType,
    OnDreamEventEvaluator,
    OnAgentIdleEvaluator
)

__all__ = [
    "JobDefinition",
    "JobEventBus",
    "JobScheduler",
    "TriggerConfig",
    "TriggerEvaluator",
    "TriggerRegistry",
    "TriggerType",
    "OnDreamEventEvaluator",
    "OnAgentIdleEvaluator",
]
