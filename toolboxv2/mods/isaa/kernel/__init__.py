from .instace import Kernel
from .kernelin import CLIKernel, DiscordKernel, WhatsAppKernel
from .types import KernelConfig, KernelState, KernelMetrics
from .types import InteractionType, LearningRecord, UserPreferences, Memory, MemoryType, ScheduledTask, TaskStatus
from .types import Signal, SignalType, ISignalBus, UserState, UserContext, IStateMonitor, ProactivityContext, ProactivityDecision
from .types import IDecisionEngine, IProAKernel, IOutputRouter, ConsoleOutputRouter

from .models import ContextStore, ProactiveActionTracker, LearningEngine, MemoryStore, TaskScheduler, WebSocketOutputRouter, MultiChannelRouter, AgentIntegrationLayer
from .kernelin.tools.discord_tools import DiscordKernelTools
from .kernelin.tools.whatsapp_tools import WhatsAppKernelTools

__version__ = "1.0.0"
__all__ = ["Kernel", "CLIKernel", "KernelConfig", "KernelState", "KernelMetrics", "InteractionType",
           "LearningRecord", "UserPreferences", "Memory", "MemoryType", "ScheduledTask",
           "TaskStatus", "Signal", "SignalType", "ISignalBus", "UserState", "UserContext",
           "IStateMonitor", "ProactivityContext", "ProactivityDecision", "IDecisionEngine",
           "IProAKernel", "IOutputRouter", "ConsoleOutputRouter", "ContextStore", "ProactiveActionTracker",
           "LearningEngine", "MemoryStore", "TaskScheduler", "WebSocketOutputRouter", "MultiChannelRouter",
           "AgentIntegrationLayer", "DiscordKernelTools", "WhatsAppKernelTools"]
