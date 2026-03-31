import asyncio
import logging
import os
from typing import Callable, Dict, Optional

from toolboxv2.utils.workers.event_manager import Event, EventType, ZMQEventManager

logger = logging.getLogger(__name__)

_registry_instance: Optional["InterfaceRegistry"] = None


def get_registry() -> "InterfaceRegistry":
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = InterfaceRegistry()
    return _registry_instance


class InterfaceRegistry:

    def __init__(
        self,
        pub_endpoint: str = "tcp://127.0.0.1:5555",
        sub_endpoint: str = "tcp://127.0.0.1:5556",
        req_endpoint: str = "tcp://127.0.0.1:5557",
        rep_endpoint: str = "tcp://127.0.0.1:5557",
    ):
        try:
            from toolboxv2.utils.workers.config import load_config
            cfg = load_config()
            zmq_cfg = getattr(cfg, "zmq", None)
            if zmq_cfg:
                pub_endpoint = getattr(zmq_cfg, "pub_endpoint", pub_endpoint)
                sub_endpoint = getattr(zmq_cfg, "sub_endpoint", sub_endpoint)
                req_endpoint = getattr(zmq_cfg, "req_endpoint", req_endpoint)
                rep_endpoint = getattr(zmq_cfg, "rep_endpoint", rep_endpoint)
        except Exception as e:
            logger.debug(f"init error: {e}")
        self._pub_endpoint = pub_endpoint
        self._sub_endpoint = sub_endpoint
        self._req_endpoint = req_endpoint
        self._rep_endpoint = rep_endpoint
        self._connected = False
        self._event_manager: Optional[ZMQEventManager] = None
        self._worker_id = f"iface_registry_{os.getpid()}"
        self._subs: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Liveness
    # ------------------------------------------------------------------

    def probe_broker(self, timeout_ms: int = 200) -> bool:
        import socket
        try:
            host, port = self._pub_endpoint.replace("tcp://", "").split(":")
            with socket.create_connection((host, int(port)), timeout=timeout_ms / 1000):
                return True
        except (ConnectionRefusedError, TimeoutError, OSError):
            return False

    def is_online(self) -> bool:
        return self._connected

    def health_check(self) -> bool:
        """Aktiver probe — nur für startup/reconnect, nicht per-chunk."""
        return self.probe_broker(timeout_ms=50)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        if not self.probe_broker():
            logger.warning(
                "InterfaceRegistry: broker not reachable — "
                "start via cli_worker_manager first"
            )
            return False
        self._event_manager = ZMQEventManager(
            worker_id=self._worker_id,
            pub_endpoint=self._pub_endpoint,
            sub_endpoint=self._sub_endpoint,
            req_endpoint=self._req_endpoint,
            rep_endpoint=self._rep_endpoint,
            is_broker=False,
        )
        await self._event_manager.start()
        self._connected = True
        logger.info(f"InterfaceRegistry connected as {self._worker_id}")
        return True

    async def stop(self) -> None:
        self._connected = False
        if self._event_manager:
            try:
                await self._event_manager.stop()
            except Exception as e:
                logger.debug(f"stop error: {e}")
        self._event_manager = None

    # ------------------------------------------------------------------
    # PUB / SUB
    # ------------------------------------------------------------------

    async def publish(self, id: str, data: dict) -> bool:
        if not self._connected or self._event_manager is None:
            return False
        try:
            event = Event(
                type=EventType.CUSTOM,
                source=self._worker_id,
                target=id,
                payload=data,
            )
            await self._event_manager.publish(event)
            return True
        except Exception as e:
            logger.debug(f"publish failed [{id}]: {e}")
            return False

    def publish_sync(self, id: str, data: dict) -> None:
        """Für sync context (_ingest_chunk). Fire-and-forget."""
        if not self._connected:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(id, data))
        except RuntimeError:
            pass  # kein loop → skip, nie crashen

    def register_sub(
        self,
        id: str,
        callback: Callable[[dict], None],
        filter_prefix: bool = False,  # True → startswith(id), False → exact match
    ) -> bool:
        if not self._connected or self._event_manager is None:
            return False

        async def _wrapper(event: Event) -> None:
            # prefix mode: "icli.task" matched "icli.task.abc123"
            # exact mode:  "icli.task.abc123" matched nur "icli.task.abc123"
            match = event.target.startswith(id) if filter_prefix else event.target == id
            if not match:
                return
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event.payload)
                else:
                    callback(event.payload)
            except Exception as e:
                logger.debug(f"sub callback error [{id}]: {e}")

        self._event_manager.on(EventType.CUSTOM, _wrapper)
        self._event_manager.subscribe(id)  # ZMQ socket-level prefix filter
        self._subs[id] = callback
        return True

