# -*- coding: utf-8 -*-
"""Ersatz-LLM fuer die icli2-Betriebstests: OpenAI-kompatibler HTTP-Server (Prozessgrenze, Regel 2).

Der Agent spricht ueber litellm per HTTP mit diesem Server - genau wie mit einem echten Anbieter.
Alle Anfragen werden mitgeschrieben (`requests`), damit ein Test pruefen kann, was beim Modell ankommt.

Steuerung ueber die letzte Nutzer-Nachricht des Laufs:
  CALL <tool> <json-args>   -> das Modell ruft genau dieses Tool einmal auf (bei Bedarf vorher load_tools),
                               danach final_answer mit dem Tool-Ergebnis
  sonst                     -> final_answer mit einem Echo der Nachricht
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_CALL = re.compile(r"CALL\s+(\w+)\s+(\{.*\})\s*$", re.S)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else json.dumps(content)


class FakeLLM:
    def __init__(self):
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        owner = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self._send(200, {"data": [{"id": "fake"}]})

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                call, text = owner._decide(body)
                if body.get("stream"):
                    self._stream(call, text)
                else:
                    msg = {"role": "assistant", "content": text}
                    if call:
                        msg["tool_calls"] = [{"id": f"call_{time.time_ns()}", "type": "function",
                                              "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
                    self._send(200, {"id": "c1", "object": "chat.completion", "model": "fake",
                                     "choices": [{"index": 0, "message": msg,
                                                  "finish_reason": "tool_calls" if call else "stop"}],
                                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

            def _stream(self, call, text):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()

                def ev(delta, fin=None):
                    d = {"id": "c1", "object": "chat.completion.chunk", "model": "fake",
                         "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                    self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode())
                    self.wfile.flush()

                ev({"role": "assistant", "content": ""})
                if call:
                    ev({"tool_calls": [{"index": 0, "id": f"call_{time.time_ns()}", "type": "function",
                                        "function": {"name": call[0], "arguments": json.dumps(call[1])}}]})
                    ev({}, "tool_calls")
                else:
                    ev({"content": text})
                    ev({}, "stop")
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}/v1"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> "FakeLLM":
        self._thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    # -- Verhalten ---------------------------------------------------------
    def _decide(self, body: dict):
        msgs = body.get("messages", []) or []
        tools = [t["function"]["name"] for t in body.get("tools", []) or []]
        idx = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=-1)
        user = _text(msgs[idx].get("content")) if idx >= 0 else ""
        after = msgs[idx + 1:] if idx >= 0 else []
        called = [tc.get("function", {}).get("name") for m in after if m.get("role") == "assistant"
                  for tc in (m.get("tool_calls") or [])]
        results = [_text(m.get("content")) for m in after if m.get("role") == "tool"]
        with self._lock:
            self.requests.append({"user": user, "tools": tools, "called": called, "results": results,
                                  "messages": msgs})
        m = _CALL.search(user)
        if m:
            name, args = m.group(1), json.loads(m.group(2))
            if name not in called:
                if name in tools:
                    return (name, args), None
                if "load_tools" in tools and called.count("load_tools") < 2:
                    return ("load_tools", {"tools": [name]}), None
            if "final_answer" in tools:
                return ("final_answer", {"answer": f"Tool {name}: " + (results[-1][:300] if results else "-"),
                                         "success": True}), None
        if "final_answer" in tools:
            return ("final_answer", {"answer": "Echo: " + user[-200:], "success": True}), None
        return None, "Echo: " + user[-200:]

    def prompts(self) -> list[str]:
        with self._lock:
            return [r["user"] for r in self.requests]
