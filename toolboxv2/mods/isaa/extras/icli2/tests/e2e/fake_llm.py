"""OpenAI-kompatibles Ersatz-LLM: 1. Runde Tool-Aufruf, danach final_answer (gestreamt)."""
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
LOG = open("/tmp/e2e/llm.log", "a")
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"data":[{"id":"fake"}]}')
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        msgs, tools = body.get("messages", []), [t["function"]["name"] for t in body.get("tools", []) or []]
        had_tool = any(m.get("role") == "tool" for m in msgs)
        last = next((m for m in reversed(msgs) if m.get("role") == "user"), {})
        user = last.get("content") if isinstance(last.get("content"), str) else json.dumps(last.get("content"))
        LOG.write(json.dumps({"stream": body.get("stream"), "n_msgs": len(msgs), "tools": tools[:40], "had_tool": had_tool, "user": (user or "")[-160:]}) + "\n"); LOG.flush()
        n_tool = sum(1 for m in msgs if m.get("role") == "tool")
        first_user = next((m.get("content") for m in msgs if m.get("role") == "user"), "") or ""
        first_user = first_user if isinstance(first_user, str) else json.dumps(first_user)
        if "fehler429" in (user or ""):                  # echter 429 im OpenRouter-Format
            body = json.dumps({"error": {"message": "Rate limit exceeded: free-models-per-day. Add 10 credits to unlock 1000 free model requests per day", "code": 429}}).encode()
            self.send_response(429); self.send_header("Content-Type", "application/json"); self.send_header("retry-after", "1")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        call = None
        if "limitshell" in (user or "") and "vfs_shell" in tools:
            call = ("vfs_shell", {"command": f"ls /x{n_tool}"})
        elif "limit-test" in (user or "") and "think" in tools:
            call = ("think", {"thought": f"Ich denke weiter nach (Schritt {n_tool + 1})."})
        elif "SUBTASK" in first_user and tools:          # Sub-Agent: zwei Werkzeug-Schritte, dann Ergebnis
            open("/tmp/e2e/subtools.log", "a").write(json.dumps(tools) + "\n")
            step = next((t for t in ("look", "vfs_shell", "plan") if t in tools), None)
            if n_tool < 2 and step:
                call = (step, {"path": f"/sub{n_tool}", "command": f"ls /sub{n_tool}", "reason": "Ordner ansehen"})
            elif "final_answer" in tools:
                call = ("final_answer", {"answer": "## Sub-Ergebnis\n\nZwei Ordner gelesen, alles in Ordnung.", "success": True})
        elif "subagent-test" in (user or "") and "spawn_sub_agent" in tools and n_tool == 0:
            call = ("spawn_sub_agent", {"task": "SUBTASK: Lies die Ordner /sub0 und /sub1.\nFasse danach kurz zusammen.", "wait": True, "budget": 20000, "max_iterations": 6})
        if call is not None:
            pass
        elif "think" in tools and n_tool == 0: call = ("think", {"thought": "Ich sehe zuerst im Dateisystem nach."})
        elif "vfs_shell" in tools and n_tool == 1: call = ("vfs_shell", {"command": "ls /"})
        elif "final_answer" in tools: call = ("final_answer", {"answer": "## Fertig\n\nIch habe **vfs_shell** benutzt. Echo: " + (user or "")[-40:], "success": True})
        text = None if call else "Antwort ohne Tools. Echo: " + (user or "")[-60:]
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            def ev(delta, fin=None):
                d = {"id": "c1", "object": "chat.completion.chunk", "model": "fake", "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                self.wfile.write(("data: " + json.dumps(d) + "\n\n").encode()); self.wfile.flush(); time.sleep(0.01)
            ev({"role": "assistant", "content": ""})
            if call:
                args = json.dumps(call[1])
                ev({"tool_calls": [{"index": 0, "id": "call_" + str(int(time.time()*1000)), "type": "function", "function": {"name": call[0], "arguments": ""}}]})
                for i in range(0, len(args), 25): ev({"tool_calls": [{"index": 0, "function": {"arguments": args[i:i+25]}}]})
                ev({}, "tool_calls")
            else:
                for w in text.split(" "): ev({"content": w + " "})
                ev({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            msg = {"role": "assistant", "content": text}
            if call: msg["tool_calls"] = [{"id": "call_1", "type": "function", "function": {"name": call[0], "arguments": json.dumps(call[1])}}]
            out = {"id": "c1", "object": "chat.completion", "model": "fake", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if call else "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(json.dumps(out).encode())
ThreadingHTTPServer(("127.0.0.1", 8765), H).serve_forever()
