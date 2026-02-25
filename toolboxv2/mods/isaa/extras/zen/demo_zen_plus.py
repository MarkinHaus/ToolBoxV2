"""
ZenPlus Phase 3 Demo — 3D graph + sub-agents + full detail views.

python demo_zen_plus.py          # single agent (3D graph visible)
python demo_zen_plus.py multi    # 3 agents + sub-agents + 2 jobs
python demo_zen_plus.py jobs     # jobs only

Navigation:
  Grid:   Tab/↑↓←→ select, Enter=focus, Esc=exit
  Focus:  ↑↓ scroll, g=3D graph, t=tools, i=iterations, h=thoughts, Esc=grid
  Detail: ↑↓ navigate items, Esc=back
"""

import asyncio
import random
from zen_plus import ZenPlus


async def fake_stream(zp: ZenPlus, name: str, steps: int = 6,
                      delay=(0.3, 1.0), sub_agents: list[str] = None):
    tools = ["vfs_read", "vfs_write", "tool_web_search", "docker_run",
             "vfs_grep", "spawn_sub_agent", "vfs_execute", "tool_goto"]
    thoughts = [
        "Analyzing the project structure to find entry points and dependencies. "
        "The import graph suggests a circular dependency between auth and db modules "
        "which could cause the initialization race condition we're seeing.",
        "The test results suggest a regression in the authentication module. "
        "Specifically, the JWT validation is failing for tokens issued before the "
        "key rotation that happened in the last deployment.",
        "I should check the Docker container logs for the deployment error. "
        "The last successful build was 3 hours ago, and the CI pipeline shows "
        "a timeout in the integration test stage.",
        "This pattern matches what we saw in the config parser issue from sprint 14. "
        "The YAML loader is not handling multi-line strings with special characters "
        "correctly when the file encoding is UTF-8 with BOM.",
        "Need to validate the API response schema before proceeding. "
        "The OpenAPI spec shows required fields that are missing from the "
        "actual response when the user has no active subscription.",
    ]
    json_results = [
        '{"success": true, "content": "# Config Module\\nversion = 2.3\\nhost = localhost\\nport = 8080\\n\\n[database]\\nurl = postgres://localhost/app\\npool_size = 10", "lines": 8, "file_type": "cfg"}',
        '{"success": true, "content": "# Main Entry\\n\\ndef main():\\n    app = create_app()\\n    app.run(host=\\"0.0.0.0\\", port=8080)\\n\\nif __name__ == \\"__main__\\":\\n    main()", "lines": 7, "file_type": "py"}',
        '{"success": true, "matches": 5, "results": ["/src/auth.py:42", "/src/db.py:15"]}',
        '{"success": false, "error": "Connection refused on port 5432", "stderr": "psql: could not connect"}',
        '{"success": true, "content": "# API Response\\n```json\\n{\\n  \\"user\\": \\"admin\\",\\n  \\"permissions\\": [\\"read\\", \\"write\\"]\\n}\\n```", "file_type": "md"}',
    ]

    for i in range(1, steps + 1):
        base = {"agent": name, "iter": i, "max_iter": steps,
                "tokens_used": i * 1800, "tokens_max": 18000}
        if i == 1:
            base["persona"] = random.choice(["engineer", "architect", "analyst"])
            base["skills"] = random.sample(
                ["code_analysis", "testing", "docker", "web_search", "file_ops", "debugging"], 3)

        # Reasoning (LONG thoughts)
        zp.feed_chunk({**base, "type": "reasoning",
                       "chunk": random.choice(thoughts)})
        await asyncio.sleep(random.uniform(*delay) * 0.5)

        # Tools (1-3 per iteration)
        for _ in range(random.randint(1, 3)):
            tool = random.choice(tools)
            zp.feed_chunk({**base, "type": "tool_start", "name": tool,
                           "args": f'{{"path": "/src/{name}/mod_{i}.py", "query": "def main", "encoding": "utf-8"}}'})
            await asyncio.sleep(random.uniform(0.2, 0.6))

            result = random.choice(json_results)
            zp.feed_chunk({**base, "type": "tool_result", "name": tool, "result": result})
            await asyncio.sleep(0.1)

        # Sub-agent work (if configured)
        if sub_agents and i >= 2 and random.random() > 0.4:
            sub = random.choice(sub_agents)
            sub_base = {**base, "is_sub": True, "_sub_agent_id": sub}
            zp.feed_chunk({**sub_base, "type": "reasoning",
                           "chunk": f"Sub-agent {sub}: Investigating the test failures in module {name}. "
                                    f"Running isolated test suite with coverage tracking enabled."})
            await asyncio.sleep(0.3)
            zp.feed_chunk({**sub_base, "type": "tool_start", "name": "docker_run",
                           "args": f'{{"command": "pytest tests/{name}/ -v --cov"}}'})
            await asyncio.sleep(0.5)
            success = random.random() > 0.3
            zp.feed_chunk({**sub_base, "type": "tool_result", "name": "docker_run",
                           "result": f'{{"success": {str(success).lower()}, "stdout": "{'12 passed' if success else '3 failed, 9 passed'}", "exit_code": {0 if success else 1}}}'})

        # Content
        lines = [
            f"Step {i}: Analyzed {random.randint(3, 15)} files in {name} module.\n",
            f"Found {random.randint(1, 20)} matches, {random.randint(0, 3)} critical issues.\n",
            f"Updated {random.randint(1, 5)} configurations for '{name}'.\n",
        ]
        for line in random.sample(lines, k=random.randint(1, 2)):
            zp.feed_chunk({**base, "type": "content", "chunk": line})
            await asyncio.sleep(0.05)

    zp.feed_chunk({"agent": name, "type": "done", "success": True,
                    "iter": steps, "max_iter": steps})


async def fake_job(zp, tid, agent, query, dur=3.0, kind="job"):
    zp.inject_job(tid, agent, query, "running", kind=kind)
    await asyncio.sleep(dur)
    zp.update_job(tid, "completed" if random.random() > 0.1 else "failed")


async def demo_single():
    zp = ZenPlus.get()
    zp.clear_panes()
    t = asyncio.create_task(fake_stream(zp, "main-agent", 8, sub_agents=["tester", "linter"]))
    await zp.start()
    if not t.done():
        t.cancel()
        try: await t
        except asyncio.CancelledError: pass
    print("\n  ◎ Demo ended.\n")


async def demo_multi():
    zp = ZenPlus.get()
    zp.clear_panes()
    tasks = [
        asyncio.create_task(fake_stream(zp, "architect", 5, (0.5, 1.2), ["planner"])),
        asyncio.create_task(fake_stream(zp, "engineer", 8, (0.3, 0.8), ["tester", "linter"])),
        asyncio.create_task(fake_stream(zp, "reviewer", 4, (0.6, 1.0))),
        asyncio.create_task(fake_job(zp, "job_1", "coder", "fix auth bug #42", 10, "job")),
        asyncio.create_task(fake_job(zp, "bg_1", "deployer", "deploy v2.3.1", 15, "bg")),
    ]
    async def _wait():
        await asyncio.gather(*tasks[:3], return_exceptions=True)
        zp.signal_stream_done()
    w = asyncio.create_task(_wait())
    await zp.start()
    for t in tasks + [w]:
        if not t.done():
            t.cancel()
            try: await t
            except asyncio.CancelledError: pass
    print("\n  ◎ Multi-agent demo ended.\n")


async def demo_jobs():
    zp = ZenPlus.get()
    zp.clear_panes()
    zp.feed_chunk({"agent": "scheduler", "type": "content",
                   "chunk": "Job scheduler active.\n", "iter": 1, "max_iter": 1})
    jobs = [
        asyncio.create_task(fake_job(zp, "j1", "coder", "CSS fix", 4, "job")),
        asyncio.create_task(fake_job(zp, "j2", "analyst", "Q3 report", 6, "job")),
        asyncio.create_task(fake_job(zp, "bg1", "backup", "nightly", 8, "bg")),
        asyncio.create_task(fake_job(zp, "d1", "reviewer", "PR #88", 5, "delegate")),
    ]
    async def tick():
        for i in range(20):
            await asyncio.sleep(1.5)
            r = sum(1 for j in zp._jobs.values() if j.status == "running")
            zp.feed_chunk({"agent": "scheduler", "type": "content",
                           "chunk": f"  [{i+1}] {r} jobs running\n", "iter": 1, "max_iter": 1})
    t = asyncio.create_task(tick())
    await zp.start()
    for x in jobs + [t]:
        if not x.done():
            x.cancel()
            try: await x
            except asyncio.CancelledError: pass
    print("\n  ◎ Jobs demo ended.\n")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "single"
    {"single": demo_single, "multi": demo_multi, "jobs": demo_jobs}.get(mode, demo_single)
    asyncio.run({"single": demo_single, "multi": demo_multi, "jobs": demo_jobs}[mode]())
