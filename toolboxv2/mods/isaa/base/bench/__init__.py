import asyncio
import json
import os
from typing import Dict, List, Any

# from benchmark import Benchmark

# # Schnell
# report = Benchmark().run_sync(my_model, mode='quick')

# # Mit deinem Framework
# from benchmark import MAKERAdapter
# adapter = MAKERAdapter(flow_agent)
# report = await adapter.benchmark("gpt-4", mode="standard")

# # Vergleich (gleicher Seed)
# report_a = await bench.run(model_a, seed=42)
# report_b = await bench.run(model_b, seed=42)

from benchmark import Benchmark, Report, RowModelAdapter, MAKERAdapter, AgentAdapter
from dashboard import Dashboard

filepath_ = r"reports6.json"

def load_reports_from_file(filepath: str) -> Dict[str,List[Report]]:
    if not os.path.exists(filepath):
        print("no reports found")
        return {
        "all":[],
        "Row": [],
        "Agent": [], # TODO add apdapter
        "MADAP": [], # TODO fix
    } # TODO: fix , android build server logtim running agnet web safty disckrd bot oline update web installer
    with open(filepath) as f:
        data = json.load(f)
        return data

def save_reports_to_file(filepath: str, reports: Dict[str,List[Dict[str, Any]]]):
    with open(filepath, 'w') as f:
        json.dump({k: [r.to_dict() for r in v] for k, v in reports.items()}, f, indent=2)

async def main():
    # Benchmarks laufen lassen
    reports = load_reports_from_file(filepath_)
    models= {
        # Local Potential
        # Mistral
        "mistral-8b": "openrouter/mistralai/ministral-8b-2512",
        "mistral-3b": "openrouter/mistralai/ministral-3b-2512",
        "devstral":"openrouter/mistralai/devstral-2512:free",
        "mistral-c": "openrouter/mistralai/mistral-small-creative",
        # # kimi
        "kimi-k2": "openrouter/moonshotai/kimi-k2-0905",
        # # Llama
        "llama-3.1-8b": "openrouter/meta-llama/llama-3.1-8b-instruct",
        "llama-3.1-70b": "openrouter/meta-llama/llama-3.3-70b-instruct",
        # glm-4.6
        # "glm-4.6": "openrouter/z-ai/glm-4.6", way to slow 892s
        # # olmo
        "olmo-7b": "openrouter/allenai/olmo-3-7b-instruct",
        # # gemma
        "gemma-3-12": "openrouter/google/gemma-3-12b-it",
        "gemma-3-27b": "openrouter/google/gemma-3-27b-it",
        # # gemini
        "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
        "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
        "gemini-2.5-flash-lite": "openrouter/google/gemini-2.5-flash-lite",
        # # Claude
        "claude-haiku-4.5": "openrouter/anthropic/claude-haiku-4.5",
        # # Grok
        "grok-code-fast-1": "openrouter/x-ai/grok-code-fast-1",
        "grok-fast-4": "openrouter/x-ai/grok-4-fast",
        # # deepseek to slow
        # "deepseek-v3.2-s": "openrouter/deepseek/deepseek-v3.2-speciale",
        # "deepseek-v3.2": "openrouter/deepseek/deepseek-v3.2",
        # # qwen
        "qwen-3-vl-32b": "openrouter/qwen/qwen3-vl-32b-instruct",
        # "qwen3-vl-8b-thinking": "openrouter/qwen/qwen3-vl-8b-thinking", # to slow
        "qwen3-plus": "openrouter/qwen/qwen3-coder-plus",
        "qwen3-flash": "openrouter/qwen/qwen3-coder-flash",
        # Flagship
        #"gpt-5.1": "openrouter/openai/gpt-5.1",
        #"gpt-5.2": "openrouter/openai/gpt-5.2",
        "claude-4.5-opus": "openrouter/anthropic/claude-opus-4.5",
        # "gemini-3-pro": "openrouter/google/gemini-3-pro-preview",
        # "Mistral-Large": "openrouter/mistralai/mistral-large-2512",

    }
    models = {
        # "gemma-3-27b": "openrouter/google/gemma-3-27b-it",
        #"qwen3-vl-8b-thinking": "openrouter/qwen/qwen3-vl-8b-thinking",
        "qwen-3-vl-32b": "openrouter/qwen/qwen3-vl-32b-instruct",
        # "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
        # "gemini-2.5-flash-lite": "openrouter/google/gemini-2.5-flash-lite",
        "olmo-7b": "openrouter/allenai/olmo-3-7b-instruct",
        "nemotron-3": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        "gemma-3n-e4b": "openrouter/google/gemma-3n-e4b-it",
        "gemini-3-flash": "openrouter/google/gemini-3-flash-preview",
        "mixtral-8x22b": "openrouter/mistralai/mixtral-8x22b-instruct",
        "mixtral-8x7b": "openrouter/mistralai/mixtral-8x7b-instruct",
    }
    models_ = {
        "grok-code-fast-1": "openrouter/x-ai/grok-code-fast-1",
        "grok-fast-4": "openrouter/x-ai/grok-4-fast",
        "gemini-2.5-flash": "openrouter/google/gemini-2.5-flash",
        "claude-haiku-4.5": "openrouter/anthropic/claude-haiku-4.5",
        "qwen3-plus": "openrouter/qwen/qwen3-coder-plus",
        "mistral-8b": "openrouter/mistralai/ministral-8b-2512",
    }

    from toolboxv2 import get_app
    app = get_app()
    isaa = app.get_mod("isaa")
    agent = await isaa.get_agent("self")
    seed = 7645
    # run all row test in parallel

    async def run_row(m_id:str, model_name:str):
        adapter = RowModelAdapter(agent, model_name=model_name)
        res = await adapter.benchmark(m_id, mode='full', seed=seed)
        return res, 'Row'

    async def run_maker(m_id:str, model_name:str):
        agent.amd.fast_llm_model = model_name
        agent.amd.complex_llm_model = model_name
        adapter = MAKERAdapter(agent)
        res = await adapter.benchmark(m_id+'MADAP', mode='full', seed=seed)
        return res, 'MADAP'

    async def run_agent(m_id:str, model_name:str):
        agent.amd.fast_llm_model = model_name
        agent.amd.complex_llm_model = model_name
        adapter = AgentAdapter(agent)
        res = await adapter.benchmark(m_id+'Agent', mode='full', seed=seed)
        return res, 'Agent'

    tasks = []
    for model_id, model_sig in models.items():
        # creat all tasks
        do = True
        for model_id_s in reports["all"]:
            print(model_id_s, type(model_id_s))
            model_id_s_name = model_id_s.model if hasattr(model_id_s, "model") else model_id_s.model_id if hasattr(model_id_s, "model_id") else model_id_s.get("model") if isinstance(model_id_s, dict) else model_id_s
            if model_id == model_id_s_name:
                do = False
                break
        if not do:
            continue
        tasks.append(asyncio.create_task(run_agent(model_id, model_sig)))
        tasks.append(asyncio.create_task(run_row(model_id, model_sig)))
        tasks.append(asyncio.create_task(run_maker(model_id, model_sig)))
    # wait for all tasks to complete
    import tqdm
    for task in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        report, key = await task
        reports["all"].append(report)
        reports[key].append(report)
        # save_reports_to_file(filepath_, reports)

    # Dashboard generieren
    # Dashboard.save(reports["MADAP"], "MADAP_comparison.html")
    Dashboard.save(reports["all"], "comparisonALLMo7.html")
    # Dashboard.save(reports["Agent"], "comparisonAgent.html")

if __name__ == "__main__":
    asyncio.run(main())
