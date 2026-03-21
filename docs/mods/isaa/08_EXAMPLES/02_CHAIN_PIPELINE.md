# 02_CHAIN_PIPELINE - Agent Chains

## Problem
Wie verbinde ich mehrere Agenten/Schritte zu einer Pipeline?

## Loesung
- Sequenz - Schritt A -> B -> C
- Parallel - A, B, C gleichzeitig

## Code

    class ChainPipeline:
        def add_step(self, name, func, **kwargs):
            self.steps.append({"name": name, "func": func})
            return self
        
        def execute(self, initial_input):
            for step in self.steps:
                result = step["func"]()
                if not result.get("success"):
                    return final_answer(answer=f"Fehlgeschlagen", success=False)
            return context["results"]

## Parallel mit Sub-Agents

    spawn_sub_agent(task="Task A", output_dir="/workspace/a/", wait=False)
    spawn_sub_agent(task="Task B", output_dir="/workspace/b/", wait=False)
    results = wait_for([agent_a, agent_b], timeout=300)
