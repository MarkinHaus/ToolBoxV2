import json
import os
import threading
import time

from litellm import max_tokens

from toolboxv2.utils.Irings.network import NetworkManager
from toolboxv2.utils.Irings.one import IntelligenceRing

from dataclasses import dataclass

from typing import List, Dict, Optional


@dataclass
class AgentState:
    current_focus: str  # Current ring ID being focused on
    last_action: Dict[str, any]  # Structured last action info
    connection_strengths: Dict[str, float]
    inner_state: Dict[str, any]  # Track inner monologue and inference state


class CognitiveNetwork:
    def __init__(
        self,
        user_llm,  # Interface LLM for user interaction
        agent_llm,  # Internal agent LLM
        initial_ring_content: str,
        name: str = "main",
        num_empty_rings: int = 32,
        inference_time: float = 60 * 60 * 2,
        auto_save: bool = True,
        network_file="network.hex",
        display=False,
        think_llm=None,
        format_class=None,
        rs_config=None,
        tk=False,
        web=False,
    ):
        self._processing_thread = None
        self.name = name
        self.user_llm = user_llm
        self.think_llm = think_llm if think_llm is not None else user_llm
        self.format_class = format_class if format_class is not None else user_llm
        self.agent_llm = agent_llm
        self.inference_time = inference_time

        # Initialize network
        preset_ring = IntelligenceRing("initial-ring-" + name)
        if len(preset_ring.concept_graph.concepts.keys()) == 0:
            preset_ring.process(initial_ring_content, name="Workflow-" + self.name, metadata={"importance": "high"})
        self.auto_save = auto_save
        self.network_file = network_file
        if auto_save and network_file is not None and os.path.exists(network_file):
            self.network = NetworkManager.load_from_file(network_file)
        else:
            self.network = NetworkManager(
                num_empty_rings=num_empty_rings,
                preset_rings=[preset_ring],
                name=self.name,
            )

        self._v = None
        self.display = None
        if display:
            if tk:
                from toolboxv2.utils.Irings.tk_live import NetworkVisualizer
            elif web:
                from toolboxv2.utils.Irings.nicegui_live import NetworkVisualizer
            else:
                raise ValueError("tk or web")
            v = NetworkVisualizer(self.network)
            if web:
                self.display = v.create_ui
            self._v = v
            self.og_processor = self.network.process_input

            def process_with_visualization(text, **k):
                """Process input text and update visualization"""
                self.network.resume()
                results = self.og_processor(text, v, **k)
                self.network.freeze()
                return results

            self.network.process_input = lambda text, **k: process_with_visualization(text, **k)

        # Initialize agent state with improved structure
        self.agent_state = AgentState(
            current_focus="initial-ring-" + self.name,
            last_action={
                "type": "initialization",
                "target_ring": "initial-ring-" + self.name,
                "timestamp": time.time(),
                "details": "System initialized"
            },
            connection_strengths={},
            inner_state={
                "monologue": [],
                "last_inference": time.time(),
                "thought_chain": [],
                "attention_focus": None
            }
        )

        self.lock = threading.Lock()
        self.network.freeze()
        self.network.resume()

        self.rs_config = rs_config or {
            "confidence_threshold": 0.5,
            "retrieval_threshold": 0.8,
            "max_notes": 2,
            "max_depth": 2,
            "max_branches_per_node": 1,
            "max_retrievals_per_branch": 2
        }

    def print_inner_state(self):
        print(json.dumps(self.agent_state.inner_state, indent=2))

    def print_last_monolog(self):
        print(json.dumps(self.agent_state.inner_state.get("monologue", [{}])[-1], indent=2))

    def internal_update(self):

        sto_llm = self.agent_llm
        self.agent_llm = self.think_llm

        self._v.mark_active(self.agent_state.current_focus) if self._v is not None else None
        if not self.agent_state.inner_state["monologue"]:
            context = self._inner_monologue("I Think Therefor I am?")
        elif time.time() - self.agent_state.inner_state["monologue"][-1].get("timestamp",
                                                                             time.time()) > self.inference_time:
            context = self._inner_monologue(self.agent_state.inner_state["monologue"][-1].get("thought"))
        else:
            context = self.agent_state.inner_state["monologue"][-1].get("thought")

        results = self.network.process_input(context, ref=self.agent_state.inner_state["monologue"][-1].get("thought"))

        for ring_id in results[::-1]:
            self._v.mark_active(ring_id) if self._v is not None else None
            time.sleep(0.25) if self._v is not None else None

        self._v.mark_active(self.agent_state.current_focus) if self._v is not None else None

        self.agent_llm = sto_llm

    def _inner_monologue(self, context: str) -> str:
        """Generate inner thoughts based on current context"""
        texts = [t for t in [x['metadata']['text'] if 'text' in x['metadata'] else x['name'] for x in self._get_active_concepts()]
         if self.network.rings[self.agent_state.current_focus].input_processor.pcs(t,context) > 0.2 ]
        concepts_str = "\n\nConcept: ".join(texts)

        thought = self.vFlare(context[:100000]+'...\n'+concepts_str[:10000]+'...')

        print("SYSTEM inner ms", thought)
        thought = str(thought)
        self.agent_state.inner_state["monologue"].append({
            "type": "inner_monologue",
            "timestamp": time.time(),
            "thought": thought,
        })

        self.agent_state.inner_state["last_inference"] = time.time()
        return thought

    def vFlare(self, prompt):
        from pydantic import BaseModel
        class FinalOutput(BaseModel):
            """Key thought reasoning"""
            thought: str

        return self.format_class(FinalOutput, prompt).thought

    def _get_active_concepts(self) -> List[Dict]:
        """Get currently active concepts from focused ring"""
        ring = self.network.rings[self.agent_state.current_focus]
        return [
                   {
                       "name": concept.name,
                       "metadata": concept.metadata,
                       "stage": concept.stage
                   }
                   for concept in ring.concept_graph.concepts.values()
                   if concept.stage > 0
               ][:25]

    def process_user_input(self, user_input: str, callback=None) -> str:
        """
        Process user input with immediate response and background processing

        Args:
            user_input: The input text from user
            callback: Optional function to handle the final processed result
        """
        # Get immediate interface response
        if not self.agent_state.inner_state['thought_chain']:
            self._inner_monologue(user_input)
        interface_response = self.user_llm.process(
            f"User input: {user_input[:100000]}\nProvide immediate clear response."
            f"PAST HISTORY Agen Inner thought: {self.agent_state.inner_state['thought_chain'][-1] if self.agent_state.inner_state['thought_chain'] else 'None'}\n"
            f"HISTORY END\nProvide an immediate, clear, and concise response to the user based on the available context."
            ,
            max_tokens=None
        )

        if callback is None:
            callback = lambda x: print("Callback AGENT response:", x)

        # Create background processing thread
        def background_processor():
            with self.lock:
                current_time = time.time()

                # Check if processing is already running
                if hasattr(self, '_processing_thread') and self._processing_thread.is_alive():
                    self._guided_thoughts(user_input)
                else:
                    if current_time - self.agent_state.inner_state["last_inference"] <= self.inference_time:
                        thought = self._inner_monologue(user_input)
                        self.agent_state.inner_state["thought_chain"].append(thought)
                        self.agent_state.inner_state["last_inference"] = current_time
                        self.internal_update()

                if len(user_input) > 200000:
                    self.network.process_input(user_input, ref=user_input)

                # Process through network
                res = self._process_agent_action(interface_response)

                # Get active concepts
                ac = '\n\n'.join([x.get('metadata', {}).get('text', '')
                                  for x in self._get_active_concepts()][:self.network.max_new])

                # Generate detailed response
                final_response = self.user_llm.process(
                    f"Agent result: {res}\n"
                    f"active_concepts: \n{ac}"
                    f"Agent Inner thought: {self.agent_state.inner_state['thought_chain'][-1] if self.agent_state.inner_state['thought_chain'] else 'None'}\n"
                    f"User Input: {user_input[:100000]}\n"
                    f"Analyze the context and user input, then generate a well - reasoned, detailed response that addresses the user's needs. Consider the active concepts and previous inner thoughts to provide a comprehensive solution.",
                    max_tokens=None
                )

                if self.auto_save:
                    self.network.save_to_file(self.network_file)

                # Call callback with final response if provided
                if callback:
                    callback(str(final_response))

        # Start background processing thread
        self._processing_thread = threading.Thread(target=background_processor)
        self._processing_thread.start()

        return str(interface_response)

    def _guided_thoughts(self, context: str):
        """
        Generate guided thoughts based on current context and previous processing
        """
        if not self.agent_state.inner_state["thought_chain"]:
            return

        last_thought = self.agent_state.inner_state["thought_chain"][-1]

        guided_prompt = f"""
        Previous thought: {last_thought}
        New context: {context}
        """
        thought = self.vFlare(guided_prompt)
        print("SYSTEM inner gt:", thought)
        self.agent_state.inner_state["monologue"].append({
            "type": "guided_thoughts",
            "timestamp": time.time(),
            "thought": thought,
        })
        self.agent_state.inner_state["last_inference"] = time.time()

    def _get_ring_info(self) -> Dict:
        """Get detailed information about current ring state"""
        return {
            "active_ring": self.agent_state.current_focus,
            "active_concepts": self._get_active_concepts(),
            "ring_connections": self.network.connections[self.agent_state.current_focus],
            "ring_metrics": self.network.rings[self.agent_state.current_focus].concept_graph.concepts.keys()
        }

    def _process_agent_action(self, instruction: str) -> str:
        """Handle internal agent's interaction with the network"""
        # Allow for inner thought before action
        thought = self._inner_monologue(instruction)

        # Get agent's action decision
        action_prompt = f"""
        Current focus: {self.network.rings[self.agent_state.current_focus].get_concept_by_id(self.network.rings[self.agent_state.current_focus].retrieval_system.find_similar(self.network.rings[self.agent_state.current_focus].input_processor.process_text(thought))[0])}
        Last action: {self.agent_state.last_action['details']}
        Inner thought: {thought}
        Active concepts: {[x['metadata']['text'] if 'text' in x['metadata'] else x['name'] for x in self._get_active_concepts()]}
        Instruction: {instruction}

        Perform next actions if necessary based on the current state of the system:
        """

        action = self.agent_llm.process(action_prompt, max_tokens=1500)
        action = str(action)
        # Process action through network
        result = self.network.process_input(instruction, ref=action)

        # Update agent state with structured action info
        self._update_agent_state(action, result)
        if len(self.network.metrics.latest_concepts) == 0:
            return action
        code = self.network.metrics.latest_concepts[-1]
        concept = self.network.get_concept_from_code(code)
        i = -2
        while concept is None:
            if abs(i) >= len(self.network.metrics.latest_concepts):
                return action
            code = self.network.metrics.latest_concepts[i]
            concept = self.network.get_concept_from_code(code)
            i -= 1

        return concept.metadata.get('text', action)

    def _update_agent_state(self, action: str, result: list):
        """Update agent's internal state"""
        self.agent_state.last_action = {
            "type": "action",
            "action_text": action,
            "result": result,
            "timestamp": time.time(),
            "target_ring": self.agent_state.current_focus,
            "details": f"Executed: {action} with result: {result}"
        }

        # Update connection strengths based on interaction
        for ring_id in self.network.rings:
            if ring_id not in self.agent_state.connection_strengths:
                self.agent_state.connection_strengths[ring_id] = 0.5

            if ring_id in result:
                self.agent_state.connection_strengths[ring_id] += 0.1
            else:
                self.agent_state.connection_strengths[ring_id] *= 0.95
        self.agent_state.current_focus = list(self.agent_state.connection_strengths.keys())[
            list(self.agent_state.connection_strengths.values())
            .index(max(self.agent_state.connection_strengths.values()))]

    def get_agent_status(self) -> Dict:
        """Get current status of the agent"""
        return {
            "current_focus": self.agent_state.current_focus,
            "last_action": self.agent_state.last_action,
            "connection_strengths": self.agent_state.connection_strengths,
            "inner_state": {
                "last_thought": self.agent_state.inner_state["thought_chain"][-1] if self.agent_state.inner_state[
                    "thought_chain"] else None,
                "inference_count": len(self.agent_state.inner_state["thought_chain"]),
                "attention_focus": self.agent_state.inner_state["attention_focus"]
            },
            "network_metrics": self.network.get_metrics(),
            "ring_info": self._get_ring_info()
        }


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class ImprovedLLM:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Move model to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def process(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            # Prepare inputs
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512  # Limit input length
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.2
                )

            # Decode and clean response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )

            # Remove the input prompt from the response
            prompt_length = len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
            response = response[prompt_length:].strip()

            return response

        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            return f"Error processing input: {str(e)}"


class AgentLLM:
    def __init__(self, name, model_name: str = "groq/llama-3.1-8b-instant"):
        from toolboxv2.mods.isaa.Agents import Agent, get_free_agent_data_factory

        self.agent = Agent(
            amd=get_free_agent_data_factory(name, model_name),  # model="GPT4All-13B-snoozy.ggmlv3.q4_0.bin"
            stream=False
        )

    def process(self, prompt: str, max_tokens: int = 100) -> str:
        try:
            return self.agent.run_model(llm_message=self.agent.get_llm_message(prompt), max_tokens=max_tokens)
        except Exception as e:
            print(f"Error in LLM processing: {str(e)}")
            return f"Error processing input: {str(e)}"


# Example usage

def t_llm():
    # Initialize the improved LLM
    llm = ImprovedLLM("gpt2")

    # Test with some prompts
    test_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks briefly."
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = llm.process(prompt)
        print(f"Response: {response}")

    llm = AgentLLM("llama-3.1")

    # Test with some prompts
    test_prompts = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain neural networks briefly."
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        response = llm.process(prompt)
        print(f"Response: {response}")


def main():
    # Initialize LLMs
    user_interface_llm = AgentLLM("user")  # For user interaction
    agent_llm = AgentLLM("internal")  # For internal processing

    # Initial content for the base ring
    initial_content = """
    Think deep and oly for ur self
    """

    # Create cognitive network
    cognitive_net = CognitiveNetwork(
        user_llm=user_interface_llm,
        format_class=agent_llm.agent.format_class,
        agent_llm=agent_llm,
        initial_ring_content=initial_content,
        inference_time=5,
        num_empty_rings=4,
        auto_save=False,
        name="test1",
        display=False,
    )

    t = time.process_time()
    cognitive_net.internal_update()
    print("time.process_time() - t:", time.process_time() - t)


    cognitive_net.print_last_monolog()

    # Example interaction
    user_inputs = [
        "Hello, can you help me solve a math problem?",
        "What's 25 times 16?",
        "Can you reset your thinking?",
        "Let's try a different topic"
    ]
    response = cognitive_net.process_user_input("Hi main Name Ist Markin!")

    cognitive_net.print_last_monolog()
    print(response)

    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        response = cognitive_net.process_user_input(user_input)
        cognitive_net.print_last_monolog()
        print(f"System: {response}")
        # Print agent status after each interaction
        cognitive_net.network.show.display_state('metrics')

    cognitive_net.print_inner_state()
    input("WAIT for")
    cognitive_net.print_inner_state()

    while user_input := input():
        response = cognitive_net.process_user_input(user_input)
        cognitive_net.print_last_monolog()
        print(f"System: {response}")
    input()


if __name__ == "__main__":
    #network = NetworkManager.load_from_file("network.hex")
    #v = NetworkVisualizer(network)
    #input("DONE?")

    main()
