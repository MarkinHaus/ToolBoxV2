import json
import os
from typing import List, Dict
from dataclasses import dataclass
import threading
import time

from toolboxv2.utils.Irings.network import NetworkManager
from toolboxv2.utils.Irings.tk_live import NetworkVisualizer
from toolboxv2.utils.Irings.zero import IntelligenceRing


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
        num_empty_rings: int = 32,
        inference_time: float = 0.5,  # Time in seconds for inner inference
        auto_save: bool = True,
        network_file="network.hex",
        display = False,

    ):
        self.user_llm = user_llm
        self.agent_llm = agent_llm
        self.inference_time = inference_time

        # Initialize network
        preset_ring = IntelligenceRing("initial-ring")
        preset_ring.concept_graph.add_concept(
            name="Workflow",
            vector=preset_ring.input_processor.process_text(initial_ring_content),
            ttl=-1,  # Immortal concept
            metadata={"text": initial_ring_content, "importance": "high"}
        )
        self.auto_save = auto_save
        self.network_file = network_file
        if auto_save and network_file is not None and os.path.exists(network_file):
            self.network = NetworkManager.load_from_file(network_file)
        else:
            self.network = NetworkManager(
                    num_empty_rings=num_empty_rings,
                    preset_rings=[preset_ring]
                )

        if display:
            v = NetworkVisualizer(self.network)
            self.og_processor = self.network.process_input

            def process_with_visualization(text):
                """Process input text and update visualization"""
                self.network.resume()
                results = self.og_processor(text)
                for ring_id in results[::-1]:
                    v.mark_active(ring_id)
                    time.sleep(0.25)

                self.network.freeze()
                return results

            self.network.process_input = lambda text: process_with_visualization(text)

        # Initialize agent state with improved structure
        self.agent_state = AgentState(
            current_focus="initial-ring",
            last_action={
                "type": "initialization",
                "target_ring": "initial-ring",
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

    def _inner_monologue(self, context: str) -> str:
        """Generate inner thoughts based on current context"""
        monologue_prompt = f"""
        Current Context: {context}
        Current Focus Ring: {self.agent_state.current_focus}
        Last Action: {self.agent_state.last_action['details']}
        Active Concepts: {[x['metadata']['text'] if 'text' in x['metadata'] else x['name'] for x in  self._get_active_concepts()]}

        Generate inner reasoning about the current situation and next steps.
        """

        thought = self.agent_llm.process(monologue_prompt, max_tokens=220)
        thought = str(thought)
        self.agent_state.inner_state["monologue"].append({
            "timestamp": time.time(),
            "thought": thought
        })
        return thought

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

    def process_user_input(self, user_input: str) -> str:
        """Process user input through the interface LLM"""
        # Allow for inner inference time
        current_time = time.time()
        if current_time - self.agent_state.inner_state["last_inference"] >= self.inference_time:
            thought = self._inner_monologue(user_input)
            self.agent_state.inner_state["thought_chain"].append(thought)
            self.agent_state.inner_state["last_inference"] = current_time

        if len(user_input) > 200000:
            self.network.process_input(user_input)
        # Get interface LLM's interpretation
        interface_response = self.user_llm.process(
            f"User input: {user_input[:100000]}\nLast thought: {self.agent_state.inner_state['thought_chain'][-1] if self.agent_state.inner_state['thought_chain'] else 'None'}\nProvide clear instruction for agent.",
            max_tokens=200
        )

        interface_response = str(interface_response)

        # Process through network
        with self.lock:
            res = self._process_agent_action(interface_response)

        # Get relevant ring information

        # Generate user-facing response
        final_response = self.agent_llm.process(
            f"Agent result: {res}\n"
            # f"Network Info: {self.network.get_references(user_input, top_k=2, concept_elem='metadata')}\n"
            f"active_concepts: {self._get_active_concepts()}"
            f"Inner thought: {self.agent_state.inner_state['thought_chain'][-1] if self.agent_state.inner_state['thought_chain'] else 'None'}\n"
            f"Generate user-friendly response. based on the user query and provided data: {user_input[:100000]}\n",
            max_tokens=250
        )
        final_response = str(final_response)
        if self.auto_save:
            self.network.save_to_file(self.network_file)
        return final_response

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
        Current focus: {self.agent_state.current_focus}
        Last action: {self.agent_state.last_action['details']}
        Inner thought: {thought}
        Active concepts: {[x['metadata']['text'] if 'text' in x['metadata'] else x['name'] for x in  self._get_active_concepts()]}
        Instruction: {instruction}

        Decide next action:
        1. Process in current ring
        2. Switch focus to another ring
        3. Activate reset ring
        4. Update connections
        """

        action = self.agent_llm.process(action_prompt, max_tokens=50)
        action = str(action)
        # Handle reset ring activation
        if "reset" in action.lower():
            return self._activate_reset_ring()

        # Process action through network
        result = self.network.process_input(instruction)

        # Update agent state with structured action info
        self._update_agent_state(action, result)
        if len(result) == 0:
            return ''
        fall_back_ring = list(self.network.rings.values())[-1]
        simmilad_data = self.network.rings.get(result[0], fall_back_ring).retrieval_system.find_similar(
            self.network.rings.get(result[0], fall_back_ring).input_processor.process_text(thought))
        if len(simmilad_data) == 0:
            return ''
        data = self.network.rings.get(result[0], fall_back_ring).get_concept_by_id(simmilad_data[0][0]).metadata
        if 'text' in data:
            return data['text']
        return str(data)

    def _activate_reset_ring(self) -> str:
        """Handle reset ring activation"""
        self.network.process_input("activate_reset")

        # Reset agent state with structured information
        self.agent_state = AgentState(
            current_focus="initial-ring",
            last_action={
                "type": "reset",
                "target_ring": "initial-ring",
                "timestamp": time.time(),
                "details": "Reset protocol executed"
            },
            connection_strengths={},
            inner_state={
                "monologue": [],
                "last_inference": time.time(),
                "thought_chain": [],
                "attention_focus": None
            }
        )

        return "Reset protocol executed"

    def _update_agent_state(self, action: str, result: str):
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
    Base cognitive framework:
    1. Process user input
    2. Evaluate context
    3. Select appropriate action
    4. Generate response
    """

    # Create cognitive network
    cognitive_net = CognitiveNetwork(
        user_llm=user_interface_llm,
        agent_llm=agent_llm,
        initial_ring_content=initial_content,
        inference_time=0.6,
        num_empty_rings=1,
        auto_save=False,
    )

    # Example interaction
    user_inputs = [
        "Hello, can you help me solve a math problem?",
        "What's 25 times 16?",
        "Can you reset your thinking?",
        "Let's try a different topic"
    ]
    response = cognitive_net.process_user_input("Hi main Name Ist Markin!")
    print(response)

    for user_input in user_inputs:
        print(f"\nUser: {user_input}")
        response = cognitive_net.process_user_input(user_input)
        print(f"System: {response}")
        # Print agent status after each interaction
        cognitive_net.network.show.display_state('metrics')
        cognitive_net.get_agent_status()

    input()


if __name__ == "__main__":

    #network = NetworkManager.load_from_file("network.hex")
    #v = NetworkVisualizer(network)
    #input("DONE?")

    main()
