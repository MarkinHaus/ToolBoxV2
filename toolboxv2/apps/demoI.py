import streamlit as st
import threading
from toolboxv2 import get_app

class ToolsUI:
    def __init__(self, tools_instance):
        self.tools = tools_instance
        self.setup_ui()

    def setup_ui(self):
        st.title("ISAA Multi-Model Agent Interface")
        st.sidebar.title("Navigation and Controls")

        # Sidebar Navigation
        app_mode = st.sidebar.selectbox("Choose the app mode", [
            "Home", "Debug Metrics", "Network/Graph", "Current Action", "Agent Management", "Run Agent", "Cognitive Network"
        ])

        # Model Selection
        available_models = ["Model A", "Model B", "Model C"]  # Replace with actual model names
        selected_model = st.sidebar.selectbox("Select Model", available_models)

        # Agent Management Controls
        agent_controls = st.sidebar.expander("Agent Controls")
        if agent_controls.button("Start Agent"):
            self.start_agent(selected_model)
        if agent_controls.button("Stop Agent"):
            self.stop_agent()

        # App Mode Selection
        if app_mode == "Home":
            self.show_home()
        elif app_mode == "Debug Metrics":
            self.show_debug_metrics()
        elif app_mode == "Network/Graph":
            self.show_network_graph()
        elif app_mode == "Current Action":
            self.show_current_action()
        elif app_mode == "Run Agent":
            self.show_run_agent()
        elif app_mode == "Cognitive Network":
            self.show_cognitive_network()

    def start_agent(self, model_name):
        # Initialize and start the agent in a separate thread
        # Example:
        # self.agent_thread = threading.Thread(target=self.tools.start_agent, args=(model_name,))
        # self.agent_thread.start()
        pass  # Implement agent start logic here

    def stop_agent(self):
        # Stop the running agent
        # Example:
        # self.tools.stop_agent()
        pass  # Implement agent stop logic here

    def show_home(self):
        st.header("Welcome to ISAA Multi-Model Agent Interface")
        st.write("""
            This interface allows you to manage multiple models and agents,
            monitor their status, and view debugging information.
        """)

    def show_debug_metrics(self):
        st.header("Debug Metrics")

        # Display Rings Metrics
        st.subheader("Rings Metrics")
        rings_metrics = self.tools.get_context_memory().get_metrics()
        st.write("Rings Metrics:")
        st.json(rings_metrics)

        # Display Network Metrics
        st.subheader("Network Metrics")
        network_metrics = self.tools.get_context_memory().cognitive_network.network.get_metrics()
        st.write("Network Metrics:")
        st.json(network_metrics)

    def show_network_graph(self):
        st.header("Network/Graph Visualization")

        # Display Network Graph
        st.subheader("Network Graph")
        network_graph = self.tools.get_context_memory().cognitive_network.network
        st.write("Network Graph Visualization:")
        # Here you can use a library like NetworkX or similar to visualize the graph
        # For simplicity, we'll just display the connections
        connections = network_graph.connections
        st.write("Connections:")
        st.json(connections)

    def show_current_action(self):
        st.header("Current Action Point")

        # Display Current Action
        st.subheader("Current Action")
        current_action = self.tools.get_context_memory().cognitive_network.agent_state.last_action
        st.write("Current Action:")
        st.json(current_action)

        # Display Inner State
        st.subheader("Inner State")
        inner_state = self.tools.get_context_memory().cognitive_network.agent_state.inner_state
        st.write("Inner State:")
        st.json(inner_state)

    def show_run_agent(self):
        st.header("Run Agent")

        # Input for agent name
        agent_name = st.text_input("Enter Agent Name", "self")

        # Input for task
        task = st.text_area("Enter Task", "Please provide a task for the agent.")

        # Button to run the agent
        if st.button("Run Agent"):
            with st.spinner("Running agent..."):
                result = self.tools.run_agent(agent_name, task)
                st.write("Result:")
                st.json(result)

    def show_cognitive_network(self):
        st.header("Cognitive Network")

        # Input for cognitive network task
        task = st.text_area("Enter Cognitive Network Task", "Please provide a task for the cognitive network.")

        # Button to run the cognitive network task
        if st.button("Run Cognitive Network Task"):
            with st.spinner("Running cognitive network task..."):
                result = self.tools.run(task)  # Assuming `run` is the method to run cognitive network tasks
                st.write("Result:")
                st.json(result)


if __name__ == "__main__":
    # Initialize the tools instance
    from toolboxv2 import get_app
    tools_instance = get_app().get_mod("isaa")
    tools_instance.init_isaa(build=True)
    ui = ToolsUI(tools_instance)
