import numpy as np
import streamlit as st
import time
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    layout="wide",
    page_title="Multi-Agent System UI",
    initial_sidebar_state="expanded"
)

# Define color palette
PRIMARY_COLOR = "#FF4B4B"
SECONDARY_COLOR = "#0068C9"
BACKGROUND_COLOR = "#FFFFFF"
SURFACE_COLOR = "#F0F2F6"
TEXT_COLOR = "#262730"
ACCENT_COLOR = "#09AB3B"

# Move configuration to sidebar
st.sidebar.title("Configuration")
view = st.sidebar.radio(
    "Select View",
    ["Overview", "Agents", "Tasks"],
    key="main_view",
    label_visibility="visible"
)

with st.sidebar.expander("Configuration Options"):
    parameter1 = st.slider("Parameter 1", 0, 100, 50)
    option = st.selectbox("Select Option", ["Option 1", "Option 2", "Option 3"])

# Main content area
main_col, graph_col = st.columns([9, 1])
user_input = st.chat_input("Enter your message here:")
with main_col:
    st.markdown(f'<div style="background-color:{BACKGROUND_COLOR}; padding:2rem;">', unsafe_allow_html=True)
    if view == "Overview":
        st.header("Overview Dashboard")
        # Overview content
    elif view == "Agents":
        st.header("Agents Management")
        # Agents content
    else:
        st.header("Tasks Management")
        # Tasks content

    # Chat-like input and processing window
    st.subheader("Chat-like Input")

    show_g = st.toggle("Show Graph")
    if st.button("Send"):
        with st.spinner("Processing..."):
            # Simulate processing time
            time.sleep(2)
            st.success("Processing complete!")
            st.write(f"Received: {user_input}")

    st.subheader("Processing Window")
    with st.expander("Current Operations"):
        st.write("No current operations.")

    st.subheader("Multi-modal Input")
    with st.form("multi_modal_form"):
        text_input = st.text_input("Text Input")
        file_upload = st.file_uploader("Drag and drop files here", type=["txt", "pdf", "png", "jpg"])
        audio_input = st.file_uploader("Or record audio", type=["wav", "mp3"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Text Input:", text_input)
            if file_upload is not None:
                st.write("File Uploaded:", file_upload.name)
            if audio_input is not None:
                st.write("Audio Uploaded:", audio_input.name)

    st.markdown('</div>', unsafe_allow_html=True)

if show_g:
    with graph_col:
        st.subheader("3D Graph")
        # 3D graph visualization
        np.random.seed(42)
        x = np.random.rand(100)
        y = np.random.rand(100)
        z = np.random.rand(100)
        node_labels = [f"Node {i + 1}" for i in range(100)]
        node_details = [f"Details for Node {i + 1}" for i in range(100)]

        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,  # Color based on z-values
                colorscale='Viridis',
                opacity=0.8
            ),
            text=node_labels,  # Hover text
            hoverinfo='text'
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(nticks=4, showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(nticks=4, showbackground=True, backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(nticks=4, showbackground=True, backgroundcolor="rgb(230, 230,230)")
            ),
            margin=dict(r=0, l=0, b=0, t=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Node selection dropdown
        selected_node = st.selectbox("Select Node", node_labels, label_visibility="visible")
        node_index = node_labels.index(selected_node)
        st.write(node_details[node_index])
        st.markdown('</div>', unsafe_allow_html=True)
