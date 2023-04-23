import streamlit as st

from toolboxv2 import App
from toolboxv2.mods_dev.isaa import AgentConfig


def run_isaa_verb(app: App, user_text: str, todolist_agent_config: AgentConfig):
    app.logger.info("Isaa is running")

    todolist_agent_config.step_between = "Find The Best Tool To solve the Problem"
    response = app.AC_MOD.run_agent("todolist", user_text, todolist_agent_config)

    return response, todolist_agent_config.task_list

# Streamlit app
st.title('Interactive Todo List Agent')

app = App()
app.inplace_load("isaa", "toolboxv2.mods_dev.")
app.new_ac_mod('isaa')
app.AC_MOD.loade_keys_from_env()
todolist_agent_config = app.AC_MOD.get_agent_config_class("todolist")

user_text = st.text_input("Enter your task:", "Erzeuge mir eine Todo liste zu Ereiche Eines Haus Baus welche schritte müssen dafür beachtet werden?")
response, task_list = run_isaa_verb(app, user_text, todolist_agent_config)

st.write("Response:", response)
st.write("Todo List:")
for task in task_list:
    st.write(task)
