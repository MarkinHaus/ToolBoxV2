import os
from platform import system

import requests
import streamlit as st

from toolboxv2 import App
import sys
from streamlit_option_menu import option_menu

global app_tb_dv


def edit_config_entry(c, index):
    try:
        key, value = c[0], eval(c[1]) if isinstance(c[1], str) else c[1]
    except TypeError:
        st.warning("Invalid configuration Data")
        st.warning(c)
        return
    except SyntaxError:
        st.info("value pased as string")
        key, value = c[0], c[1]

    skey = key.replace('~', '')

    with st.form(f"addit{key}"):
        st.subheader(skey)
        # Check the type of the value
        if isinstance(value, str):
            # Show a text input for editing the string value
            edited_value = st.text_input(f"Edit {index}:", value, key=f"str#{skey}#{key}")
        elif isinstance(value, bool):
            # Show a checkbox for editing the boolean value
            edited_value = st.checkbox(f"Edit {index}:", value, key=f"str#{skey}#{key}")
        elif isinstance(value, dict):
            # Show a checkbox for editing the boolean value
            edited_value = st.text_area(f"Edit {index}:", value, key=f"str#{skey}#{key}")
        elif isinstance(value, int):
            # Show a number input for editing the integer value
            edited_value = st.number_input(f"Edit {index}:", value, key=f"str#{skey}#{key}")
        elif isinstance(value, list):
            # Show a multi-select box for editing the list value
            if len(value) == 1:
                edited_value = ['']
                edited_value[0] = st.text_input(f"Edit {index}:", value[0], key=f"str#{skey}#{key}")
            else:
                edited_value = st.text_area(f"Edit {index}:", value, key=f"str#{skey}#{key}")
        else:
            st.write(value)
            edited_value = ""

        # Save button
        if st.form_submit_button("Save"):
            with st.spinner("Save data local () pleas do NOT forget to Save Data left <-"):

                if key == "TB-DEV":
                    dev_data = {
                        "helper~~~:": {}
                        , "macro~~~~:": ['mainTool', 'cloudM', 'welcome']
                        , "m_color~~:": {}
                        , "debug~~~~:": True
                        , "mute~load:": False
                        , "name-spa~:": ['']
                        , "load~mode:": ['I']
                    }
                    app_tb_dv.AC_MOD.open_l_file_handler()
                    app_tb_dv.AC_MOD.open_s_file_handler()
                    for _key, _value in dev_data.items():
                        app_tb_dv.AC_MOD.add_to_save_file_handler(_key, str(_value))
                    st.info("Init - Dev")
                    return

                # Update the config entry with the edited value
                config_t[index] = [key, edited_value]
                if conf_name == "mainTool":
                    app_tb_dv.config_fh.add_to_save_file_handler(str(key), str(edited_value))
                else:
                    app_tb_dv.AC_MOD.add_to_save_file_handler(str(key), str(edited_value))


st.title("☁")
st.title("Dev Tools")

if "login" not in st.session_state:

    with st.form(f"token-form"):
        placeholder = st.empty()
        em = placeholder.container()

        em.warning("Pleas enter Token to Continue if you dont have one jet go to https://simpel.com")
        em.caption("dev-tool-web-streamlit Welcomes you 😄")
        token = em.text_input("Token", type="password")
        tbname = em.text_input("Name der app", value="dev")
        em.caption("Pleas dont char ur Token🔒")
        if em.form_submit_button("Submit-Token"):
            with st.spinner("Verifying Token..."):
                print("Name: ", __name__)
                app_tb_dv = App(tbname + "-")
                st.session_state.app = app_tb_dv
                app_tb_dv.load_mod("cloudM")
                app_tb_dv.new_ac_mod("cloudM")
                # res = app_tb_dv.AC_MOD.validate_jwt([{'token': token, "data": {
                #     "passdb": True,
                #     "max-p": 2
                # }}], app_tb_dv)
                # print(res)
                # st.info(res)
                # log_in([0, username, password])
                st.session_state.login = {'auth': True, 'token': '0'}
                em.write("Login successful, ples click on dev tool now to kontionue")
        else:
            st.stop()

        login = st.session_state.login
        em.empty()

if "app" not in st.session_state:
    config_file = "dev.config"
    id_name = ""
    print("Name: ", __name__)
    for i in sys.argv[2:]:
        if i.startswith('data'):
            d = i.split(':')
            config_file = d[1]
            id_name = d[2]
    app_tb_dv = App("dev-")
    st.warning("ato go back to dev tools")
    st.session_state.app = app_tb_dv

app_tb_dv = st.session_state.app

st.write(app_tb_dv.id)

if "acmod" not in st.session_state:
    st.session_state.acmod = "mainTool"
    app_tb_dv.MACRO.append("mainTool")
    # app_tb_dv.save_load("welcome")
    # app_tb_dv.new_ac_mod("welcome")

ac_mod = st.session_state.acmod

tabs_titels = ["reste - mod", "Run mod - functon ", "config mod"]

tabs = st.tabs(tabs_titels)

with tabs[0]:
    with st.sidebar:
        # 3. CSS style definitions
        selected3 = option_menu("Mod-list", app_tb_dv.MACRO[::-1],
                                icons=app_tb_dv.MACRO,
                                menu_icon="cast", default_index=0,
                                # styles={
                                #    "container": {"padding": "0!important", "background-color": "#fafafa"},
                                #    "icon": {"color": "orange", "font-size": "25px"},
                                #    "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                                #                 "--hover-color": "#eee"},
                                #    "nav-link-selected": {"background-color": "green"},
                                # }
                                )

    filename = st.text_input("Mod")

    if st.button(f"loading mod: {filename if filename else selected3}"):
        if ac_mod != "mainTool":
            app_tb_dv.reset()
            if app_tb_dv.AC_MOD:
                app_tb_dv.remove_mod(ac_mod)
        st.info(f"loading mod: {filename if filename else selected3}")
        if filename if filename else selected3 != "mainTool":
            mod = app_tb_dv.save_load(filename if filename else selected3)
            st.write(str(mod))
            app_tb_dv.new_ac_mod(filename if filename else selected3)
            st.session_state.acmod = filename if filename else selected3
# {"token": 0,"data": {'username':'kinr3','password':'root'}}
with tabs[1]:
    with st.sidebar:
        # 3. CSS style definitions
        st.write("Mod all mod functions")
        st.write(app_tb_dv.SUPER_SET)
        if "disabled" not in st.session_state:
            st.session_state.private = True
            st.session_state.disabled = False
            st.session_state.options = True  # get True : post
            st.session_state.server_online = True  # local True : online
            st.session_state.auto_c = False

        col1, col2 = st.columns(2)

        with col1:
            st.checkbox("Cloud ☁ request?", key="disabled")
            st.checkbox("Private", key="private")
            st.checkbox("Use auto Completion ~", key="auto_c")

        with col2:
            st.checkbox(f"{'Online' if st.session_state.server_online else 'Local'}", key="server_online",
                        disabled=not st.session_state.disabled, )
            st.checkbox(f"{'post' if st.session_state.options else 'get'}", key="options",
                        disabled=not st.session_state.disabled, )

    st.subheader(ac_mod)
    name = st.text_input("name")
    command = st.text_input("command")

    options_n = list(set(app_tb_dv.autocompletion(name)))
    options_c = list(set(app_tb_dv.autocompletion(command)))

    st.caption(f"running : ~ name : {options_n}  command {options_c}")

    if st.session_state.disabled:
        v = str(f'"token": {st.session_state.login["token"]},"data": {{}}')
        data = st.text_input("data", value='{' + v + '}')

    if st.button(f"run function {name}"):
        with st.spinner("Running function"):
            if st.session_state.auto_c:
                if len(options_n) > 0:
                    name = options_n[0]

            if st.session_state.disabled:  # socket - request
                url = f"http://194.233.168.22:5000/"
                if not st.session_state.server_online:
                    url = f"http://localhost:5000/"

                if st.session_state.options:
                    url += f"post/{ac_mod}/run/{name}?command={command}"
                else:
                    url += f"get/{ac_mod}/run/{name}?command={command}"

                st.info(f"get from {url=}")
                r = requests.post(url, json=eval(data))
                if r.status_code != 200:
                    st.warning(f"an error occurred: {r.text} : {r.status_code} : {r.headers}")
                else:
                    st.write(f"x-process-time: {float(r.headers['x-process-time']):.4f}")
                    st.write(r.json())
            else:
                ret = app_tb_dv.run_function(name, command.split(".#"))
                st.write(ret)

with tabs[2]:
    st.write("Mod config data")

    config_t = []
    conf_name = ""

    if ac_mod == "mainTool":
        conf_name = app_tb_dv.config_fh.file_handler_filename
        st.write(conf_name)
        config_t = app_tb_dv.config_fh.file_handler_load

    elif not app_tb_dv.get_file_handler_name():
        st.write("mod has no file handler")
        conf_name = "None"
        st.stop()
    else:
        conf_name = app_tb_dv.get_file_handler_name()
        st.write("File name : " + str(conf_name))
        config_t = app_tb_dv.AC_MOD.file_handler_load

    list_conf = [c[0].replace('~', '') for c in config_t]

    if len(list_conf) == 0:
        list_conf = ["no configuration", "Default TB-CLI configuration", "Default TB-API configuration",
                     "Default TB-DEV configuration"]
        st.write("No configuration specified")
        config_t = [["0", 0], ["TB-CLI", "version: 0"], ["TB-API", "version: 0"], ["TB-DEV", "version: 0"]]

    with st.sidebar:
        # 3. CSS style definitions
        selected = option_menu("Config-Data-list", list_conf,
                               icons=list_conf,
                               menu_icon="cast", default_index=0,
                               # styles={
                               #    "container": {"padding": "0!important", "background-color": "#fafafa"},
                               #    "icon": {"color": "orange", "font-size": "25px"},
                               #    "nav-link": {"font-size": "25px", "text-align": "left", "margin": "0px",
                               #                 "--hover-color": "#eee"},
                               #    "nav-link-selected": {"background-color": "green"},
                               # }
                               )
        with st.spinner("Save data to file"):
            st.caption("Save Config-data:")
            st.caption(f"filename: {conf_name}")
            if st.button(f"Save {ac_mod}"):
                if ac_mod.upper() == "mainTool".upper():
                    # app_tb_dv.config_fh.open_s_file_handler()
                    # app_tb_dv.config_fh.save_file_handler()
                    app_tb_dv.save_exit()
                    app_tb_dv.exit_all_modules()
                    app_tb_dv.exit()
                    st.experimental_rerun()
                else:
                    app_tb_dv.AC_MOD.open_s_file_handler()
                    app_tb_dv.AC_MOD.save_file_handler()
                    app_tb_dv.AC_MOD.file_handler_storage.close()
                    app_tb_dv.remove_mod(ac_mod)

    config = config_t[list_conf.index(selected)]
    edit_config_entry(config, list_conf.index(selected))


# eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJ2IjoiMC4yLjMiLCJleHAiOjE2ODY4MTczODgsInVzZXJuYW1lIjoia2lucjMiLCJwYXNzd29yZCI6Ijc4YjNiOGVjMDVjMjY4YzY4NjdlMjI1OGI3YzY3NzhjYjFlYzczYjVlZDJlNjNhZDdkZmFlMGFiMWRmNzFjOTZkYzA5OTVkNTQxN2VlNzJhZTA0ZTMxYmU0ZDcxNTA2NTE4ZWQ5N2M4NWU4NTkwYTE2YjM2MTU5M2VmYzc1YzE3MTdhYmI2YmRjNWZmNzA2ZDc4OTIzNGY4NTlmZDQ2NTEwNjkxMzk3ZDA3Y2IwMTE0YTY0NTIwMmI0MDQwYTIzYSIsImVtYWlsIjoibWFya2luaGF1c21hbm5zQGdtYWlsLmNvbSIsInVpZCI6IjkwZDI3MjEwLTFhOTQtNGUxMC1hZjRjLTZlYjk4ZDRmNzM0YyIsImF1ZCI6ImFwaS1sb2NhbGhvc3QifQ.dd6kgm9E7f4boy5zxjOP25xmD1ej35-XWaUC2aUyJxmgderm024VHX6zmRo8aJt2q4nrqBaJf4ogqRmUR7CJlg
