def api_keys(app):
    app.AC_MOD.add_str_to_config(["", 'WOLFRAM_ALPHA_APPID', "-"])
    app.AC_MOD.add_str_to_config(["", 'HUGGINGFACEHUB_API_TOKEN', ""])
    app.AC_MOD.add_str_to_config(["", 'OPENAI_API_KEY', ""])
    app.AC_MOD.add_str_to_config(["", 'REPLICATE_API_TOKEN', ""])
    app.AC_MOD.add_str_to_config(["", 'IFTTTKey', "-MZ4yF-"])
    app.AC_MOD.add_str_to_config(["", 'SERP_API_KEY', ""])


def pinecone_keys(app):
    app.AC_MOD.config["PINECONE_API_KEY"] = '----'
    app.AC_MOD.config["PINECONE_API_ENV"] = '--'
