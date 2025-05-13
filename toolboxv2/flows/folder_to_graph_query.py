import os.path

from toolboxv2 import App

NAME = "folderTGQ"


def path_getter(path):
    from toolboxv2.mods.isaa.subtools.file_loder import route_local_file_to_function
    loder, docs_loder = route_local_file_to_function(path)
    docs = docs_loder()
    return [doc.page_content for doc in docs], [doc.metadata if doc.metadata else {'doc-from': path} for doc in docs],

async def run(app:App, __, mem_name=None, path=None):
    from toolboxv2.mods.isaa.base.AgentUtils import AISemanticMemory
    from toolboxv2.mods.isaa.base.KnowledgeBase import KnowledgeBase
    from toolboxv2.mods.isaa.isaa_modi import get_multiline_input

    if mem_name is None:
        app.print("No mem_name arg")
        return

    isaa = app.get_mod("isaa", spec=NAME)
    agent = isaa.init_isaa(build=True)
    memory : AISemanticMemory = isaa.get_memory()
    mem_instance: KnowledgeBase | None = None

    if path is not None:
        if not os.path.exists(path):
            app.print("No path found")
            return
        try:
            mem_instance = memory.create_memory(mem_name)
        except ValueError:
            mem_instance = isaa.get_memory(mem_name)

        app.print("Loading Docs")
        docs, metadata = path_getter(path)
        print("Loading",docs)
        print("Loading",metadata)
        await mem_instance.add_data(docs, metadata)
        mem_instance.save(mem_name + '_save.pkl')
        mem_instance.vis(output_file=mem_name+'_graph.html')
    else:
        mem_instance = isaa.get_memory(mem_name)
        mem_instance.load(mem_name+'_save.pkl')

    while user_input := get_multiline_input("USER:"):
        if user_input.startswith('#'):
            user_input = user_input[1:]
            context = mem_instance.query_concepts(user_input)
        else:
            context = mem_instance.retrieve_with_overview(user_input)
        print("Context:", context)
        result = agent.mini_task(user_input, "user", f"Use this context! {context}", persist=True)
        print("")
        print("AGENT:", result)


if __name__ == "__main__":
    import asyncio

    from toolboxv2 import get_app
    asyncio.run(run(get_app(), None, "uni_lina", r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\data\uniAnna"))
