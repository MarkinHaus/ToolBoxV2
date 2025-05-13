import json
from asyncio import Future
from concurrent.futures import ThreadPoolExecutor
from logging import Logger
from urllib.parse import urlparse

from tqdm import tqdm

from toolboxv2 import Spinner, get_app, get_logger
from toolboxv2.mods.CloudM.ModManager import download_files
from toolboxv2.mods.isaa import Tools
from toolboxv2.mods.isaa.extras.modes import CreatePrompt, DivideMode, TextExtractor
from toolboxv2.mods.isaa.subtools.file_loder import load_from_file_system
from toolboxv2.mods.isaa.subtools.web_loder import read_git_repo, route_url_to_function
from toolboxv2.utils.extras.Style import print_prompt

Name = 'isaa.code'
version = "0.0.2"
export = get_app(from_=f"{Name}.module.EXPORT").tb
test_only = export(mod_name=Name, test_only=True, version=version, state=True)


def parse_code(source, isaa: Tools, space_name) -> str:
    """ start analyse, write docs and save to vector store mit meat daten analysed-code """

    mem = isaa.get_memory()
    mem_name = space_name + 'Code'
    docs = mem.split_text(mem_name, source, separators='py')
    results = []
    with tqdm(total=len(docs), unit='steps', desc='parsing input') as pbar:
        for code_prat in docs:
            result = isaa.mini_task_completion_format("Your Task is to analyse the given code in terms of:"
                                                      "\n- Exportable and reusable components,"
                                                      "\n- Internal structure,"
                                                      "\n- External sources."
                                                      "Provide your findings in an structured markdown format,"
                                                      " highlighting "
                                                      "key observations and any recommendations for"
                                                      "improvement."
                                                      f" source-code : {code_prat}"
                                                      ,
                                                      "Expected format markdown format")
            pbar.update()
            results.append(result)

    mem.add_data(mem_name, results)
    return mem_name


def analyze_usage_from_markdown(documentation: str, isaa: Tools, space_name: str) -> str:
    """
    Analyzes Markdown documentation to extract information about how to use the described codebase,
    enabling a bot to implement and understand concrete aspects of the documentation and codebase.

    Parameters:
    - documentation (str): The Markdown documentation to be analyzed.
    - isaa (Tools): A toolset for performing operations like context memory management and task completions.
    - space_name (str): The name of the space where the analysis results will be stored.

    Returns:
    - None
    """
    mem = isaa.get_memory()
    mem_name = space_name + 'Docs'
    # Split the documentation into sections for detailed analysis
    documentation_sections = mem.split_text(mem_name, documentation, separators='md')

    # Prepare to store the results of the documentation analysis
    usage_information_results = []

    # Initialize a progress bar for tracking the documentation analysis progress
    with tqdm(total=len(documentation_sections), unit='sections', desc='Extracting Usage Information') as progress_bar:
        for section in documentation_sections:
            # Define the task for extracting usage information
            analysis_task = (
                "Your task is to analyze the given documentation section to extract detailed information on:\n"
                "- How to set up the environment or dependencies,\n"
                "- Step-by-step usage instructions,\n"
                "- Code examples and their explanations,\n"
                "- API descriptions and how to interact with them,\n"
                "- Any configuration or customization options.\n\n"
                "Summarize the extracted information in a structured format that a bot can later use to "
                "understand and implement aspects of the codebase.\n\n"
                f"Documentation Section:\n{section}\n"
            )

            # Perform the analysis using the provided tools
            analysis_result = isaa.mini_task_completion_format(
                mini_task=analysis_task,
                format_="Expected format markdown format"
            )

            # Update the progress bar
            progress_bar.update()

            # Append the analysis result to the list of results
            usage_information_results.append(analysis_result)

    # Store the analysis results in the context memory
    mem.add_data(mem_name, usage_information_results)
    return mem_name


def parsing_inputs(isaa, request: str or list, existing_code_base: str, local_sources: list,
                   remote_sources: list, space_name: str, working_directory: str):
    total = 4
    if local_sources is not None:
        total += len(local_sources)
    if remote_sources is not None:
        total += len(remote_sources)
    with tqdm(total=total, unit='steps', desc='parsing input') as pbar:
        result = isaa.mini_task_completion_format(f"Your Task is to evaluate the complexity of the request."
                                                  f"first entry is if docs ar needed "
                                                  f"second entry is if examples ar needed "
                                                  f"and the last entry is a complexity measurement between 0 and 1,"
                                                  f" 0 means no actions needed 1 means its seams impossible. request : {request}. No extra Characters!"
                                                  ,
                                                  {'docs': {'type': 'bool'}, 'examples': {'type': 'bool'},
                                                   'complexity': {'type': 'float'}})
        result = result.strip() #.replace("f", "F").replace("t", "T")
        print(result)
        result_list = list(json.loads(result).values())
        print(result_list)
        try:
            result_list = eval(result)
        except ValueError:
            return "Expected format cond not parse agent interpretation"
        if len(result_list) != 3:
            return f"Expected format invalid agent interpretation {result}"
        pbar.update(1)
        pbar.write("Parsing existing_code_base")
        is_code = False
        is_url = False
        is_file = False
        is_folder = False
        if existing_code_base is not None:
            urlparse_existing_code_base = urlparse(existing_code_base)
            is_code = existing_code_base.startswith('```')
            is_url = urlparse_existing_code_base.scheme.startswith('http')

            if not is_code and not is_url and len(urlparse_existing_code_base.scheme) < 3:
                is_file = '.' in urlparse_existing_code_base.path
                is_folder = not is_file

            code_data = ""
            docs_data = ""

            if is_code:
                code_data = existing_code_base
            if is_url:
                existing_code_base = download_files(existing_code_base,
                                                    "\\external_source",
                                                    "Pulling content from url",
                                                    print)
                is_file = True
            if is_file:
                with open(existing_code_base) as f:
                    content = f.read()
                if urlparse_existing_code_base.path.endswith('.py'):
                    code_data = content
                if urlparse_existing_code_base.path.endswith('.md'):
                    docs_data = content
            if is_folder:
                data = load_from_file_system(existing_code_base)
                code_data = "\n\n".join([x.page_content for x in data])
            if code_data != '':
                parse_code(code_data, isaa, space_name)
            if docs_data != '':
                analyze_usage_from_markdown(docs_data, isaa, space_name)
        pbar.update(1)
        pbar.write("Parsing local_sources")
        if local_sources is not None:
            for source in local_sources:
                if source[-3] == '.' or source.endswith('.py') or source.endswith('.html'):
                    with open(source) as f:
                        content = f.read()
                    if source.endswith('.md'):
                        analyze_usage_from_markdown(content, isaa, space_name)
                    else:
                        parse_code(content, isaa, space_name)
                else:
                    data = load_from_file_system(source)[1]()
                    parse_code("\n\n".join([x.page_content for x in data]), isaa, space_name)
                pbar.update(1)
        pbar.update(1)
        pbar.write("Parsing remote_sources")
        if remote_sources is not None:
            loders = []
            for source in remote_sources:
                if 'github' in source.lower():
                    read_git_repo(working_directory, source)
                    data = load_from_file_system(working_directory + "git")
                    parse_code("\n\n".join([x.page_content for x in data]), isaa, space_name)
                else:
                    loder, docs = route_url_to_function(source)
                    loders.append(loder)

                pbar.update(1)
        pbar.update(1)

    return {"request": request, "result_list": result_list}, [space_name + 'Code', space_name + 'Docs']


def code_writer_agent_loop(isaa: Tools, task: str, memspaces: list[str], max_iterations: int = 6, v_code_base=r""):
    """
    needed functions mini agent ide, interactive umgebung

    """

    don = False
    iteration = 0
    functions = []

    coder_agent_builder = isaa.get_default_agent_builder("code")
    coder_agent_builder.set_verbose(True).set_functions(functions=functions)  # .set_tasklist(tasks).set_task_index(0)
    coder_agent = isaa.register_agent(coder_agent_builder)

    while iteration < max_iterations and not don:
        isaa.run_agent(coder_agent, task, max_iterations, running_mode="oncex", persist=True)


def runner(request: str or list,
           existing_code_base: str = None,
           local_sources: list = None,
           remote_sources: list = None,
           save_changes: bool = False,
           crate_new_files: bool = False,
           working_directory: str = None):
    """
    Fuction to use isaa and agents to generate production redy code

    Abaluf paln:
        - aufberitung der input parameter
            - parsing von request
                - herausfinden von benötigten informationen -> list [docs:bool, examples:bool]
            - parsing von existing_code_base
                - wenn None -> do nothing
                - wenn code,docs -> start analyse, write docs and save to vector store mit meat daten analysed-code or analysed-docs
                - wenn path ->
                    - finde docs folder + code and save to vector store mit meat daten row-docs or row-code
                    - wenn kein docs folder gefunden wurde save code and to vector store mit meat daten row-code
                - wenn url download to local_sources.path and ->
                    - gist = start analyse, write docs and save to vector store mit meat daten analysed-code
                    - git = *- wenn path
                    - jupiter-notebook = *- gist
            - parsing von local_sources ->
                - wenn code = *- wenn path ->
                - wenn text = use text_parser *- wenn code,docs
                - wenn md = use md_parser *- wenn code,docs
                - wenn pdf = use pdf_parser *- wenn code,docs
            - parsing von remote_sources download to local_sources.path and *- parsing von local_sources ->
        - entgegenehemn der input im format :
            request:dict = {"request":str, "docs":bool, "examples":bool}
            sources:list = str<VectorStorName>,  str<VectorStorName>
        - divide | request in subsequenzen aufbereiten
            - use isaa and llmMode divideMode to dived request["request"] into a json dict
            - pars json dict wit anything_from_str_to_dict(data: str, expected_keys: dict = None)
        - serialise | request dict into
            - dict format keys form 0 to n
            - items from keys is a list of task that can be execute in parallel.
            - a item is just a string withs representatives a part of the divided request.
        - entgegenehemn der input im format :
            request_dict:dict = {"0":["change thing one", "refactor the man page"], 1":["change thing tow"]"...}
        - information collection phase
            - if request["docs"] or request["examples"]
                - for request_dict.phases[i] test if docs needed or examples
                    - yes -> search in vector base docs_base or code_base and use gather information + request_dict.phases[i] withe
                     llm to provide problem specific documentation (explain + exact informations)
                     build concrete prompt seam request_dict[f'{i}{k}_prompt'] = {request_dict.phases[i][k]}{'additional informations'}
                    - no -> request_dict[f'{i}{k}_prompt'] ={request_dict.phases[i][k]}

        - main phase
            - start len(request_dict[i]) instances of parallel worker instances
                -> worker instances
                 - write code snippet according to request_dict[f'{i}{k}_prompt']
                  - test if compiles
                    - re write in format ````[language]\n[comment-prefix] [file-path] [content]```
                  - save code snippet to divide_code

                --> final worker re wirte code snippet and run integration test combine the snippets up to the final version
                 - test version run cond bas test and code test format : ````[language]\n[comment-prefix] [file-path]:[lines start-end, obtonal] [content]```
        - if crate_new_files
            - write new content to files
        - if save_changes
            - write new content to lines
        return new code or cahnsed and eddetet file pathes



    Args:
        request (str or list): in str for nl changes or rove implementation plans, or list for multibyte actions
        existing_code_base (str, optional): the code in an ````[language]\n[comment-prefix] [file-path]``` format
                                            or the path or url to the repository
        local_sources (list[str], optional): local sources code-, text-, md-, and pdf-files
        remote_sources (list[str], optional): remote sources gist-,GitHub-, pdf-url, and documentation urls
        save_changes (bool, optional): default False
        crate_new_files (bool, optional): default False
        working_directory (str, None): path
    Returns:
        new_code (str): new code or path withe changes
    """
    app = get_app(from_=f"{Name}.runner")
    isaa: Tools = app.get_mod('isaa')
    isaa.register_agents_setter(lambda x: x.set_logging_callback(print_prompt))
    # isaa.global_stream_override = True
    logger: Logger = get_logger()

    space_name = "testSp"
    if working_directory is None:
        working_directory = '.\\'

    logger.info("Start parsing input Parameters")
    request_dict, sources_list = parsing_inputs(isaa=isaa,
                                                request=request,
                                                existing_code_base=existing_code_base,
                                                local_sources=local_sources,
                                                remote_sources=remote_sources,
                                                space_name=space_name,
                                                working_directory=working_directory)

    isaa.print(request_dict["result_list"])  #

    if request_dict["result_list"][2] < 0.3:
        pass

    mem = isaa.get_memory()
    with tqdm(total=1, unit='step', desc='divide request') as pbar:
        imple_plan = isaa.mini_task_completion(request_dict['request'], isaa.controller.rget(DivideMode))
        pbar.update()
    imple_request_list = imple_plan.split('\n\n')
    llm_prompts = []
    with tqdm(total=len(imple_request_list), unit='prompt', desc='generating prompts') as pbar:
        def helper(task):
            if not task or len(task) < 10:
                return
            print("Task:", task)
            prompt_request = "Task: " + str(task)
            # prompt_request += "\n full pan: " + imple_plan
            infos = []
            if request_dict["result_list"][0]:
                infos += mem.search(sources_list[1], task)
            if request_dict["result_list"][1]:
                infos += mem.search(sources_list[0], task)
            if len(infos) > 0:
                prompt_request += "Additional informations :" + isaa.mini_task_completion(
                    f"Collect informations for {task} information: " + '\n'.join(
                        [i[0].page_content if isinstance(i, tuple) else i.page_content for i in infos]),
                    mode=TextExtractor,
                    fetch_memory=True, all_mem=False)

            return isaa.mini_task_completion(prompt_request, isaa.controller.rget(CreatePrompt))

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures: set[Future] = {executor.submit(helper, _task) for _task in imple_request_list if _task}
            for futures_ in futures:
                llm_prompts.append(futures_.result())
                pbar.update(1)
    print(llm_prompts)
    with Spinner(symbols='t', message='writing code snippets'):
        sup_sulotns = isaa.mini_task_completion_format(llm_prompts,
                                                       '```[language]\n[comment-prefix] [file-path]:[lines start-end, '
                                                       'obtonal] [content]```',
                                                       None,
                                                       "code")
    print(sup_sulotns)

    mem.add_data("Coding" + space_name, sup_sulotns)

    """
        - main phase
            - start len(request_dict[i]) instances of parallel worker instances
                -> worker instances
                 - write code snippet according to request_dict[f'{i}{k}_prompt']
                  - test if compiles
                    - re write in format ```[language]\n[comment-prefix] [file-path] [content]```
                  - save code snippet to divide_code

                --> final worker re wirte code snippet and run integration test combine the snippets up to the final version
                 - test version run cond bas test and code test format : ````[language]\n[comment-prefix] [file-path]:[lines start-end, obtonal] [content]```
        - if crate_new_files
            - write new content to files
        - if save_changes
            - write new content to lines
        return new code or cahnsed and eddetet file pathes
    """
    # dsd
    return sup_sulotns



if __name__ == "__main__":
    app = get_app(from_=f"{Name}.runner", name="test-runner")
    isaa: Tools = app.get_mod('isaa')
    # logger: Logger = get_logger()
    request_1 = """First I have a daemon (python script) which is always running in the background. now I want to create a class which should imitate this daemon, i.e. when I start the class on my system for the first time, a domain should open which has all the values and attributes of this class. if i now call this class a second time in a different program, i should not get the class but an interface to the class. all calls to the class and intarcitons are made via the daemonen satat but are native to me."""
    # Expected response format: [False, False, 0.1]
    request_2 = "Implement an OAuth2 authentication flow in a Flask application."
    # Expected response format: [True, True, 0.5]
    request_3 = "Optimize a SQL query to reduce execution time for a large dataset."
    # Expected response format: [True, False, 0.4]
    request_4 = "Develop a machine learning model to predict stock prices based on historical data."
    # Expected response format: [True, True, 0.8]
    request_5 = "Design a simple HTML page with a header, footer, and a main content area."
    # Expected response format: [False, False, 0.2]
    requests = [
        request_1,
        # request_2,
        # request_3,
        # request_4,
        # request_5,
    ]
    results = []
    #for request_ in requests:
    result = runner(
        "The fuction logic from the TimedFinancemanager is redundet and wrong get the idee and generate a v2",
        local_sources=[r"C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\mods\FinacWidget.py"]
    )
    print(f"Request \nResult ============={result}=================")
    print(f"Request {results}\n")
