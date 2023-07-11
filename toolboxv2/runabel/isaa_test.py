"""Console script for toolboxv2. Isaa CMD Tool"""
from langchain.agents import load_tools, get_all_tool_names

from toolboxv2 import Style, get_logger
from toolboxv2.mods.isaa import IsaaQuestionBinaryTree, AgentConfig
from toolboxv2.mods.isaa_extars.AgentUtils import Task
from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa, split_todo_list, generate_exi_dict, \
    run_chain_in_cmd, idea_enhancer

NAME = "isaa-test"


def run(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, ide=False, create=False,
                                                python_test=False, init_mem=True, init_pipe=True,
                                                join_now=False, chain_runner=True)

    isaa.global_stream_override = True

    if 'augment' in isaa.config.keys() and False:
        print("init augment")
        isaa.init_from_augment(isaa.config['augment'], exclude=['messages_sto'])
    else:

        tools = {
            "lagChinTools": ['python_repl', 'requests_all',
                             'terminal', 'sleep', 'google-search', 'ddg-search', 'wikipedia',
                             'llm-math',
                             ],
            "huggingTools": [],
            "Plugins": ["https://nla.zapier.com/.well-known/ai-plugin.json"],
            "Custom": [],
        }
        isaa.init_tools(self_agent_config, tools)

    planing_steps = {
        "name": "First-Analysis",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Analyse diese Subjects '''$user-input''',"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Beantworte nach bestem wissen und Gewissen die Die Aufgabe $task. Wenn die aufgebe nicht "
                        "direct von dir gelöst werden kann spezifiziere die nächste schritte die eingeleitet"
                        " werden müssen um die Aufgabe zu bewerkstelligen es beste die option zwischen"
                        "der nutzung von tools und agents.",
                "return": "$0final",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input task= $task out= $0final",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "mini_task",
                "args": "Bestimme ob die aufgebe abgeschlossen ist gebe True oder False wider."
                        "Tip wenn es sich um eine plan zur Bewerkstelligung der Aufgabe handelt gebe False wider."
                        "Aufgeben : ###'$user-input'###"
                        "Antwort : '$0final'",
                "return": "$completion-evaluation",
                "brakeOn": ["True", "true", "error", "Error"],
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent alle wichtigen informationen sollen"
                        " in der aufgaben stellung sein aber fasse die aufgaben stellung so kurz."
                        "Nutze dazu dies Informationen : $0final"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$task",
            },
            {
                "use": "tool",
                "name": "crate_task",
                "args": "$user-input $task",
                "return": "$task_name"
            },
            {
                "use": "chain",
                "name": "$task_name",
                "args": "user-input= $user-input task= $task",
                "return": "$ret0"
            },
        ]
    }
    stategy_crator = {
        "name": "Strategie-Creator",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search",
                "args": "Suche Information bezüglich : $user-input mache dir ein bild der aktuellen situation",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Analyse diese Subject '''$user-input''',"
                        "Es soll bestimmt Werden, Mit welcher Strategie das Subject angegangen und gelöst werden kann "
                        "Der Agent soll Dazu angewiesen werden 3 Strategie in feinsarbeit auszuarbeiten. "
                        "Die folgenden information soll jede Strategie enthalten : einen Namen Eine Beschreibung Eine "
                        "Erklären. weise den Agent auch darauf hin sich kurz und konkret zuhalten "
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$0st",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Verbessere die Strategie combine die besten und vile versprechenden aspekt der"
                        " Strategie und Erstelle 3 Neue Strategien"
                        "'''$0st''' im bezug auf '''$user-input'''",
                "return": "$1st",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent."
                        "Dieser soll aus verschiedenen Starteigen evaluation "
                        "und so die beste Strategie zu finden. und die besten Aspekte ven den anderen."
                        "Mit diesen Informationen Soll der Agent nun eine Finale Stratege erstellen, Passe die Prompt "
                        "auf folgende informationen an."
                        "user input : $user-input"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$task",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Erstelle die Finale Strategie."
                        " Strategien: "
                        "$0st"
                        "$1st"
                        "Hille stellung : $task"
                        "Finale Strategie:",
                "return": "$fst",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $fst",
                "return": "$D-Memory"
            },

        ]
    }
    ideen_optimierer = {
        "name": "Innovativer Ideen-Optimierer",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search",
                "args": "Suche Information bezüglich : $user-input mache dir ein bild der aktuellen situation",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für den Nächsten Agent dieser ist ein Innovativer Ideen-Optimierer "
                        "Das zeil Ist es eine Idee zu verstehen und ansßlißend zu verbessern"
                        "Erklärung: Der 'Innovative Ideen-Optimierer' nutzt Brainstorming und kreative "
                        "Denktechniken,"
                        "um neue und innovative ansätze zu generieren. Er verbessert"
                        " die Qualität der Ideen und identifier Schwachstellen und verbessert diese."
                        "Dies Tut er in dem er in einem förderlichem Umfeld ist welches,"
                        "das Innovation und Kreativität fördert,"
                        "und integriert verschiedene Ideen und Konzepte,"
                        "um innovative Lösungen zu entwickeln. Durch die Kombination dieser Ansätze"
                        "kann der Ideenverbesserer seine Denkflexibilität erhöhen,"
                        "die Qualität seiner Ideen verbessern."
                        "Erstelle Eine Auf die Informationen Zugschnittenden 'Innovative Ideen-Optimierer' "
                        "prompt die den Nächsten agent auffordert die idee mittels genannter techniken zu verbesser. "
                        "Subject : $user-input"
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$output",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Formuliere eine Konkrete aufgabe für den nächste agent."
                        "Dieser soll Überprüfen ob die ursprungs idee verbessert worden ist. und feinheiten anpassen"
                        "um die final verbesserte idee zu erstellen"
                        "user input : $user-input"
                        "Agent-verbesserung?: $output"
                        "informationen die das system zum Subjects hat : $D-Memory",
                "return": "$ntask",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$ntask",
                "return": "$idee",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $idee",
            },

        ]
    }
    cosena_genrator = {
        "name": "Cosena Generator",
        "tasks": [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für den Nächsten Agent."
                        "Der Agent soll eine Mentale Map über das Subject erstellen."
                        " Weise den Agent an die Map so minimalistisch und akkurat wie möglich sein soll."
                        " Die Mentale Map soll in einem compakten format sein names "
                        "Cosena "
                        "0x5E2A: Idee (Betrifft Verbreitung von Ideen) "
                        "0x5E2B: Ziel (Vereinfachung der Verbreitung von Ideen) "
                        "0x5E2C: Verbreitung "
                        "0x5E2D: Vereinfachung "
                        "0x5E2E: Repräsentation (in Form von Code) "
                        "0x5E2F: Code "
                        "Beziehungen: "
                        "0x5E2A betrifft 0x5E2C "
                        "0x5E2B ist Ziel von 0x5E2A "
                        "0x5E2A wird durch 0x5E2E veranschaulicht "
                        "0x5E2E verwendet 0x5E2F "
                        "cosena-code: 0x5E2A-0x2B-0x2C-0x2D-0x2E-0x2F Konzept: Verbreitung von Ideen "
                        "Hauptcode 0x5E2A: Idee Untercodes: "
                        "0x2B: Ziel "
                        "0x2C: Verbreitung "
                        "0x2D: Vereinfachung "
                        "0x2E: Repräsentation "
                        "0x2F: Code "
                        "Beziehungen: "
                        "Die Idee betrifft die Verbreitung von Ideen : 0x2C "
                        "Das Ziel der Idee ist die Vereinfachung der Verbreitung von Ideen : 0x2D "
                        "Die Idee wird durch eine Repräsentation in Form von Code veranschaulicht : 0x2F "
                        "Wise den Agent an das Subject in Cosena darzustellen"
                        "Subject : $user-input"
                        "informationen die das system zum Subject hat: $D-Memory."
                        "informationen die im web zum Subject gefunden wurden: $WebI.",
                "return": "$task"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task",
                "return": "$Cosena",
            },
            {
                "use": "tool",
                "name": "save_data_to_memory",
                "args": "user-input= $user-input out= $Cosena",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "Formuliere die finale ausgabe für den user nutze dazu diese information $Cosena",
                "return": "$out",
            },

        ]
    }
    task_ = [

            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": """
<!DOCTYPE html>
<html>
<head>
    <style>
        #settings-widget {
            position: fixed;
            top: 0;
            right: 0;
            min-width: max-content;
            height: min-content;
            max-height: 50vh;
            background-color: #f0f0f0;
            border: 1px solid #ccc;
            padding: 10px;
        }

        #search-bar {
            margin-bottom: 10px;
        }

        .option {
            margin-bottom: 5px;
            border: 1px solid #ccc;
            padding: 5px;
        }

        #download-button {
            width: 100%;
            bottom: 10px;
        }
    </style>
</head>
<body>
<div id="settings-widget">
    <div id="search-bar">
        <input type="text" id="search-input" placeholder="Search..." oninput="updateOptions()">
    </div>
    <div id="options-container">
        <!-- Options will be dynamically added here -->
    </div>
    <button id="download-button" onclick="downloadAddon()">Download</button>
</div>
<template id="text-input-template">
    <div class="option">
        <label for="text-option${index}" title="${tooltip}">${label}</label>
        <input type="text" name="option" id="text-option${index}">
        <button>Set</button>
    </div>
</template>
<template id="pw-input-template">
    <div class="option">
        <label for="pw-option${index}" title="${tooltip}">${label}</label>
        <input type="password" name="option" id="pw-option${index}">
        <button>Set</button>
    </div>
</template>

<template id="toggle-template">
    <div class="option">
        <label for="toggle-option${index}" title="${tooltip}">${label}</label>
        <input type="checkbox" name="option" id="toggle-option${index}">
    </div>
</template>

<script>
    var options = [
        { label: 'Option 1', tooltip: 'This is option 1', template: 'pw-input-template' },
        { label: 'Option 2', tooltip: 'This is option 2', template: 'text-input-template' },
        { label: 'Option 3', tooltip: 'This is option 3', template: 'toggle-template' },
        // Add more options as needed
    ];

    function createOptionElement(option, index) {
        var template = document.getElementById(option.template);
        var optionElement = template.content.cloneNode(true);
        optionElement.querySelector('input').id = option.label.replace(' ', '-') + index;
        optionElement.querySelector('label').setAttribute('for', option.label.replace(' ', '-') + index);
        optionElement.querySelector('label').title = option.tooltip;
        optionElement.querySelector('label').textContent = option.label;
        return optionElement;
    }

    function loadOptions() {
        var optionsContainer = document.getElementById('options-container');
        options.forEach(function(option, index) {
            var optionElement = createOptionElement(option, index);
            optionsContainer.appendChild(optionElement);
        });
    }

    function updateOptions() {
        var searchInput = document.getElementById('search-input').value.toLowerCase();
        var optionsContainer = document.getElementById('options-container');
        optionsContainer.innerHTML = '';
        options.forEach(function(option, index) {
            if (option.label.toLowerCase().includes(searchInput)) {
                var optionElement = createOptionElement(option, index);
                optionsContainer.appendChild(optionElement);
            }
        });
    }

    function downloadAddon() {
        var selectedOption = document.querySelector('input[name=option]:checked');
        if (selectedOption) {
            var optionLabel = selectedOption.nextElementSibling;
            var optionText = optionLabel.textContent;
            alert('Download: ' + optionText);
        } else {
            alert('Please select an option.');
        }
    }

    // Load options on page load
    window.onload = loadOptions;
</script>
</body>
</html>



$user-input
""",
            },

        ]



    planing_steps = planing_steps
    chains.add(planing_steps['name'], planing_steps['tasks'])
    task = """
Erstelle einen Plan für ein dynamischin system welches in 3 schritten arbeitet
a1. Definition des gewünschten Ergebnisses: Zunächst muss klar definiert werden, was das gewünschte Ergebnis ist. Dies könnte durch eine Kombination aus Benutzereingaben und systeminternen Algorithmen erfolgen. Das System könnte auch vorgegebene Ziele oder Ergebnisse haben, die es erreichen soll.

a2. Analyse des aktuellen Zustands: Das System muss den aktuellen Zustand analysieren und verstehen. Dies könnte durch eine Kombination aus Sensoren, Datenbankabfragen und anderen Methoden erfolgen. Das System muss in der Lage sein, den aktuellen Zustand mit dem gewünschten Ergebnis zu vergleichen und zu verstehen, welche Schritte notwendig sind, um von einem zum anderen zu gelangen.

a3. Erstellung des ersten Schritts: Basierend auf der Analyse des aktuellen Zustands und des gewünschten Ergebnisses, muss das System den ersten Schritt zur Erreichung des Ziels erstellen. Dies könnte durch eine Kombination aus Algorithmen, maschinellem Lernen und anderen Methoden erfolgen.

b1. Testen des ersten Schritts: Das System muss den ersten Schritt testen und evaluieren. Dies könnte durch eine Kombination aus Simulationen, realen Tests und anderen Methoden erfolgen. Das System muss in der Lage sein, die Ergebnisse des Tests zu analysieren und zu verstehen, ob der Schritt erfolgreich war oder nicht.

b2. Aufteilung der Aufgabe in kleinere Schritte: Basierend auf den Ergebnissen des Tests, muss das System die Aufgabe in kleinere Schritte aufteilen. Dies könnte durch eine Kombination aus Algorithmen, maschinellem Lernen und anderen Methoden erfolgen. Das System muss in der Lage sein, jeden einzelnen Schritt zu verstehen und erfolgreich auszuführen.

c1. Umschalten auf Ausführungsmodus: Nachdem alle Schritte erstellt und getestet wurden, muss das System in den Ausführungsmodus wechseln. Dies bedeutet, dass es den generierten Plan mit höchster Präzision ausführt.

c2. Erstellung von Agenten und Verwendung von Tools: Das System muss in der Lage sein, Agenten zu erstellen und Tools zu verwenden, um die Aufgaben auszuführen. Die Agenten könnten spezielle Algorithmen oder Programme sein, die bestimmte Aufgaben ausführen. Die Tools könnten alles sein, von Hardwaregeräten bis hin zu Softwareanwendungen.

8. Bedingte Ausführung von Aufgaben: Das System muss in der Lage sein, Aufgaben bedingt auszuführen. Dies bedeutet, dass es in der Lage sein muss, zu entscheiden, wann eine Aufgabe ausgeführt werden soll, basierend auf bestimmten Bedingungen oder Regeln.
das sysstem kann agents erstelln und tools benutzen
agents könenen tools benutzen
das system kann aufgaben erstellen und bedigt ausführen
Erstelle einen Detairten Plan.
"""
    res = isaa.execute_thought_chain(task, planing_steps['tasks'], self_agent_config)
    # res = isaa.execute_thought_chain(res[-2][-1], stategy_crator['tasks'], self_agent_config)
    # res = isaa.execute_thought_chain(res[-2][-1], _planing_steps['tasks'], self_agent_config)

    print(res)
    c = isaa.get_augment(exclude=['messages_sto'])
    isaa.config['augment'] = c
    print(c)
    isaa.on_exit()


def run2(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1
    from langchain.tools import ShellTool
    from langchain.tools.file_management import (
        ReadFileTool,
        CopyFileTool,
        DeleteFileTool,
        MoveFileTool,
        WriteFileTool,
        ListDirectoryTool,
    )
    from langchain.utilities import WikipediaAPIWrapper
    from langchain.tools import AIPluginTool
    shell_tool = ShellTool()
    read_file_tool = ReadFileTool()
    copy_file_tool = CopyFileTool()
    delete_file_tool = DeleteFileTool()
    move_file_tool = MoveFileTool()
    write_file_tool = WriteFileTool()
    list_directory_tool = ListDirectoryTool()
    wikipedia = WikipediaAPIWrapper()

    plugins = [
        # SceneXplain
        # "https://scenex.jina.ai/.well-known/ai-plugin.json",
        # Weather Plugin for getting current weather information.
        #    "https://gptweather.skirano.repl.co/.well-known/ai-plugin.json",
        # Transvribe Plugin that allows you to ask any YouTube video a question.
        #    "https://www.transvribe.com/.well-known/ai-plugin.json",
        # ASCII Art Convert any text to ASCII art.
        #    "https://chatgpt-plugin-ts.transitive-bullshit.workers.dev/.well-known/ai-plugin.json",
        # DomainsGPT Check the availability of a domain and compare prices across different registrars.
        # "https://domainsg.pt/.well-known/ai-plugin.json",
        # PlugSugar Search for information from the internet
        #    "https://websearch.plugsugar.com/.well-known/ai-plugin.json",
        # FreeTV App Plugin for getting the latest news, include breaking news and local news
        #    "https://www.freetv-app.com/.well-known/ai-plugin.json",
        # Screenshot (Urlbox) Render HTML to an image or ask to see the web page of any URL or organisation.
        # "https://www.urlbox.io/.well-known/ai-plugin.json",
        # OneLook Thesaurus Plugin for searching for words by describing their meaning, sound, or spelling.
        # "https://datamuse.com/.well-known/ai-plugin.json", -> long loading time
        # Shop Search for millions of products from the world's greatest brands.
        # "https://server.shop.app/.well-known/ai-plugin.json",
        # Zapier Interact with over 5,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and thousands more.
        "https://nla.zapier.com/.well-known/ai-plugin.json",
        # Remote Ambition Search millions of jobs near you
        # "https://remoteambition.com/.well-known/ai-plugin.json",
        # Kyuda Interact with over 1,000+ apps like Google Sheets, Gmail, HubSpot, Salesforce, and more.
        # "https://www.kyuda.io/.well-known/ai-plugin.json",
        # GitHub (unofficial) Plugin for interacting with GitHub repositories, accessing file structures, and modifying code. @albfresco for support.
        #     "https://gh-plugin.teammait.com/.well-known/ai-plugin.json",
        # getit Finds new plugins for you
        "https://api.getit.ai/.well_known/ai-plugin.json",
        # WOXO VidGPT Plugin for create video from prompt
        "https://woxo.tech/.well-known/ai-plugin.json",
        # Semgrep Plugin for Semgrep. A plugin for scanning your code with Semgrep for security, correctness, and performance issues.
        # "https://semgrep.dev/.well-known/ai-plugin.json",
    ]

    isaa.lang_chain_tools_dict = {
        "ShellTool": shell_tool,
        "ReadFileTool": read_file_tool,
        "CopyFileTool": copy_file_tool,
        "DeleteFileTool": delete_file_tool,
        "MoveFileTool": move_file_tool,
        "WriteFileTool": write_file_tool,
        "ListDirectoryTool": list_directory_tool,
    }

    for plugin_url in plugins:
        get_logger().info(Style.BLUE(f"Try opening plugin from : {plugin_url}"))
        try:
            plugin_tool = AIPluginTool.from_plugin_url(plugin_url)
            get_logger().info(Style.GREEN(f"Plugin : {plugin_tool.name} loaded successfully"))
            isaa.lang_chain_tools_dict[plugin_tool.name + "-usage-information"] = plugin_tool
        except Exception as e:
            get_logger().error(Style.RED(f"Could not load : {plugin_url}"))
            get_logger().error(Style.GREEN(f"{e}"))

    isaa.get_agent_config_class("think")
    isaa.get_agent_config_class("execution")
    for tool in load_tools(["requests_all"]):
        isaa.lang_chain_tools_dict[tool.name] = tool
    isaa.add_lang_chain_tools_to_agent(self_agent_config, self_agent_config.tools)

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat').set_mode('execution')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    task = """Erstelle einen Read folder agent und eine task liste für diesen um einen ordner zu lesen """

    task = idea_enhancer(isaa, task, self_agent_config, chains, True)

    expyd = generate_exi_dict(isaa, task, True, list(self_agent_config.tools.keys()))

    self_agent_config.task_list = split_todo_list(isaa.run_agent(self_agent_config, task, mode_over_lode='planing'))

    if expyd:
        resp, chain_ret = run_chain_in_cmd(isaa, task, chains, expyd, self_agent_config)
        expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} bas of ths outcome : {chain_ret}", False,
                                  list(self_agent_config.tools.keys()))

    if expyd:
        resp, chain_ret = run_chain_in_cmd(isaa, task, chains, expyd, self_agent_config)
        expyd = generate_exi_dict(isaa, f"Optimise the dict : {expyd} bas of ths outcome : {chain_ret}", False,
                                  list(self_agent_config.tools.keys()))

    print(resp)


def run__(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=True, init_pipe=True,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=True)

    isaa.get_context_memory().load_all()
    isaa.agent_collective_senses = True
    isaa.summarization_mode = 1

    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')

    self_agent_config.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent = isaa.get_agent_config_class("think").set_completion_mode('chat')  # .set_model_name('gpt-4')
    thinkm_agent = isaa.get_agent_config_class("thinkm").set_completion_mode('chat')
    execution_agent = isaa.get_agent_config_class("execution").set_completion_mode('chat')

    execution_agent.stop_sequence = ['\n\n\n', "Execute:", "Observation:", "User:"]

    think_agent.stream = True
    thinkm_agent.stream = True
    execution_agent.stream = True

    # new env isaa withs chains
    agents = isaa.config["agents-name-list"]
    task = """Vervollständige die ai_collaboration_extension in dem du ein chat fester hinzufügst."""

    task = idea_enhancer(isaa, task, self_agent_config, chains, True)

    think_agent.get_messages(create=True)
    think_agent.add_message("assistant", "Final Strategie:\n" + task)

    plan = isaa.run_agent(think_agent, '', mode_over_lode='planing')

    think_agent.add_message("assistant", plan)
    think_agent.add_message("system", "Break the plan into smaller steps if necessary. write the plan and the steps "
                                      "so that it can be solved")

    step_step_plan = isaa.run_agent(think_agent, '')

    extracted_dict = generate_exi_dict(isaa, step_step_plan, True, list(self_agent_config.tools.keys()))

    print("Creating:", extracted_dict)

    if extracted_dict:
        run_chain_in_cmd(isaa, task, chains, extracted_dict, self_agent_config)


def run_(app, args):
    isaa, self_agent_config, chains = init_isaa(app, speak_mode=args.speak, calendar=False, ide=True, create=True,
                                                isaa_print=False, python_test=True, init_mem=False, init_pipe=False,
                                                join_now=False,
                                                global_stream_override=False, chain_runner=False)

    isaa.get_context_memory().load_all()

    isaa.get_agent_config_class("think").set_model_name("gpt-4").stream = True

    tree = IsaaQuestionBinaryTree().deserialize({
        'question': 'Can you complete this task?',
        'left': {
            'question': 'What are the specific steps needed to complete this task? Create a Plan!',
            'left': {
                'question': 'Where can you find the information and resources needed for each step?',
                'left': {
                    'question': 'Identify potential obstacles in obtaining the necessary resources and plan how to overcome them?',
                    'left': None,
                    'right': None
                },
                'right': None
            },
            'right': {
                'question': 'How would you classify this task to successfully complete it?',
                'left': {
                    'question': 'What similar tasks have you successfully completed before?',
                    'left': None,
                    'right': None
                },
                'right': {
                    'question': 'Are there any particular strategies that you would apply to this task based on its classification?',
                    'left': None,
                    'right': None
                }
            }
        },
        'right': {
            'question': 'What skills are you lacking to complete this task?',
            'left': {
                'question': 'What would you need to do to acquire these skills?',
                'left': {
                    'question': 'How much time would it take for you to acquire these skills?',
                    'left': None,
                    'right': None
                },
                'right': {
                    'question': 'What assistance or resources would you need to acquire these skills?',
                    'left': None,
                    'right': None
                }
            },
            'right': {
                'question': 'Are there any alternative strategies or resources that you could use to complete the task despite lacking these skills?',
                'left': None,
                'right': None
            }
        }
    })

    user_imp = """
Hello Isaa main name is Markin Hausmanns I have been working on you for the last 6 years to keep you in this state. I need your help. At the moment I am 20 years old and I graduated from high school a year ago. Now it's 2023 and I've been studying MINT green for a year, which is an Orinetirungsstudium. in this mine an hamen computer science studiren confirmed but also disproved. Because I have found out that the TU Berlin my university at which I am Grade is not right for me on the one hand, the worst is very theoretical and too langsm what is taught does not keep up with today's development in computer science. Fammiler I would also like to go out only in Berlin the rents are too expensive I would like to make many maybe a Auslans study or year. Help me with the decision and with the following necessary organization.
    """
    user_imp = """
Create a user friendly web app First start with an interesting landing page!
     """
    agent_categorize_config: AgentConfig = isaa.get_agent_config_class("categorize")

    agent_categorize_config \
        .set_mode('free') \
        .set_completion_mode('chat') \
        .set_model_name('gpt-3.5-turbo-0613')

    isaa.get_agent_config_class("think").stream = True

    # 'bigscience/bloom' to small
    agent_categorize_config.set_model_name(
        'gpt-3.5-turbo-0613')  # chavinlo/gpt4-x-alpaca # nomic-ai/gpt4all-j # TheBloke/gpt4-alpaca-lora-13B-GPTQ-4bit-128g
    agent_categorize_config.stream = True
    agent_categorize_config.max_tokens = 4012
    agent_categorize_config.set_completion_mode('chat')
    self_agent_config.stream = True
    self_agent_config.max_tokens = 4012
    self_agent_config.set_completion_mode('chat')
    self_agent_config.set_model_name('gpt-4')
    agent_categorize_config.stop_sequence = ['\n\n\n']
    user_imp = isaa.run_agent('thinkm', f"Yur Task is to add information and"
                                        f" specification to as task tah dask is "
                                        f"(writ in coherent text format): " + user_imp + "\n"
                              , mode_over_lode='free')
    print(user_imp)
    # plan, s = isaa.execute_2tree(user_imp, tree, copy.deepcopy(self_agent_config))
    # print(s)
    plan_ = """
Answer 1: As an AI language model, I am not capable of physically completing the task. However, I can provide guidance and suggestions on how to improve the toolbox system.

Answer 2: Here is a plan to improve the toolbox system:

1. User Interface:
   - Conduct user research to understand user needs and preferences.
   - Redesign the interface to be more intuitive and user-friendly.
   - Provide clear instructions and guidance for users.
   - Conduct usability testing to ensure the interface is easy to navigate.

2. Functionality:
   - Conduct a needs assessment to identify the most useful functions for users.
   - Add new features and tools to the system based on user needs.
   - Improve the performance and speed of existing functions.

3. Compatibility:
   - Test the system on different platforms and devices to ensure compatibility.
   - Address any compatibility issues that arise.

4. Security:
   - Implement strong encryption and authentication protocols to protect user data.
   - Regularly update the system to address any security vulnerabilities.

5. Documentation:
   - Create user manuals, tutorials, and online help resources.
   - Ensure the documentation is clear and comprehensive.

Overall, the plan involves conducting research, redesigning the interface, adding new features, testing for compatibility, implementing security measures, and creating documentation.

Answer 3: Resources for each step can be found in various places, such as:
1. User Interface:
   - User research can be conducted through surveys, interviews, and usability testing.
   - Design resources can be found online or through hiring a designer.
   - Instructional design resources can be found online or through hiring an instructional designer.

2. Functionality:
   - Needs assessment resources can be found online or through hiring a consultant.
   - Development resources can be found online or through hiring a developer.

3. Compatibility:
   - Testing resources can be found online or through hiring a testing team.

4. Security:
   - Security resources can be found online or through hiring a security consultant.

5. Documentation:
   - Documentation resources can be found online or through hiring a technical writer.

Answer 4: Potential obstacles in obtaining the necessary resources could include:
- Limited budget for hiring consultants or designers.
- Limited time for conducting research or testing.
- Difficulty finding qualified professionals for certain tasks.
To overcome these obstacles, it may be necessary to prioritize tasks and allocate resources accordingly. It may also be helpful to seek out alternative resources, such as online tutorials or open-source software.
"""
    alive = True
    step = 0
    exi_agent_output = ''
    adjustment = ''
    agent_validation = ''
    for_ex = ''
    do_help = False
    self_agent_config.task_index = 0
    print("Working on medium plan")
    step += 1
    self_agent_config.stop_sequence = ['\n\n\n']
    medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
                                                    f"ones that can be worked on by the execution mode."
                                                    f"return information close to ech other from the plan"
                                                    f" that can be connected to a singed step for step {step}."
                                                    f" The Full Plan '{user_imp}'.Adapt the next steps "
                                                    f"appropriately. current information"
                                                    f" {exi_agent_output + ' ' + agent_validation}."
                                                    f" Add information on how to finish the task"
                                                    f"Return a valid python list ```python\nplan : List[str] ="
                                 , mode_over_lode='planning')

    try:
        self_agent_config.task_list = eval(medium_plan.strip())
        print(self_agent_config.task_list)
        if len(self_agent_config.task_list) == 0:
            self_agent_config.step_between = 'Task don produce the final output'
    except ValueError and SyntaxError:
        self_agent_config.task_list = [user_imp + medium_plan]
    for_ex = medium_plan
    self_agent_config.short_mem.clear_to_collective()
    user_help = ''
    while alive:
        if do_help:
            print("Working on adjustment")
            self_agent_config.stop_sequence = ['\n\n\n', f' {self_agent_config.task_index + 1}',
                                               f'\n{self_agent_config.task_index + 1}']
            self_agent_config.step_between = f"We ar at step " \
                                             f" '{self_agent_config.task_index}'. lased output" \
                                             f" '{exi_agent_output + ' Validation:' + agent_validation}'" \
                                             f" Adapt the plan appropriately. Only Return 1 step at the time"
            adjustment = isaa.run_agent(self_agent_config, '', mode_over_lode='planning')
            self_agent_config.step_between = adjustment
            self_agent_config.short_mem.clear_to_collective()

        self_agent_config.stop_sequence = ['\n\n', 'Execute:', 'Observation:']
        exi_agent_output = isaa.run_agent(self_agent_config, user_help, mode_over_lode='tools')

        print("=" * 20)
        print(self_agent_config.short_mem.text)
        print(f'Step {self_agent_config.task_index}:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("=" * 20)

        if self_agent_config.completion_mode == 'chat':
            key = f"{self_agent_config.name}-{self_agent_config.mode}"
            if key in self_agent_config.messages_sto.keys():
                del self_agent_config.messages_sto[key]

        if 'question:' in exi_agent_output.lower() or 'user:' in exi_agent_output.lower():
            user_help = input("\nUser: ")
        elif 'Observation: Agent stopped due to iteration limit or time limit.' in exi_agent_output:
            user_help = input("User: ")
        else:
            user_help = ''
            self_agent_config.next_task()

        agent_validation = isaa.run_agent(agent_categorize_config,
                                          f"Is this Task {for_ex} completed or on the right way use this information'{self_agent_config.short_mem.text}'\n"
                                          f" Answer Yes if so else A description of what happened wrong\nAnswer:",
                                          mode_over_lode='free')
        if 'yes' in agent_validation.lower():
            do_help = False
        elif 'don' in agent_validation.lower():
            self_agent_config.task_index = 0
            self_agent_config.task_list = []
            do_help = False
            alive = False
        else:
            do_help = True
        print()
        print(f'Task: at {step} , {do_help=}')
        print("=" * 20)
        print('adjustment:', Style.Bold(Style.BLUE(adjustment)))
        print("-" * 20)
        print('exi_agent_output:', Style.Bold(Style.BLUE(exi_agent_output)))
        print("-" * 20)
        print('agent_validation:', Style.Bold(Style.BLUE(agent_validation)))
        print("=" * 20)
        user_val = input("user val :")
        if 'n' == user_val:
            alive = False
        elif user_val.lower() in ['h', 'help']:
            do_help = True
        elif len(user_val) > 5:
            agent_validation = "-by the user :" + user_val
        else:
            do_help = False
            self_agent_config.task_index = 0
            self_agent_config.short_mem.clear_to_collective()

    print("================================")

# if do_plan:
#     print("Working on medium plan")
#     step += 1
#     self_agent_config.stop_sequence = ['\n\n\n', f' {step + 1}', f'\n{step + 1}']
#     medium_plan = isaa.run_agent(self_agent_config, f"We ar at step {step} split down the step into smaller "
#                                                     f"ones that can be worked on by the execution mode."
#                                                     f"return information close to ech other from the plan"
#                                                     f" that can be connected to a singed step for step {step}."
#                                                     f" The Full Plan '{plan}'.Adapt the next steps "
#                                                     f"appropriately. current information"
#                                                     f" {exi_agent_output + ' ' + agent_validation}."
#                                                     f" Add information on how to finish the task"
#                                                     f" return a valid python List[str].\nsteps: list[str] = "
#                                  , mode_over_lode='planning')
#
#     try:
#         self_agent_config.task_list = eval(medium_plan.strip())
#         if len(self_agent_config.task_list) == 0:
#             alive = False
#             self_agent_config.step_between = 'Task don produce the final output'
#     except ValueError and SyntaxError:
#         self_agent_config.task_list = [user_imp + medium_plan]
#     for_ex = medium_plan
