from toolboxv2.mods.isaa_extars.isaa_modi import init_isaa

NAME = "isaa-init-chains"


def run(app, args):

    isaa, self_agent_config, chains = init_isaa(app, speak_mode=False, calendar=False, ide=False, create=False)

    cahins = isaa.get_chain()

    cahins.add("Write_Tool_demo", [
        {
            "use": "tool",
            "description": "reading ToolBox and Main tool documentation",
            "name": "read",
            "args": "Toolbox.isaa_docs",
            "return": "$file-content",
            "text-splitter": 10000
        },
        {
            "use": "tool",
            "description": "reading base module documentation",
            "name": "read",
            "args": "$user-input",
            "return": "$file-content",
            "text-splitter": 10000
        },
        {
            "use": "agent",
            "mode": "tools",
            "completion-mode": "text",
            "name": "self",
            "args": "Act as a Python and programming expert your specialties are listing function for later implementation. you are known to think in small and detailed steps to get the right result. Your task : list the functions withe functionalities summary and an use case.\n$file-content",
            "chuck-run": "$file-content",
            "return": "$function-content"
        },
        {
            "use": "agent",
            "name": "think",
            "args": "Act as a Python and programming expert your specialties are writing documentation. you are known to think in small and detailed steps to get the right result. Your task : write an compleat documentation about $function-content"
        }
    ])

    cahins.add("Rad_Lage_File_and_writ_summary", [
        {
            "use": "tool",
            "name": "read",
            "args": "$user-input",
            "return": "$file-content",
            "text-splitter": 10000
        },
        {
            "use": "agent",
            "name": "think",
            "args": "Act as an summary expert your specialties are writing summary. you are known to think in small and detailed steps to get the right result. Your task : $file-content",
            "chuck-run": "$file-content",
            "return": "$summary"
        }
    ])

    cahins.add("next_three_days", [
        {
            "use": "tool",
            "name": "Calender",
            "args": "Rufe die Ereignisse der nachsten 3 Tage ab",
            "return": "$events"
        },
        {
            "use": "agent",
            "name": "summary",
            "args": "Fasse die Ereignisse $events der nachsten 3 Tage übersichtlich zusammen",
            "return": "$summary"
        }
    ])

    cahins.add("get_a_differentiated_point_of_view", [
        {
            "use": "tool",
            "name": "search",
            "args": "Suche Information zu $user-input",
            "return": "$infos-0"
        },
        {
            "use": "tool",
            "name": "search",
            "args": "Suche nach argument und information die f�r $user-input sprechen bezier $infos-0 mit ein",
            "return": "$infos-1"
        },
        {
            "use": "tool",
            "name": "search",
            "args": "Suche nach argument und information die gegen $user-input sprechen bezier $infos-0 mit ein",
            "return": "$infos-2"
        },
        {
            "use": "agent",
            "name": "think",
            "args": "fasse die information zu Thema $infos-0 \nPro seite $infos-1 \n\nCon seite $infos-2 \n\ndiffernzirte zusammen und berichte"
        }
    ])

    cahins.add("Generate_unit_Test", [
        {
            "use": "agent",
            "name": "code",
            "args": "Write a unit test for this function $user-input",
            "return": "$unit-test"
        },
        {
            "use": "agent",
            "name": "think",
            "args": "Act as a Python and programming expert your specialties are unit test. you are known to think in small and detailed steps to get the right result. Your task : Check if the unit test is correct $unit-test \nit should test this function $function\nif the unit test contains errors fix them.\nif the function contains errors fix them.\nreturn the function and the unit test."
        }
    ])

    cahins.add("gen_tool", [
        {
            "use": "tool",
            "name": "read",
            "args": "sum.data",
            "return": "$file-content",
            "text-splitter": 10000
        },
        {
            "use": "tool",
            "name": "read",
            "args": "Toolbox_docs.md",
            "return": "$docs-content"
        },
        {
            "use": "agent",
            "name": "think",
            "args": "Act as an summary expert your specialties are writing summary. you are known to think in small and detailed steps to get the right result. Your task : $file-content",
            "chuck-run": "$file-content",
            "return": "$summary"
        },
        {
            "use": "agent",
            "mode": "tools",
            "completion-mode": "text",
            "name": "self",
            "args": "Act as a Python and programming expert your specialties are writing Tools class and functions. you are known to think in small and detailed steps to get the right result. The MainTool: $docs-content\n\ninformation: $summary\n\n Your task : $user-input\n\n$file-content",
            "chuck-run": "$file-content",
            "return": "$function-content"
        }
    ])

    cahins.add("Generate_docs", [
        {
            "use": "tool",
            "name": "read",
            "args": "$user-input",
            "return": "$file-content",
            "text-splitter": 10000
        },
        {
            "use": "agent",
            "name": "self",
            "mode": "free",
            "completion-mode": "text",
            "args": "Act as a Python and programming expert your specialties are summarize functionalities of functions in one sentence. you are known to think in small and detailed steps to get the right result. Your task : list the functions withe functionalities summary and an use case.\n$file-content",
            "chuck-run": "$file-content",
            "return": "$function-content"
        },
        {
            "use": "agent",
            "name": "think",
            "args": "Act as a Python and programming expert your specialties are writing documentation. you are known to think in small and detailed steps to get the right result. Your task : write an compleat documentation about $function-content",
            "return": "$docs"
        },
        {
            "use": "tool",
            "name": "insert-edit",
            "args": "Toolbox_docs.md $docs"
        }
    ])

    cahins.add("calendar_entry", [
        {
            "use": "agent",
            "name": "categorize",
            "args": "Bestimme den Typ des Kalendereintrags basierend auf $user-input",
            "return": "$entry-type"
        },
        {
            "use": "tool",
            "name": "Calender",
            "args": "Speichere den Eintrag Typ: $entry-type \nUser: $user-input",
            "infos": "$Date"
        }
    ])

    cahins.save_to_file()

    exit(0)
