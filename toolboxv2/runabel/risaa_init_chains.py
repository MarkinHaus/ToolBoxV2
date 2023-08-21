from toolboxv2.mods.isaa import Tools
from toolboxv2.utils.toolbox import get_app

NAME = "isaa-init-chains"


def run(app, args):
    isaa: Tools = app.get_mod('isaa')
    chains = isaa.get_chain()

    first = False
    if first:
        chains.add("Write_Tool_demo", [
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

        chains.add("Rad_Lage_File_and_writ_summary", [
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

        chains.add("next_three_days", [
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

        chains.add("get_a_differentiated_point_of_view", [
            {
                "use": "tool",
                "name": "search_web",
                "args": "Suche Information zu $user-input",
                "return": "$infos-0"
            },
            {
                "use": "tool",
                "name": "search_web",
                "args": "Suche nach argument und information die f�r $user-input sprechen bezier $infos-0 mit ein",
                "return": "$infos-1"
            },
            {
                "use": "tool",
                "name": "search_web",
                "args": "Suche nach argument und information die gegen $user-input sprechen bezier $infos-0 mit ein",
                "return": "$infos-2"
            },
            {
                "use": "agent",
                "name": "think",
                "args": "fasse die information zu Thema $infos-0 \nPro seite $infos-1 \n\nCon seite $infos-2 \n\ndiffernzirte zusammen und berichte"
            }
        ])

        chains.add("Generate_unit_Test", [
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

        chains.add("gen_tool", [
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

        chains.add("Generate_docs", [
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

        chains.add("calendar_entry", [
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

        chains.add("First-Analysis", [
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
                "args": "Beantworte nach bestem wissen und Gewissen die Die Aufgabe. Wenn die aufgebe nicht "
                        "direct von dir gelöst werden kann spezifiziere die nächste schritte die eingeleitet"
                        " werden müssen um die Aufgabe zu bewerkstelligen es beste die option zwischen"
                        "der nutzung von tools und agents. aufgabe : $task",
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
        ])

        chains.add("Python-unit-test", [
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Die Nächste Prompt für das schrieben eines unit"
                        " test aufbau :  '''$user-input''', "
                        " Die prompt soll den agent auffordern eine unit test mit dem "
                        "python modul unittest zu schrieben."
                        """
füge Konkrete code Beispiele an da der nähste agent den aufbau nicht erhält. so ist deine aufgabe auch him diesen zu
 erklären und dan agent anzuleiten für die zu testende function einen test zu schreiben geb hin dafür
  auch die function.""",
                "return": "$task"
            },
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": "Schreibe einen unit test und erfülle die aufgabe "
                        " Der agent soll best practise anwenden :"
                        " 1. Verwenden Sie unittest, um Testfälle zu erstellen und Assertions durchzuführen."
                        "2. Schreiben Sie testbaren Code, der kleine, reine Funktionen verwendet und Abhängigkeiten "
                        "injiziert."
                        "3. Dokumentieren Sie Ihre Tests, um anderen Entwicklern zu helfen, den Zweck und die "
                        "Funktionalität der Tests zu verstehen."
                        "Task: $task\n\n"
                        "Code: $user-input",
                "return": "$return",
            },
        ])

        chains.add("Strategie-Creator", [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search_web",
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

        ])

        chains.add("Innovativer Ideen-Optimierer", [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "tool",
                "name": "search_web",
                "args": "Suche Information bezüglich : $user-input mache dir ein bild der aktuellen situation",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für den Nächsten Agent dieser ist ein Innovativer Ideen-Optimierer "
                        "Das zeil Ist es eine Idee zu verstehen und ansßlißend zu verbessern"
                        "um neue und innovative ansätze zu generieren.Verbessere"
                        " die Qualität der Ideen und identifier Schwachstellen und verbessert diese."
                        "integriere verschiedene Ideen und Konzepte,"
                        "um innovative Lösungen zu entwickeln. Durch die Kombination dieser Ansätze"
                        "kann der Ideenverbesserer seine Denkflexibilität erhöhen,"
                        "die Qualität der Ideen verbessern."
                        "Erstelle Eine Auf die Informationen Zugschnittenden 'Innovativer Ideen-Optimierer' "
                        "prompt die den Nächsten agent auffordert die idee erweitern und zu verbesser. "
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

        ])

        chains.add("Cosena Generator", [
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

        ])

        chains.add("function improver v1", [
            {
                "use": "tool",
                "name": "memory",
                "args": "$user-input",
                "return": "$D-Memory"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$user-input\nWelche technologien werden ind dieser function verwendend?"
                        " welche von diesen Ist veraltet und sollte Überarbeitet werden.",
                "return": "$technologien",
            },
            {
                "use": "tool",
                "name": "search_web",
                "args": "Suche nach information bezüglich : $technologien gibe "
                        "mir die und Version und Anwendung beispiele.",
                "return": "$WebI"
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Der Näste schritt ist die Analyse. **Analyse der aktuellen Funktion**: "
                        "Zunächst ist es wichtig, die aktuelle Funktion zu verstehen. Dazu "
                        "gehört das Verstehen der Logik, der Eingabe- und Ausgabeformate und "
                        "der aktuellen Leistung. Hierbei können Tools wie Profiler und Debugger "
                        "hilfreich sein."
                        "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                        " den Nächsten Agent um Die funktion detailliert zu analysis"
                        "function : $user-input\n"
                        "Web Infos: $WebI\n"
                        "informationen die das system zum Subjects hat : $D-Memory\n",
                "return": "$ntask",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$ntask"
                        "function : $user-input\n",
                "return": "$Analyse",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Der Näste schritt ist die **Identifizierung von "
                        "Verbesserungsbereichen**: Basierend auf der Analyse identifizieren wir "
                        "Bereiche, die verbessert werden können. Dies könnte das Logging,"
                        " das Error Handling, die Effizienz und die "
                        "Anpassungsfähigkeit der Funktion umfassen."
                        "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                        " den Nächsten Agent."
                        "function : $user-input\n"
                        "Analyse: $Analyse\n"
                        "informationen die das system zum Subjects hat : $D-Memory\n",
                "return": "$ntask2",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$ntask2"
                        "function : $user-input\n"
                        "Analyse : $Analyse\n",
                "return": "$Verbesserungsbereichen",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Der Näste schritt ist die **Entwicklung einer Strategie zur "
                        "Verbesserung**: Nachdem die Verbesserungsbereiche identifiziert "
                        "wurden, entwickeln wir eine Strategie zur Verbesserung. Dies könnte "
                        "die Verwendung von KI-Tools zur Automatisierung bestimmter "
                        "Teilschritte, die Verbesserung des Logging's "
                        "und des Error Handlings durch die Verwendung geeigneter Bibliotheken "
                        "und Techniken, die Verbesserung der Effizienz durch die Verwendung "
                        "effizienterer Algorithmen oder Datenstrukturen und die Verbesserung "
                        "der Anpassungsfähigkeit durch die Verwendung flexiblerer "
                        "Datenstrukturen und Algorithmen umfassen."
                        "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                        " den Nächsten Agent."
                        "function : $user-input\n"
                        "Verbesserungsbereichen: $Verbesserungsbereichen\n"
                        "informationen die das system zum Subjects hat : $D-Memory\n",
                "return": "$ntask3",
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$ntask3"
                        "function : $user-input\n"
                        "Verbesserungsbereichen : $Verbesserungsbereichen\n",
                "return": "$Strategie",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Der Näste schritt ist die **Implementierung der Verbesserungen**: "
                        "Nachdem die Strategie entwickelt wurde, implementieren wir die "
                        "Verbesserungen. Dies könnte die Neuschreibung bestimmter Teile des "
                        "Codes, die Hinzufügung neuer Funktionen oder die Änderung der Art und "
                        "Weise, wie bestimmte Operationen durchgeführt werden, umfassen."
                        "Erstelle Basierend auf den dir vorliegenden Informationen ein Prompt für"
                        " den Nächsten Agent. Erstelle Einen Konkreten Schritt für schritt"
                        " anweisung um eine Finale function zu erstellen."
                        "Web Infos: $WebI\n\n"
                        "Analyse: $Analyse\n\n"
                        "Verbesserungsbereichen: $Verbesserungsbereichen\n\n"
                        "$Strategie: $Strategie\n\n"
                        "informationen die das system zum Subjects hat : $D-Memory\n",
                "return": "$ntask4",
            },
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": "$ntask4"
                        "function : $user-input\n"
                        "Erstelle die perfecte function.",
                "return": "$function",
            },
        ])

        chains.add("first widget generator v1", [
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
                "args": "Erstelle Eine Prompt für die Erstellung eines Möglichen aufbaus eines Html Widget für diese "
                        "Subjects '''$user-input''', zu erstellen,"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task0"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task0",
                "return": "$0final",
            },
            {
                "use": "agent",
                "mode": "generate",
                "name": "self",
                "args": "Erstelle Eine Prompt für die Entwicklung der specification"
                        " für diese Subjects '''$user-input''',"
                        "informationen die das system zum Subjects hat: $D-Memory",
                "return": "$task1"
            },
            {
                "use": "agent",
                "mode": "free",
                "name": "think",
                "args": "$task1 ,benötigte elemente $0final",
                "return": "$1final",
            },
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": """
    Beisiel :
    <template id="text-widget-template">
        <div className="text-widget widget draggable">
            <div className="text-widget-from widget-from"></div>
            <span className="text-widget-close-button widget-close-button">X</span>
            <label htmlFor="text-widget-text-input"></label>
            <!-- mor spesific widget content -->
        </div>
    </template>


    function addWidget(element_id, id, context, content="") {
        console.log("ADDING Widget ", element_id);
        const targetElement = document.getElementById(element_id);
        let speechBalloon = createWidget(id, context, targetElement, content);
        targetElement.appendChild(speechBalloon);
        speechBalloon.id = id;
        return speechBalloon
    }

    function createWidget(textarea_id, context, targetElement, content="") {
        const template = document.getElementById('widget-template');
        const widget = template.content.cloneNode(true).querySelector('.text-widget');
        const fromElement = widget.querySelector('.text-widget-from');
        fromElement.textContent = context;
        const textarea = widget.querySelector('#text-widget-text-input');
        const widget_injection = widget.querySelector('#text-widget-injection');
        textarea.id = textarea_id+'-Text';
        textarea.value = content;

        const closeButton = widget.querySelector('.text-widget-close-button');
        closeButton.addEventListener('click', closeWidget);
        widget_injection.addEventListener('click', ()=>{
            console.log("widget_injection:testWidget")
            WS.send(JSON.stringify({"ChairData":true, "data": {"type":"textWidgetData","context":context,
                    "id": textarea_id, "text": textarea.value}}));
        });

        function closeWidget() {
            widget.style.animation = 'text-widget-fadeOut 0.5s';
            setTimeout(() => {
                widget.style.display = 'none';
                targetElement.removeChild(widget);
            }, 500);
        }

        return widget;
    }
    Der bereitgestellte Code ist ein Beispiel für ein Widget-System, das in einer Webanwendung verwendet wird. Es besteht aus einem HTML-Template und zwei JavaScript-Funktionen, `addTextWidget` und `createTextWidget`.

    Das HTML-Template definiert die Struktur des Widgets. Es enthält ein Textfeld, einen Senden-Button und einen Schließen-Button. Das Textfeld wird verwendet, um Text einzugeben, der Senden-Button, um den eingegebenen Text zu senden, und der Schließen-Button, um das Widget zu schließen.

    Die Funktion `addTextWidget` nimmt vier Parameter: `element_id`, `id`, `context` und `content`. `element_id` ist die ID des HTML-Elements, in das das Widget eingefügt werden soll. `id` ist die ID des Widgets. `context` ist der Kontext, in dem das Widget verwendet wird. `content` ist der anfängliche Text, der im Textfeld des Widgets angezeigt wird. Die Funktion erstellt ein neues Widget mit der Funktion `createTextWidget`, fügt es in das Ziel-HTML-Element ein und gibt das Widget zurück.

    Die Funktion `createTextWidget` nimmt die gleichen vier Parameter wie `addTextWidget`. Sie klont das HTML-Template, fügt den Kontext und den anfänglichen Text in das geklonte Template ein und fügt EventListener für die Schließen- und Senden-Buttons hinzu. Der Schließen-Button entfernt das Widget aus dem DOM. Der Senden-Button sendet den eingegebenen Text an einen WebSocket-Server. Die Funktion gibt das erstellte Widget zurück.

    Hier ist ein Beispiel, wie man diese Funktionen verwenden kann:

    ```javascript
    // Fügt ein Widget in das HTML-Element mit der ID 'myElement' ein.
    // Das Widget hat die ID 'myWidget', den Kontext 'myContext' und den anfänglichen Text 'Hello, world!'.
    addTextWidget('myElement', 'myWidget', 'myContext', 'Hello, world!');
    ```

    Mit diesem Code können Sie ein Widget-System erstellen, das es Benutzern ermöglicht, Text in einem Textfeld einzugeben, den Text zu senden und das Widget zu schließen. Dieses Konzept kann auf verschiedene Arten von Widgets angewendet werden, nicht nur auf Text-Widgets.

    Erstelle ein Neues Widget Benutze dafür das hir beschriebene muster.
    specification :
    $1final
                    """,
                "return": "$completion-evaluation",
                "brakeOn": ["True", "true", "error", "Error"],
            },
        ])

        chains.add("Clean git Commit", [
            {
                "use": "agent",
                "name": "think",
                "args": "Erstelle einen schönen git commit in Markdown format informationen : $user-input"
            }
        ])
        chains.add("Crate mini task prompt", [
            {
                "use": "agent",
                "name": "think",
                "mode": "generate",
                "args": "Erstelle einen schönen git commit in Markdown format informationen : $user-input"
            }
        ])
    else:
        chains.add("WeiteCode", [
            {
                "use": "tool",
                "name": "write-production-redy-code",
                "args": "$user-input"
            }
        ])

    chains.save_to_file()

    exit(0)


if __name__ == "__main__":
    run(get_app('main'), None)
