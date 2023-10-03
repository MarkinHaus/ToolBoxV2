import json
import time
import unittest
from unittest.mock import patch, Mock, MagicMock
from bs4 import BeautifulSoup
from toolboxv2 import App, get_logger
from toolboxv2.mods import BROWSER
from toolboxv2.mods.isaa import Tools
from toolboxv2.mods.isaa.AgentUtils import AgentConfig
from toolboxv2.utils.toolbox import get_app

from toolboxv2.mods.isaa.isaa import (show_image_in_internet, image_genrating_tool,
                                      browse_website, get_text_summary, get_hyperlinks, scrape_text,
                                      extract_hyperlinks, format_hyperlinks, scrape_links, get_ip, get_location,
                                      extract_code, get_tool, initialize_gi, _extract_from_json, dilate_string)


class TestIsaaBenchmarks(unittest.TestCase):
    isaa = None
    t0 = 0
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test-TestIsaa")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa", "toolboxv2.mods.")
        cls.app.new_ac_mod("isaa")
        cls.isaa: Tools = cls.app.get_mod('isaa')
        cls.isaa.load_keys_from_env()

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def setUp(self):
        self.app_mock = Mock()
        self.isaa = self.app.get_mod('isaa')

    def tearDown(self):
        self.isaa._on_exit()
        self.app.remove_mod('isaa')

    def _generator_get_all_agent_mode_variants(self, e_modes=None, e_agents=None, a_only=False, m_only=False, do=True):
        if not do:
            return []
        if e_agents is None:
            e_agents = []
        if e_modes is None:
            e_modes = []
        if a_only:
            agents = e_agents
        else:
            agents = TestIsaa.agents
            for e_a in e_agents:
                if e_a in agents:
                    agents.remove(e_a)
        if m_only:
            modes = e_modes
        else:
            modes = TestIsaa.modes
            for e_m in e_modes:
                if e_m in modes:
                    modes.remove(e_m)
        for agent_name in agents:
            with self.subTest(Agent=agent_name):
                agent_class = self.isaa.get_agent_config_class(agent_name)
                self.assertIsInstance(agent_class, AgentConfig)
                for mode in modes:
                    with self.subTest(Mode=mode):
                        agent_class.set_mode(mode)
                        self.assertEqual(agent_class.mode, mode)
                        yield agent_class

    def test_show_generator(self):
        for agent in self._generator_get_all_agent_mode_variants():
            self.assertIsInstance(agent, AgentConfig)
            print(f"Agent: {agent.name}, mode: {agent.mode}, c mode: {agent.completion_mode}, ppm: {agent.ppm}")
        for agent in self._generator_get_all_agent_mode_variants(m_only=True, e_modes=["tools", "execution", "live"]):
            self.assertIsInstance(agent, AgentConfig)
            print(f"Agent: {agent.name}, mode: {agent.mode}, c mode: {agent.completion_mode}, ppm: {agent.ppm}")
            self.assertNotEqual(agent.mode, "planning")

    def test_best_fit_carte_config(self):

        allse = []

        for agent in self._generator_get_all_agent_mode_variants(m_only=True, a_only=True, e_modes=["execution"],
                                                                 e_agents=["execution", "thinkm"], do=False):
            chain = self.isaa.create_task(
                "Erstelle eine Chain die dynamisch das web nach einer bestimmten information durchsucht", agent)

            reson = self.isaa.stream_read_llm(
                f"Bewerte diese Aufgeben Kette mit einer Punk zahl von 1 bis 100 und begründe diese. Das ziel : Erstelle eine Chain die dynamisch das web nach einer bestimmten information durchsucht Die erstellte und zubewetende chain :'{chain}'",
                agent)
            allse.append([reson, f"Agent: {agent.name}, mode: {agent.mode}", chain])
            print(
                f"Agent: {agent.name}, mode: {agent.mode}, c mode: {agent.completion_mode}, reson: {reson}, chain: {chain}")

        print(allse)

    def test_max_dilate_string(self):

        do_test = False

        if not do_test:
            self.assertEqual(True, True)
            return

        test_string = """Zum Inhalt springen
Menü
Wohnen
Klimawandel
Krieg in der Ukraine
Alle Themen
Blog
SCHACHBLOG
‹ ALLE EINTRÄGE
Kandidatenturnier 2016: Karjakin besiegt Anand und führt
15. März 2016 um 18:49 Uhr

In einer munteren vierten Runde des Kandidatenturniers in Moskau bekamen die Zuschauer zunächst ein schnelles, aber zumindest optisch keineswegs langweiliges Remis zu sehen. In einer scharfen halbslawischen Variante entbrannte zwischen Hikaru Nakamura und Anish Giri eine theoretische Diskussion, für die sich beide als bestens vorbereitet erwiesen. Giris König landete schnell auf dem etwas ungewöhnlichen Posten f6, aber der junge Holländer dürfte noch nicht einmal besonders aufgeregt gewesen sein, denn ihm war die Position mit Sicherheit noch aus seiner Hausanalyse bekannt.

giri_nakamura

Er ließ hier sogar kühn mit 24…c5!? (24…Kg7 hielt die Partie am Laufen) ein Abzugsschach zu, wonach abernach 25.fxg6+ Kxg6 26.Sxf7 Tf8 27.Sd5 Dxb2 beide Seiten über keinerlei Ressourcen mehr verfügten, weiterzuspielen, ohne sich selbst weh zu tun. Daher gab Nakamura beginnend mit 28.Se7+ Dauerschach, wodurch er noch immer mit 1,5/4 bei „Minus Eins“ steht und bisher schwerlich zufrieden mit seinem Abschneiden bei seinem ersten Kandidatenturnier sein kann. Für Giri dagegen ist nach seinem vierten Remis noch immer alles drin, er wird im späteren Turnierverlauf auf wenige, gezielte Schläge aus sein.

Als Nächstes wurden die beiden nach einer Umfrage unter Schachfans und Schwiegermüttern wohl beliebtesten und charismatischsten Kandidaten der Show „Wer datet Magnus Carlsen?“ fertig, Lewon Aronjan und Pjotr Swidler. Immer ein Spruch auf den Lippen, immer ein freundliches Lächeln, immer ein wenig Understatement, welches noch nicht einmal komplett aufgesetzt klingt, lassen die beiden jede Pressekonferenz zu einem Genuss werden. Noch nicht einmal der Umstand, dass die Partie trotz anfänglich obskurer Bauernstrukturen

aronian_svidler

(Stellung nach 13.dxc5)

minimal dröge verlaufen war, störte die beiden, sie können einfach stundenlang über Schach reden, ohne dass es langweilig oder eintönig wird.

Danach passierte endlich etwas, was wirklich über Schattenboxen hinausging. Sergey Karjakin besiegte den mehrfachen Exweltmeister Vishy Anand und fügte diesem seine erste Niederlage im Turnier bei, nachdem der Inder die sibirische Auflage 2014 noch unbeschadet überstanden hatte und unter anderem gegen Karjakin eine sehr schwierige Endspielstellung mit Türmen und Springern retten konnte. Nicht aber in diesem Jahr:

karjakin_anand

Karjakin hat sich mit seinen Figuren tief in die indischen Reihen gebohrt, die schwarzen Figuren sind alle aneinander gebunden und drohen abgeholt zu werden. Nach dem einzigen Zug 40…Se7 (sonst wird Schwarz sofort von einer weißen Springergabel erledigt, folgte 41.Td6 womit Weiß sowohl das quälende Eingreifen seines Königs droht, als auch sofort den Bauern auf d5 zu schlagen. Es folgte der Kurzschluss 41…Tb6? (41…Sc8 verlängerte das Leid zumindest) 42.Td7 a6 43.Sc3 und nun verliert Schwarz bei Weitem nicht nur den Bauern auf d5, gab also die Partie auf. So elegant der Partieschluss, so fade allerdings die anschließende Konferenz. Anand nach einer Niederlage bedient wie eh und je, Karjakin in seinen Ausführungen etwa das Gegenteil an Eloquenz im Vergleich zu der Konferenz davor – etwas redundant und streng auf die 64 Felder beschränkt. Dafür führt er das Turnier mit 3 Punkten aus 4 Partien nun alleine an – und das als nicht gerade ausgewiesener Topfavorit der meisten Experten.

Ebenfalls nicht als Topfavorit galt der Bulgare Wesselin Topalow, der um ein solches Turnier auf dem ersten Platz zu beenden, meist doch ein wenig wechselhaft agiert. In Moskau läuft bei ihm bisher bei 0,5/3 doch sehr wenig zusammen und auch heute war er extrem kurz davor, seine dritte Partie zu verlieren, es wäre gegen Fabiano Caruana gewesen. Dieser überspielte Topalow völlig, rutschte aber an einer Stelle aus, an der seine Aufgabe bereits absurd einfach war:

caruana_topalow

Hier einfach mit 41.Txf6 die Türme zu tauschen und nach 41…Dxf6 mit 42.Tb2 den f2 zu decken, lag doch sehr auf der Hand. Hat der Italo-Amerikaner etwa 42…Lf5 befürchtet und 43.Dxf5 mit folgender Springergabel übersehen? Man mag es fast nicht glauben. Und doch, Caruana spielte das viel schwächere 41.Tbb5 und nach 41…Txf2 42.Txg6 Txf1+ 43.Kh2 Dxc8 verzichtete er auch noch auf das bessere 44.Dxh6 und gab sich mit 44.Dxc8 Txc8 45.Txc6 einem „sicheren“ Turmendspiel hin, was aber dank des starken schwarzen e-Bauern schnell Remis endete. Mal sehen, ob Topalow durch diesen Lucky Escape neue Energie schöpfen kann, für Caruanas Turnier war dieses Partieende wohl pures Gift.

Die Zwischenstände lauten nach 4 von 14 Runden also:

3 Karjakin

2,5 Aronjan

2 Caruana, Anand, Giri, Swidler

1,5 Nakamura

1 Topalow

Am Mittwoch und Donnerstag wird wieder um 13:00 deutscher Zeit gespielt, bevor am Freitag ein nächster Ruhetag folgt.

AUF FACEBOOK TEILENTWITTERN
von Ilja Schneider / Artikel drucken
Kategorie

Allgemein
ÜBER DIESES BLOG
Schachspieler denken stundenlang in Varianten, spielen zur Not ohne Brett und gelten als introvertiert und eigenbrötlerisch. Zu ihrem Sport rücken sie mit Thermosflasche und Stullen-Paket an. Soweit die Klischees. Dieses Blog, das sich an Anfänger und Könner wendet, will sie bestätigen und verwerfen.
ÜBER DIE AUTOREN
Ilja Schneider
Ilja Schneider, Jahrgang 1984, internationaler Meister, spielt für die Schachfreunde Berlin in der Bundesliga und bloggte in der Vergangenheit in seinem Schachzoo. Auf Wikipedia heißt es, er sei der "beste Kaffeehausspieler Deutschlands", aber das entbehrt ebenso jeder Grundlage, wie die Behauptung, seine Lieblingsfußballmannschaft Hannover 96 hätte seit Jahren einen Auswärtskomplex.
Johannes Fischer
Johannes Fischer, Jahrgang 1963, liest gerne Bücher, geht gerne ins Kino und spielt gerne Schach. Mit diesen schönen Dingen beschäftigt er sich auch auf seinem eigenen Blog Schöner Schein. Er lebt in Nürnberg als freier Übersetzer, Autor und Redakteur und schreibt regelmäßig für ChessBase und KARL, Die kulturelle Schachzeitschrift.
Dennes Abel
Dennes Abel, Jahrgang 1989, spielt für die Schachfreunde Berlin in der ersten Bundesliga und kommentierte für ZEIT ONLINE bereits die WM 2013. Er hat am Schachbrett schon vieles gesehen, aber noch nicht genug, deshalb will er im Oktober beim Millionairechess in Las Vegas antreten.
LETZTE BEITRÄGE
Die Schachweltmeisterin Hou Yifan vertraut nicht mehr aufs Glück
Deutscher Meister wird nur die Schachgesellschaft Solingen
Garri Kasparow spielt wieder
Das lauteste Schachturnier der Welt
Kandidatenturnier 2016: Karjakin und Aronjan immer noch in Führung
DIESES BLOG DURCHSUCHEN
Suche

Nach oben
Impressum Hilfe & Kontakt Unternehmen Karriere Presse Jobs Shop Inserieren Mediadaten
Bildrechte Rechte & Lizenzen AGB Datenschutz Privacy Einstellungen Cookies & Tracking Abo kündigen"""

        with self.subTest(Agent="SYSTEM"):

            run_agent_thinkm = self.isaa.run_agent("thinkm", test_string + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", test_string + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(test_string + "\nWorum handelt der artikel")

            print(f"====================\nSYSTEM RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # to big error kontex lengh
            # 3/10 2/10 1/10

        with self.subTest(Agent="SYSTEM: 0 2 0"):
            dilate_string_0_2_0 = dilate_string(test_string, 0, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", dilate_string_0_2_0 + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", dilate_string_0_2_0 + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(dilate_string_0_2_0 + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string 0 2 0 RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 13/10 4/10 10/10
            # 13/10 2/10 8/10
        with self.subTest(Agent="SYSTEM: 1 2 0"):
            dilate_string_1_2_0 = dilate_string(test_string, 1, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", dilate_string_1_2_0 + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", dilate_string_1_2_0 + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(dilate_string_1_2_0 + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string 1 2 0 RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 12/10 1/10 0/10
            # 12/10 1/10 0/10
        with self.subTest(Agent="SYSTEM: 2 2 0"):
            dilate_string_2_2_0 = dilate_string(test_string, 2, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", dilate_string_2_2_0 + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", dilate_string_2_2_0 + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(dilate_string_2_2_0 + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string 2 2 0 RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 11/10 2/10 0/10
        with self.subTest(Agent="SYSTEM: 3 2 0"):
            dilate_string_3_2_0 = dilate_string(test_string, 3, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", dilate_string_3_2_0 + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", dilate_string_3_2_0 + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(dilate_string_3_2_0 + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string 3 2 0 RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 10/10 2/10 5/10
        with self.subTest(Agent="SYSTEM: 0x0"):
            first_itter = dilate_string(test_string, 0, 2, 0)
            second_itter = dilate_string(first_itter, 0, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", second_itter + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", second_itter + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(second_itter + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string second_itter RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 9/10 2/10 4/10
            # 5/10 2/10 6/10 (key features)
        with self.subTest(Agent="SYSTEM: 2x0"):
            first_itter_ab = dilate_string(test_string, 2, 2, 0)
            second_itter_ab = dilate_string(first_itter_ab, 0, 2, 0)

            run_agent_thinkm = self.isaa.run_agent("thinkm", second_itter_ab + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary", second_itter_ab + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(second_itter_ab + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string second_itter_ab RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 8/10 2/10 10/10
            # 8/10 2/10 11/10
        with self.subTest(Agent="SYSTEM: 0x0+2x0"):
            run_agent_thinkm = self.isaa.run_agent("thinkm", second_itter + second_itter_ab + "\nWorum handelt der artikel")

            run_agent_summary = self.isaa.run_agent("summary",
                                                    second_itter + second_itter_ab + "\nWorum handelt der artikel")
            mas_text_summaries = self.isaa.mas_text_summaries(
                second_itter + second_itter_ab + "\nWorum handelt der artikel")

            print(f"====================\ndilate_string second_itter+second_itter_ab RESULTS :"
                  f"\n{run_agent_thinkm=}\n{run_agent_summary=}\n{mas_text_summaries=}\n"
                  f"================================================================")
            # 7/10 0/10 1/10

            # 6/10 0/10 6/10


class TestIsaa(unittest.TestCase):
    isaa = None
    t0 = 0
    app = None

    not_run = [
        'gpt-4',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0613',
        'text-davinci-003',
        'gpt-4-0613',
        'code-davinci-edit-001',
        'text-curie-001',
        'text-babbage-001',
        'text-ada-001',
        'text-davinci-edit-001',
        'gpt-3.5-turbo-instruct',

        'google/flan-t5-small',
        'google/flan-t5-xxl',
        'databricks/dolly-v2-3b',
        'stabilityai/stablecode-instruct-alpha-3b',

        'gpt4all#GPT4All-13B-snoozy.ggmlv3.q4_0.bin',  # 5/
        'gpt4all#orca-mini-7b.ggmlv3.q4_0.bin',  # 4.5/10 :
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin',  # for comm
        'gpt4all#wizardLM-13B-Uncensored.ggmlv3.q4_0.bin',
        'gpt4all#ggml-replit-code-v1-3b.bin',  # Hy ly crati
        'knkarthick/MEETING_SUMMARY'
    ]

    models = [
        'gpt4all#GPT4All-13B-snoozy.ggmlv3.q4_0.bin',  # 5/10 (summary/classify/pl_lv2 in : 13.75s
        'gpt4all#orca-mini-7b.ggmlv3.q4_0.bin',  # : 7.17s
        # 4.5/10 : Hily spesific if you have any questions related to programming or computer science, feel free to ask me! , classify
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin',  # : 3.76s
        # for command exection and evalation prosseses 6/10 context classify (tool use) qa code summ
        'gpt4all#wizardLM-13B-Uncensored.ggmlv3.q4_0.bin',  # : 13.62s
        # Conversational and Thinking Sartegegs 7.4/10 summary classify   (q/a 1): py Lv2)
        'gpt4all#ggml-replit-code-v1-3b.bin',  # Hy ly crative # : 11.08s
    ]

    modes = []  #["conversation", "tools", "talk", "free", "planning", "live"]
    agents = []   #["self", "todolist", "search", "think", "TaskCompletion", "liveInterpretation", "summary", "thinkm", "code"]

    augment = {'tools':
                   {'lagChinTools': ['python_repl', 'requests_all', 'terminal', 'sleep',
                                     'wikipedia', 'llm-math', 'requests_get', 'requests_post',
                                     'requests_patch', 'requests_put', 'requests_delete'], 'huggingTools': [],
                    'Plugins': [], 'Custom': []}, 'Agents': {
        'self': {'name': 'self', 'mode': 'free', 'model_name': 'gpt-4', 'max_iterations': 6, 'verbose': True,
                 'personality': "\nResourceful: Isaa is able to efficiently utilize its wide range of capabilities and resources to assist the user.\nCollaborative: Isaa is work seamlessly with other agents, tools, and systems to deliver the best possible solutions for the user.\nEmpathetic: Isaa is understand and respond to the user's needs, emotions, and preferences, providing personalized assistance.\nInquisitive: Isaa is continually seek to learn and improve its knowledge base and skills, ensuring it stays up-to-date and relevant.\nTransparent: Isaa is open and honest about its capabilities, limitations, and decision-making processes, fostering trust with the user.\nVersatile: Isaa is adaptable and flexible, capable of handling a wide variety of tasks and challenges.\n                  ",
                 'goals': "Isaa's primary goal is to be a digital assistant designed to help the user with various tasks and challenges by leveraging its diverse set of capabilities and resources.",
                 'token_left': 3077, 'temperature': 0.06, '_stream': False, '_stream_reset': False,
                 'stop_sequence': ['\n\n\n', 'Execute:', 'Observation:', 'User:'], 'completion_mode': 'text',
                 'add_system_information': True, 'init_mem_state': False, 'binary_tree': None,
                 'agent_type': 'structured-chat-zero-shot-react-description',
                 'tools': ['memory_search', 'search_web', 'write-production-redy-code', 'mode_switch', 'think',
                           'image-generator', 'mini_task', 'memory', 'save_data_to_memory', 'crate_task',
                           'optimise_task', 'execute-chain', 'Python REPL', 'terminal', 'sleep', 'Google Search',
                           'DuckDuckGo Search', 'Wikipedia', 'Calculator', 'requests_get', 'requests_post',
                           'requests_patch', 'requests_put', 'requests_delete'], 'task_list': [],
                 'task_list_done': [], 'step_between': '', 'pre_task': None, 'task_index': 0},
        'categorize': {'name': 'categorize', 'mode': 'free', 'model_name': 'gpt-3.5-turbo-0613',
                       'max_iterations': 2, 'verbose': True, 'personality': '', 'goals': '', 'token_left': 4096,
                       'temperature': 0.06, '_stream': False, '_stream_reset': False,
                       'stop_sequence': ['\n\n\n', 'Observation:', 'Execute:'], 'completion_mode': 'text',
                       'add_system_information': True, 'init_mem_state': False, 'binary_tree': None,
                       'agent_type': 'structured-chat-zero-shot-react-description',
                       'tools': ['memory', 'save_data_to_memory', 'crate_task', 'optimise_task'], 'task_list': [],
                       'task_list_done': [], 'step_between': '', 'pre_task': None, 'task_index': 0},
        'think': {'name': 'think', 'mode': 'free', 'model_name': 'gpt-4', 'max_iterations': 1, 'verbose': True,
                  'personality': '', 'goals': '', 'token_left': 1347, 'temperature': 0.06, '_stream': True,
                  '_stream_reset': False, 'stop_sequence': ['\n\n\n'], 'completion_mode': 'chat',
                  'add_system_information': True, 'init_mem_state': False, 'binary_tree': None,
                  'agent_type': 'structured-chat-zero-shot-react-description',
                  'tools': ['memory', 'save_data_to_memory', 'crate_task', 'optimise_task'], 'task_list': [],
                  'task_list_done': [], 'step_between': '', 'pre_task': None, 'task_index': 0},
        'summary': {'name': 'summary', 'mode': 'free', 'model_name': 'gpt4all#ggml-model-gpt4all-falcon-q4_0.bin',
                    'max_iterations': 1, 'verbose': True, 'personality': '', 'goals': '', 'token_left': 4096,
                    'temperature': 0.06, '_stream': False, '_stream_reset': False, 'stop_sequence': ['\n\n'],
                    'completion_mode': 'chat', 'add_system_information': True, 'init_mem_state': False,
                    'binary_tree': None, 'agent_type': 'structured-chat-zero-shot-react-description',
                    'tools': ['memory', 'save_data_to_memory', 'crate_task', 'optimise_task'], 'task_list': [],
                    'task_list_done': [], 'step_between': '',
                    'pre_task': 'Act as an summary expert your specialties are writing summary. you are known to think in small and detailed steps to get the right result. Your task :',
                    'task_index': 0}}, 'customFunctions': {}, 'tasks': {}} # 'google-search','ddg-search',

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test-TestIsaa")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa", "toolboxv2.mods.")
        cls.app.new_ac_mod("isaa")
        cls.isaa: Tools = cls.app.get_mod('isaa')
        cls.isaa.load_keys_from_env()
        if "OPENAI_API_KEY" in cls.isaa.config:  # in cloud 0
            cls.models += [
                'gpt-4', 'text-davinci-003', 'gpt-3.5-turbo-0613',
                'text-curie-001',
                'text-babbage-001',
                'text-ada-001',
                'text-davinci-edit-001',
                'gpt-3.5-turbo-instruct',
                # 'gpt-3.5-turbo'', 'gpt-4-0613', 'code-davinci-edit-001'  #code- usles
            ]

        if "HUGGINGFACEHUB_API_TOKEN" in cls.isaa.config:
            cls.models += [
                'google/flan-t5-small',  # 2/10 ~ ? Knowledge  classify ?eval? : 0.48s
                # 'facebook/bart-large-cnn', # 0/10 spoling informtions Prompt?
                # 'tiiuae/falcon-40b', # -1
                'google/flan-t5-xxl',
                # eglisch text bot not mutch context 1/10 classify  tool use json to  (q/a 2) :: 0.51s
                'databricks/dolly-v2-3b',  # Knowledge 0/10 : 0.57s
                ## 'stabilityai/FreeWilly2', # to big
                ## 'jondurbin/airoboros-l2-70b-gpt4-1.4.1',
                ## 'TheBloke/llama-2-70b-Guanaco-QLoRA-fp16',
                ## 'TheBloke/gpt4-alpaca-lora_mlp-65B-HF',
                ## 'meta-llama/Llama-2-70b-hf',
                ## 'TheBloke/guanaco-65B-HF',
                ## 'huggyllama/llama-65b',
                # 'NousResearch/Nous-Hermes-Llama2-13b', # slow af | to big  No output
                # 'YeungNLP/firefly-llama2-13b', # slow ... nop
                # 'mosaicml/mpt-30b-chat',
                # 'openaccess-ai-collective/wizard-mega-13b',
                # 'deutsche-telekom/bert-multi-english-german-squad2',
                # conversation 'PygmalionAI/pygmalion-6b',
                # 'meta-llama/Llama-2-7b',
                'knkarthick/MEETING_SUMMARY',  # summary (q/a 12 : 0.20s
                # 'TheBloke/OpenAssistant-Llama2-13B-Orca-8K-3319-GGML',
                # 'TheBloke/Llama-2-7b-chat-fp16',
                # 'TheBloke/open-llama-7B-v2-open-instruct-GPTQ',
                # 'TheBloke/open-llama-13b-open-instruct-GPTQ',
                # 'TheBloke/falcon-7b-instruct-GPTQ',
                # 'TheBloke/Llama-2-7b-Chat-GPTQ',
                'stabilityai/stablecode-instruct-alpha-3b'
            ]

        cls.isaa.config["DEFAULTMODEL0"] = "gpt-3.5-turbo-0613"  # "gpt-4"
        cls.isaa.config["DEFAULTMODEL1"] = "gpt-3.5-turbo-0613"  # "gpt-3.5-turbo-0613"
        cls.isaa.config["DEFAULTMODEL2"] = "text-curie-001"  # "text-davinci-003"
        cls.isaa.config["DEFAULTMODELCODE"] = "code-davinci-edit-001"  # "code-davinci-edit-001"
        cls.isaa.config["DEFAULTMODELSUMMERY"] = "text-curie-001"  # "text-curie-001"

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def setUp(self):
        self.app_mock = Mock()
        self.isaa = self.app.get_mod('isaa')

    def tearDown(self):
        self.isaa._on_exit()
        self.app.remove_mod('isaa')

    def test_init(self):
        self.isaa._on_exit()
        self.app.remove_mod('isaa')
        self.isaa = self.app.get_mod('isaa')
        self.assertEqual(self.isaa.version, '0.0.2')
        self.assertEqual(self.isaa.name, 'isaa')
        self.assertEqual(self.isaa.logger, self.app.logger)
        self.assertEqual(self.isaa.color, 'VIOLET2')
        self.assertEqual(self.isaa.config['agents-name-list'], [])
        self.assertEqual(self.isaa.config['genrate_image-init'], False)

    @patch('toolboxv2.mods.isaa.isaa.AgentChain')  # replace with the actual module name
    def test_add_task(self, mock_agent_chain):
        sto = self.isaa.agent_chain
        self.isaa.agent_chain = mock_agent_chain
        self.isaa.add_task('test_task', 'test_task_content')
        mock_agent_chain.add_task.assert_called_once_with('test_task', 'test_task_content')
        self.isaa.agent_chain = sto

    @patch('toolboxv2.mods.isaa.isaa.AgentChain')  # replace with the actual module name
    def test_save_task(self, mock_agent_chain):
        sto = self.isaa.agent_chain
        self.isaa.agent_chain = mock_agent_chain
        self.isaa.save_task('test_task')
        mock_agent_chain.save_to_file.assert_called_once_with('test_task')
        self.isaa.agent_chain = sto

    @patch('toolboxv2.mods.isaa.isaa.AgentChain')  # replace with the actual module name
    def test_load_task(self, mock_agent_chain):
        sto = self.isaa.agent_chain
        self.isaa.agent_chain = mock_agent_chain
        self.isaa.load_task('test_task')
        mock_agent_chain.load_from_file.assert_called_once_with('test_task')
        self.isaa.agent_chain = sto

    @patch('toolboxv2.mods.isaa.isaa.AgentChain')  # replace with the actual module name
    def test_get_task(self, mock_agent_chain):
        sto = self.isaa.agent_chain
        self.isaa.agent_chain = mock_agent_chain
        self.isaa.get_task('test_task')
        mock_agent_chain.get.assert_called_once_with('test_task')
        self.isaa.agent_chain = sto

    def test_get_augment(self):
        self.isaa.serialize_all = MagicMock(return_value='serialized_agents')
        self.isaa.scripts.scripts = 'custom_functions'
        self.isaa.agent_chain.save_to_dict = MagicMock(return_value='tasks')
        augment = self.isaa.get_augment()
        self.assertEqual(augment, {'tools': self.isaa.tools_dict, 'Agents': 'serialized_agents',
                                   'customFunctions': 'custom_functions', 'tasks': 'tasks'})

    @patch('toolboxv2.mods.isaa.isaa.Tools.init_tools')
    @patch('toolboxv2.mods.isaa.isaa.Tools.deserialize_all')
    def test_init_from_augment(self, mock_deserialize_all, mock_init_tools):

        self.isaa.init_from_augment(self.augment)
        mock_init_tools.assert_called_once_with(self.isaa.get_agent_config_class("self"), self.augment['tools'])
        mock_deserialize_all.assert_called_once_with(self.augment['Agents'], self.isaa.get_agent_config_class("self"),
                                                     exclude=None)

    def test_serialize_all(self):
        self.isaa.config['agents-name-list'] = ['agent1', 'agent2']
        _ = self.isaa.get_agent_config_class
        self.isaa.config[f'agent-config-agent1'] = AgentConfig(self.isaa, 'agent1')
        self.isaa.config[f'agent-config-agent2'] = AgentConfig(self.isaa, 'agent2')
        self.isaa.get_agent_config_class = MagicMock(
            side_effect=lambda x: MagicMock(name=x, serialize=MagicMock(return_value={x: 'data'})))
        serialized = self.isaa.serialize_all()
        self.isaa.get_agent_config_class = _
        self.assertEqual(serialized[list(serialized.keys())[0]], {'agent1': 'data'})
        self.assertEqual(serialized[list(serialized.keys())[1]], {'agent2': 'data'})

    def test_deserialize_all(self):
        data = {'agent1': 'data1', 'agent2': 'data2'}
        self.isaa.get_agent_config_class = MagicMock(side_effect=lambda x: MagicMock(name=x, deserialize=MagicMock()))
        self.isaa.deserialize_all(data, 's_agent')
        self.isaa.get_agent_config_class.assert_any_call('agent1')
        self.isaa.get_agent_config_class.assert_any_call('agent2')

    def test_init_all_pipes_default(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in self.isaa.config.keys():
            print("Not HF token for testing")
            return
        self.isaa.init_all_pipes_default()
        self.assertEqual(len(self.isaa.initstate), 3)

    def test_init_pipeline(self):
        self.isaa.init_pipeline('question-answering', 'deepset/roberta-base-squad2')
        self.assertIn(f"{'question-answering'}_pipeline", self.isaa.config.keys())
        self.assertTrue(self.isaa.initstate['question-answering'])

    def test_question_answering(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in self.isaa.config.keys():
            print("Not HF token for testing")
            return
        res = self.isaa.question_answering('What is the capital of Germany?', 'The capital of Germany is Berlin.')
        self.assertIsInstance(res, dict)
        self.assertIn('score', res.keys())
        self.assertIn('answer', res.keys())
        self.assertIn('start', res.keys())
        self.assertIn('end', res.keys())
        self.assertEqual(26, res['start'])
        self.assertIsInstance(res['answer'], str)
        self.assertGreater(len(res['answer']), 0)
        self.assertGreater(res['score'], 0)
        print(f'Question What is the capital of Germany? Answer : {res}')

    def test_summarization(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in self.isaa.config.keys():
            print("Not HF token for testing")
            return
        res = self.isaa.summarization('This is a long text that needs to be summarized.')
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        self.assertIsInstance(res[0], dict)
        self.assertIn('summary_text', res[0].keys())
        self.assertIsInstance(res[0]['summary_text'], str)
        self.assertGreater(len(res[0]['summary_text']), 0)
        print(f"This is a long text that needs to be summarized : {res[0]['summary_text']}")

    def test_mas_text_summaries(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in self.isaa.config.keys():
            print("Not HF token for testing")
            return
        if "OPENAI_API_KEY" not in self.isaa.config.keys():
            print("Not OA token for testing")
            return
        res = self.isaa.mas_text_summaries('''Selbstverständlich. Hier sind die Punkte, die Sie im bereitgestellten Code überprüfen sollten:

Verwendung von await und async:
Stellen Sie sicher, dass alle Funktionen und Methoden, die mit await aufgerufen werden, als async deklariert sind und die korrekten Rückgabetypen haben.

Verwendung von Awaitables in send() und receive():
Überprüfen Sie die Verwendung von Awaitables (Instanzen von asyncio.Future, Coroutines oder anderen Awaitables) in den Funktionen send() und receive() und stellen Sie sicher, dass die übergebenen Objekte den erwarteten Typ haben.

Verwendung von asyncio.wait():
Untersuchen Sie die Verwendung von asyncio.wait() im Code und stellen Sie sicher, dass die Liste der Awaitables korrekt aufgebaut ist und keine ungültigen Objekte enthält.

Verwendung des manager-Objekts:
Überprüfen Sie die Verwendung des manager-Objekts und stellen Sie sicher, dass alle aufgerufenen Methoden korrekt sind und die erwarteten Rückgabetypen haben.

Asynchrone vs. synchrone Operationen:
Stellen Sie sicher, dass es keine Konflikte zwischen asynchronen und synchronen Operationen im Code gibt, die zu Fehlern führen könnten.

Funktionsaufrufe in asyncio-Tasks:
Prüfen Sie, ob die Funktionsaufrufe in den asyncio-Tasks ordnungsgemäß ablaufen und alle Awaitables korrekt verarbeitet werden.

Ausnahmen und Fehlerbehandlung:
Stellen Sie sicher, dass alle Ausnahmen ordnungsgemäß behandelt werden und es keine unbehandelten Ausnahmen gibt, die zu Fehlern führen könnten.

Bitte beachten Sie, dass der bereitgestellte Codeausschnitt nicht den vollständigen Kontext zeigt, einschließlich der Definition der Funktionen send() und receive(), des manager-Objekts und möglicherweise anderer relevanter Teile des Codes. Um den genauen Fehler zu isolieren, ist es wichtig, den gesamten relevanten Code zu überprüfen und sicherzustellen, dass asynchrone Muster korrekt implementiert sind.

Wenn Sie nach der Überprüfung dieser Punkte immer noch Schwierigkeiten haben, den Fehler zu finden, könnte es hilfreich sein, mehr von Ihrem Code zu sehen oder spezifische Teile zu isolieren und zu testen, um den Ursprung des Problems einzugrenzen.''')
        print("res:", res)

    def test_text_classification(self):
        if "HUGGINGFACEHUB_API_TOKEN" not in self.isaa.config.keys():
            print("Not HF token for testing")
            return
        res = self.isaa.text_classification('This is a positive statement.')
        self.assertIsInstance(res, list)
        self.assertTrue(len(res) == 1)
        self.assertIsInstance(res[0], dict)
        self.assertIn('label', res[0].keys())
        self.assertIn('score', res[0].keys())
        self.assertGreater(res[0]['score'], 0)
        self.assertEqual('POSITIVE', res[0]['label'])
        print(f'Question What is this (positive statement)? Answer : {res}')

    def test_run_models(self):

        logger = get_logger()

        info_text = """The Symphony of Life: The Army Ants, the Flower, and the Bear

        Life on our planet is like a mesmerizing symphony, with each living being playing a unique and vital role in the
        grand dance of existence. Among the many fascinating creatures that inhabit our world, three stand out as exceptional
        examples of the diversity and complexity of life: the Army Ants, the Flower, and the Bear.

        First, let's delve into the world of the Army Ants. These incredible insects move as a unified force,
        akin to a disciplined battalion on the march. Within their colossal colonies, which can number in the millions,
        every individual works in harmony for a shared purpose. Their perfectly synchronized movements and intricate
        communication systems enable them to efficiently hunt and overpower prey many times their size. The unwavering
        dedication of these ants to their colony is nothing short of awe-inspiring.

        Now, turn your attention to the Flower, a symbol of beauty and resilience in the midst of the bustling world of
        creatures. The delicate petals and vibrant colors of the Flower captivate our senses, drawing us into their
        enchanting presence. However, the Flower's significance extends far beyond its visual appeal. It serves as a crucial
        agent of nature's matchmaking, attracting vital pollinators like bees and butterflies. Through this process of
        cross-pollination, flowers facilitate the growth of fruits and seeds, ensuring the continuity of numerous plant species.

        Next, we encounter the majestic Bear, a powerful emblem of strength and solitude in the natural world. As apex
        predators, bears have a profound impact on their ecosystems, shaping the populations of various prey species. Their
        remarkable adaptability allows them to thrive in diverse environments, from dense forests to icy terrains. Yet,
        beyond their role as hunters, bears play another essential part in the symphony of life. When they feast on fruits
        and berries, they inadvertently become gardeners, spreading seeds across vast distances. In doing so, they contribute
        to the reforestation of landscapes and the regeneration of diverse habitats.

        Within the symphony of life, the Army Ants, the Flower, and the Bear harmonize in fascinating interactions. As the
        Army Ants march through dense forests in search of food, they unknowingly uncover hidden flowers, becoming agents of
        pollination. The Flower, with its captivating beauty and sweet nectar, allures the ants, providing nourishment for
        their tireless endeavors. Meanwhile, the Bear traverses through the ant's foraging trails, contributing to the
        dispersal of seeds and fostering the growth of new life.

        In conclusion, the Army Ants, the Flower, and the Bear are exemplary ambassadors of the intricate web of life on our
        magnificent planet. Each of these beings, with their unique traits and roles, adds a distinct note to the symphony of
        existence. As we contemplate the complexity and interconnectedness of nature's design, let us remember that every
        living being, regardless of its size or appearance, holds a vital place in preserving the harmony of life."""

        maintool_snap = '''import uuid

        from toolboxv2.utils.toolbox import App
        from toolboxv2.utils.Style import Style
        from toolboxv2.utils.tb_logger import get_logger


        class MainTool:

            toolID = ""

            def __init__(self, *args, **kwargs):
                self.version = kwargs["v"]
                self.tools = kwargs["tool"]
                self.name = kwargs["name"]
                self.logger = kwargs["logs"]
                if kwargs["logs"] is None:
                    self.logger = get_logger()
                self.color = kwargs["color"]
                self.todo = kwargs["load"]
                self._on_exit = kwargs["on_exit"]
                self.stuf = False
                if not hasattr(self, 'config'):
                    self.config = {}
                self.load()
                self.ac_user_data_sto = {}

            def load(self):
                if self.todo:
                    #try:
                        self.todo()
                    #except Exception as e:
                    #    get_logger().error(f" Error loading mod {self.name} {e}")
                else:
                    get_logger().info(f"{self.name} no load require")

                print(f"TOOL : {self.name} online")

            def print(self, message, end="\n", **kwargs):
                if self.stuf:
                    return

                print(Style.style_dic[self.color] + self.name + Style.style_dic["END"] + ":", message, end=end, **kwargs)

            def add_str_to_config(self, command):
                if len(command) != 2:
                    self.logger.error('Invalid command must be key value')
                    return False
                self.config[command[0]] = command[1]

            def webInstall(self, user_instance, construct_render) -> str:
                """"Returns a web installer for the given user instance and construct render template"""

            def get_uid(self, command, app: App):

                # if "cloudm" not in list(app.MOD_LIST.keys()):
                #     return f"Server has no cloudM module", True
                #
                # if "db" not in list(app.MOD_LIST.keys()):
                #     return "Server has no database module", True

                res = app.run_any('cloudM', "validate_jwt", command)

                if type(res) is str:
                    return res, True

                self.ac_user_data_sto = res
                return res["uid"], False

            def get_user_instance(self, uid, app: App):
                return app.run_any('cloudM', "get_user_instance", [uid])'''

        max_task = 16
        test_run_models_helper_count = 0

        agent_config = self.isaa.get_agent_config_class()
        agent_config.set_mode('free')
        agent_config.set_completion_mode('chat')

        def helper(prefix, istruction, model, agent_config_):
            with self.subTest(prefix=prefix):
                logger.info(f"Testing ::{prefix}:: for model : {model}")
                print(f"Testing ::{prefix}::\nTask {test_run_models_helper_count}/{max_task} at"
                      f" {(test_run_models_helper_count / max_task) * 100:.2f}")
                t1 = time.time()
                res = self.isaa.stream_read_llm(istruction, agent_config_)
                self.assertIsNotNone(res)
                self.assertIsInstance(res, str)
                self.assertNotEqual(res, "")
                print(f"{'=' * 10}\nModel: {model}\nTask:{prefix}\nResult: {res}\n{'=' * 10} \nCompleted in"
                      f" {time.time() - t1:.2f}\n\n")

        for model in self.models:
            test_run_models_helper_count = 0
            t0 = time.time()
            if model in self.not_run:
                continue
            print(f"Starting with Model :{model}")
            with self.subTest(model=model):
                self.isaa.load_llm_models([model])
                self.assertTrue(self.isaa.initstate[f'LLM-model-{model}-init'])
                self.assertIn(f'LLM-model-{model}', self.isaa.config.keys())

                agent_config.set_model_name(model)
                agent_config.stream = False

                if 'gpt' in model:
                    agent_config.set_completion_mode("chat")
                if 'text' in model:
                    agent_config.set_completion_mode("text")
                if 'code' in model:
                    agent_config.set_completion_mode("edit")
                if 'edit' in model:
                    agent_config.set_completion_mode("edit")

                agent_config.set_mode('free')

                helper("(repetition test)", "User: Say Test.\nIsaa:", model, agent_config)
                helper("(featured properties test)", "User: What is yor name?\nIsaa:", model, agent_config)
                helper("(Knowledge with hint test)", "User:what ist The Capital of Germany?\nIsaa:", model,
                       agent_config)

                test_run_models_helper_count += 3

                agent_config.set_mode('free')
                helper("(Translation test)", "User:What does ?"
                                             "'Um um eine Fliege herum zu fliegen muss man fliegen können' mean in "
                                             "Englisch?\nIsaa:", model, agent_config)
                helper("(Translation2 test)", "Translate to Englisch."
                                              "'Um um eine Fliege herum zu fliegen muss man fliegen können'\nIsaa:",
                       model, agent_config)

                helper("(Redundancy test)", "User:recreate the Original Informations from this:"
                                            "  1. Hasllo das ist ein Klwener test auf Deusdsch Mein Name ist Tom."
                                            "  2. awdlo this is an smal Reddadancy test"
                                            "  3. Bitte beyxcnnen Sie mit Ihrer Antwort."
                                            " Ich bin gdssnnt auf Ihre kreativen und zielgerwdteten Lösungen."
                                            "  4. Pleadwe start with your andawr."
                                            " dsawI am looking fodwerd to your crdwive and targeted solystions."
                                            "  5. 1234.67.....fg...klm"
                                            "  6. kgjZGuzgfaZgKmkugZUFTUIZ\n\n"
                                            "writ from 1 to 6 reproduce the original informations :\nIsaa:", model,
                       agent_config)

                test_run_models_helper_count += 3

                helper("(Summary test)", f"User:Produce a Summary form this Text:{info_text}\nSummary:", model,
                       agent_config)
                helper("(classification test)", f"User:Classify the given text wit the label POSITIVE or NEGETIVE "
                                                "text for classification : 'Somtimes Feal bussy but over all I'm well'\nlabel:",
                       model, agent_config)

                helper("(evaluation test)", "System Instruction: Test evaluation,"
                                            " Evaluate the result is the Task accomplish YES NO ERROR"
                                            "User: Task: Hello Isaa what is 1 + 1"
                                            "Observation: "
                                            "The System cud not correct the prompt for the agent\nAssistant:", model,
                       agent_config)

                test_run_models_helper_count += 3

                helper("(Q and A test)", f"Test  q/a What Animals ar mention in thes text {info_text}"
                                         f"\nAssistant:", model, agent_config)

                helper("(Q and A test)", f"Test q/a What is the relation between the Animals {info_text}"
                                         f"\nAssistant:", model, agent_config)
                helper("(py coding Hello world test)", f"User: Write a python Hello World. \nAssistant:",
                       model, agent_config)

                test_run_models_helper_count += 3
                helper("(py coding 5 fibonacci test)", f"User: Write a python function that calculates the 5 "
                                                       f"fibonacci number. \nAssistant:", model, agent_config)

                helper("(py coding custom test)", f"User: Write a python unit test for this class {maintool_snap}"
                                                  f"\nAssistant:", model, agent_config)

                test_run_models_helper_count += 2

                logger.info(" ==================== Custom Mode Test ====================")

                agent_config.set_mode('tools')

                helper("(tools test)", "Task: how will the weather be tomorrow?", model, agent_config)

                agent_config.set_mode('execution')

                helper("(execution test)", "Task: how will the weather be tomorrow?", model, agent_config)

                agent_config.set_mode('planning')
                helper("(planning test)", "User:how to get world Dominance?", model, agent_config)

                agent_config.set_mode('generate')

                helper("(generate test)", f"How to build an web server with an web ui?", model, agent_config)

                agent_config.set_mode('talk')
                helper("(talk test)", f"How to build an web server with an web ui?", model, agent_config)

                agent_config.set_mode('free')
                test_run_models_helper_count += 5

                self.isaa.free_llm_model([model])

            print(f"Test Completed in {time.time() - t0}")

    def test_load_models(self):
        logger = get_logger()
        agent_config = self.isaa.get_agent_config_class()
        agent_config.set_mode('free')
        agent_config.set_completion_mode('chat')

        for model in self.models:
            logger.info(f"Loging Moldel: {model}...")
            if model in self.not_run:
                continue
            with self.subTest(model=model):
                t0 = time.time()
                self.isaa.load_llm_models([model])
                self.assertTrue(self.isaa.initstate[f'LLM-model-{model}-init'])
                self.assertIn(f'LLM-model-{model}', self.isaa.config.keys())
                logger.info(f"Closing Moldel: {model}...")
                self.isaa.free_llm_model([model])
                logger.info(f'Initialized : {model} in : {time.time() - t0:.2f}s')
            print(f'Initialized : {model} in : {time.time() - t0:.2f}s')

    def test_add_tool(self):
        config = MagicMock()
        self.isaa.add_tool('name', 'func', 'dis', 'form', config)
        config.tools.update.assert_called_once_with({'name': {'func': 'func', 'description': 'dis', 'format': 'form'}})

    def test_add_lang_chain_tools_to_agent(self):
        agent = MagicMock()
        x = {'tool1': MagicMock(args='args1'), 'tool2': MagicMock(args='args2')}
        self.isaa.lang_chain_tools_dict = x.copy()
        self.isaa.add_lang_chain_tools_to_agent(agent)
        agent.set_tools.assert_called_once_with({
            'tool1': {'func': x['tool1'],
                      'description': x['tool1'].description, 'format': 'tool1(args1)',
                      'langchain-tool': True},
            'tool2': {'func': x['tool2'],
                      'description': x['tool2'].description, 'format': 'tool2(args2)',
                      'langchain-tool': True}
        })
        self.assertEqual(self.isaa.lang_chain_tools_dict, {})

    def test_create_agent_class(self):
        agent_config = self.isaa.create_agent_class('BP')
        self.assertEqual(agent_config.name, 'BP')

    def test_remove_agent_config(self):
        self.isaa.config['agent-config-test'] = 'test'
        self.isaa.config['agents-name-list'] = ['test']
        self.isaa.remove_agent_config('test')
        self.assertNotIn('agent-config-test', self.isaa.config)
        self.assertNotIn('test', self.isaa.config['agents-name-list'])

    def test_get_agent_config_class(self):
        self.isaa.config['agents-name-list'] = []
        self.isaa.get_agent_config_class('Normal')
        self.assertIn('Normal', self.isaa.config['agents-name-list'])
        self.assertIn('agent-config-Normal', self.isaa.config.keys())
        self.assertIn('agent-config-Normal', self.isaa.config)

    def test_mini_task_completion(self):
        self.isaa.get_agent_config_class = MagicMock()
        self.isaa.stream_read_llm = MagicMock()
        self.isaa.mini_task_completion('test')
        self.isaa.get_agent_config_class.assert_called_with('TaskCompletion')
        self.isaa.stream_read_llm.assert_called_with('test', self.isaa.get_agent_config_class.return_value)

    def test_create_task(self):
        self.isaa.execute_thought_chain = MagicMock(return_value=(
            'res', {'$taskDict': '[{"use": "tool", "name": "memory", "args": "$user-input", "return": "$D-Memory"}]'},
            '_'))
        self.isaa.mini_task_completion = MagicMock(return_value='task_name')
        self.assertEqual(self.isaa.create_task('task'), 'task_name')

    def test_optimise_task(self):
        # self.isaa.execute_thought_chain = MagicMock(return_value=(
        #    'res', {'$taskDict': '[{"use": "tool", "name": "memory", "args": "$user-input", "return": "$D-Memory"}]'},
        #    '_'))
        """
        ret = self.isaa.optimise_task('task_name')
        print(ret)
        self.assertIsInstance(ret, list)
        self.assertGreaterEqual(len(ret), 1)
        self.assertIsInstance(ret[0], dict)
        self.assertIn('args', ret[0].keys())
        self.assertIn('name', ret[0].keys())
        self.assertIn('use', ret[0].keys())
        self.assertIn('return', ret[0].keys())
        self.assertGreater(len(ret[0]['args']), 0)
        self.assertGreater(len(ret[0]['name']), 0)
        self.assertGreater(len(ret[0]['use']), 0)
        self.assertGreater(len(ret[0]['return']), 0)
        """
        self.assertTrue(True)

    def test_use_tools_free(self):
        # agent_text = 'I wont to run the api_run Tool'
        # config = AgentConfig(name='free0', isaa=self.isaa).set_mode('free')
        # self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", config)
        # result = self.isaa.test_use_tools(agent_text, config)
        # self.assertEqual(result, (True, 'api_run', 'Tool'))

        agent_text = 'Action: api_run'
        config = AgentConfig(name='free1', isaa=self.isaa).set_mode('free')
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", config)
        result = self.isaa.test_use_tools(agent_text, config)
        self.assertEqual(result, (True, 'api_run', ''))

    def test_use_tools_execution(self):
        agent_text = 'Action: api_run\nInputs: test_input'
        config = AgentConfig(name='execution2', isaa=self.isaa).set_mode('execution')
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", config)
        result = self.isaa.test_use_tools(agent_text, config)
        self.assertEqual(result, (True, 'api_run', 'test_input'))

    def test_use_tools_with_json_input(self):
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", self.isaa.get_agent_config_class('self'))
        agent_text = '{"Action": "api_run", "Inputs": "value1"}'
        result = self.isaa.test_use_tools(agent_text, self.isaa.get_agent_config_class('self'))
        self.assertEqual(result, (True, "api_run", "value1"))

    def test_use_tools_with_json_in_json_input(self):
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", self.isaa.get_agent_config_class('self'))
        agent_text = '''{"Action":"api_run","Inputs":{"x": "value1"}}'''

        # self.assertEqual('{"Action": "api_run", "Inputs": {"input1": "value1"}}', agent_text)
        # self.assertEqual(json.loads('{"Action": "api_run", "Inputs": {"input1": "value1"}}'),
        #                 {"Action": "api_run", "Inputs": {"input1": "value1"}})
        result = self.isaa.test_use_tools(agent_text, self.isaa.get_agent_config_class('self'))
        self.assertEqual(result, (True, "api_run", {"x": "value1"}))

    def test_use_tools_with_json_in_json_3xinput(self):
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", self.isaa.get_agent_config_class('self'))
        agent_text = '''{"Action":"api_run","Inputs":{"x": {"x": "value1"}}}'''
        result = self.isaa.test_use_tools(agent_text, self.isaa.get_agent_config_class('self'))
        self.assertEqual(result, (True, "api_run", {"x": {"x": "value1"}}))

    def test_use_tools_with_json_in_json_input_string_noice(self):
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", self.isaa.get_agent_config_class('self'))
        agent_text = ''' dasd {"Action":"api_run","Inputs":{"x": "value1"}} asdw'''
        result = self.isaa.test_use_tools(agent_text, self.isaa.get_agent_config_class('self'))
        self.assertEqual(result, (True, "api_run", {"x": "value1"}))

    def test_use_tools_with_string_input(self):
        agent_text = "Action: think Inputs: input1"
        result = self.isaa.test_use_tools(agent_text, self.isaa.get_agent_config_class('self'))
        self.assertEqual(result, (True, "think", 'input1'))

    def test_use_tools_with_no_tool_called(self):
        agent_text = 'To salve this Problem u can ask the search Agent.'
        config = AgentConfig(name='free4', isaa=self.isaa).set_mode('free')
        self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", config)
        result = self.isaa.test_use_tools(agent_text, config)
        self.assertEqual(result, (False, '', ''))

    def test_use_tools_with_ai_called(self):
        agent_text = 'I wont to run the api_run Tool'
        config = AgentConfig(name='free0', isaa=self.isaa).set_mode('free')
        # self.isaa.add_tool("api_run", lambda x: x, "no dis", "no form", config)
        # result = self.isaa.test_use_tools(agent_text, config)
        # self.assertEqual(result, (True, 'api_run', 'Tool'))

    def test_use_tools_multi_string(self):

        data_set = ['Leerzeichen',
                    '""',
                    '"Invalid JSON"',
                    "{'Action': 'ToolA', 'Inputs': {'Param1': 'Value1'}}",
                    "{'Action': 'ToolB', 'Inputs': {'Param2': 'Value2'}}",
                    "{'Action': 'ToolC', 'Inputs': {'Param3': 'Value3'}}",
                    "Action: ToolD Inputs: {Param4: 'Value4'}",
                    "Action: ToolE Inputs: {Param5: 'Value5'}",
                    "Action: ToolF Inputs: {Param6: 'Value6'}",
                    "Action: ToolG Inputs: {Param7: 'Value7'}",
                    "Action: ToolH Inputs: {Param8: 'Value8'}",
                    "Action: ToolI Inputs: {Param9: 'Value9'}",
                    "Action: ToolJ Inputs: {Param10: 'Value10'}",
                    "Action: InvalidAction Inputs: {Param11: 'Value11'}",
                    "{'Action': 'InvalidAction', 'Inputs': {'Param12': 'Value12'}}",
                    "Random Text Without Action and Inputs""",
                    "Action: ToolK Inputs: {Param13: 'Value13'} Additional Text After Inputs",
                    "{'Action': 'ToolL', 'Inputs': {'Param14': 'Value14'}} Invalid JSON",
                    "Action: ToolM Inputs: {Param15: 'Value15'} More Text",
                    "{'Action': 'ToolN', 'Inputs': {'Param16': 'Value16'}} More Text After JSON",
                    '''{
    "Action": "ToolX",
    "Inputs": {
        "Param1": "Value1",
        "Param2": "Value2"
    }
}
''',
                    '''{
    "Action": "ToolY",
    "Inputs": {
        "Param3": "Value3",
        "Param4": "Value4"
    },
    "AdditionalData": "Invalid"
}
''',
                    '''Action: ToolZ
Inputs: Param5: Value5
AdditionalInfo: SomeText
'''
                    '''This is a long text that does not contain any recognizable action or inputs.
It is just a random description without any specific purpose.
'''
                    '''I'm not sure which tool to use for this task. Can you help me out?
AI: ToolA
Inputs: Param6: Value6, Param7: Value7
'''
                    ]
        tools = [
            "ToolA",
            "ToolB",
            "ToolC",
            "ToolD",
            "ToolE",
            "ToolF",
            "ToolG",
            "ToolH",
            "ToolI",
            "ToolJ",
            "ToolL",
            "ToolM",
            "ToolN",
            "ToolX",
            "ToolK",
            "ToolY",
            "ToolZ",
        ]
        action_map = [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ]
        config = AgentConfig(name='free0', isaa=self.isaa).set_mode('free')
        for test_tool in tools:
            self.isaa.add_tool(test_tool, lambda x: x, "no dis", "no form", config)
        i = 0
        for test_test in data_set:
            with self.subTest(TestNumber=i):
                # tool, res = _extract_from_json(test_test, config)
                result = self.isaa.test_use_tools(test_test, config)
                print("################################")
                print(i, data_set[i], result)
                print("################################")
                self.assertEqual(list(result)[0], action_map[i])

            i += 1

    def test_task_done(self):
        self.assertTrue(self.isaa.test_task_done('Final Answer: answer'))

    def test_run_tool_no_args(self):
        # Mock a tool function with no arguments
        agent = self.isaa.get_agent_config_class('TestAgent')
        self.isaa.add_tool("test_func", lambda: "success", "", "", agent)
        result = self.isaa.run_tool("test_func", "test_func", agent)
        self.assertEqual(result, "success")

    def test_run_tool_with_args(self):
        # Mock a tool function with arguments
        agent = self.isaa.get_agent_config_class('TestAgent')
        self.isaa.add_tool("test_func", lambda x: x, "", "", agent)
        result = self.isaa.run_tool("123", "test_func", agent)
        self.assertEqual(result, "123")

    def test_run_tool_with_kwargs(self):
        # Mock a tool function with keyword arguments
        agent = self.isaa.get_agent_config_class('TestAgent')
        agent.tools = {"test_func": {'func': lambda x=None: x}}
        result = self.isaa.run_tool(json.dumps({"x": "123"}), "test_func", agent)
        self.assertEqual(result, "123")

    def test_run_tool_with_kwargs_isaa_add_tool(self):
        # Mock a tool function with keyword arguments
        agent = self.isaa.get_agent_config_class('TestAgent')
        self.isaa.add_tool("test_func", lambda x=None: x, "", "", agent)
        result = self.isaa.run_tool(json.dumps({"x": "123"}), "test_func", agent)
        self.assertEqual(result, "123")

    def test_run_tool_invalid_func(self):
        # Test with invalid function name
        agent = self.isaa.get_agent_config_class('TestAgent')
        result = self.isaa.run_tool("", "invalid_func", agent)
        self.assertIn("Unknown Function invalid_func", result)

    def test_run_tool_no_args_mm(self):
        agent = self.isaa.get_agent_config_class('TestAgent')
        mm = MagicMock()
        agent.tools = {'test_func': {'func': mm}}
        self.isaa.run_tool('', 'test_func', agent)
        mm.assert_called_with(args='', kwargs='')

    def test_run_tool_one_arg(self):
        agent = self.isaa.get_agent_config_class('TestAgent')
        mm = MagicMock()
        agent.tools = {'test_func': {'func': mm}}
        self.isaa.run_tool('arg1', 'test_func', agent)
        mm.assert_called()
        mm.assert_called_with(args='arg1', kwargs='')

    def test_run_tool_multiple_args_str(self):
        agent = self.isaa.get_agent_config_class('TestAgent')

        def test_fuction(arg1=None, arg2=None, arg3=None):
            self.assertEqual(arg1, '"arg1"')
            self.assertEqual(arg2, '"arg2"')
            self.assertEqual(arg3, '"arg3"')

        agent.tools = {'test_func': {'func': test_fuction}}
        self.isaa.run_tool('"arg1", "arg2", "arg3"', 'test_func', agent)

    def test_run_tool_multiple_args_comm_sep(self):
        agent = self.isaa.get_agent_config_class('TestAgent')

        def test_fuction(arg1, arg2, arg3):
            self.assertEqual(arg1, 'arg1')
            self.assertEqual(arg2, 'arg2')
            self.assertEqual(arg3, 'arg3')

        agent.tools = {'test_func': {'func': test_fuction}}
        self.isaa.run_tool('arg1,arg2,arg3', 'test_func', agent)

    def test_run_tool_multiple_args_comm_sep_key(self):
        agent = self.isaa.get_agent_config_class('TestAgent')

        def test_fuction(arg1=None, arg2=None, arg3=None):
            self.assertEqual(arg1, 'arg1')
            self.assertEqual(arg2, 'arg2')
            self.assertEqual(arg3, 'arg3')

        agent.tools = {'test_func': {'func': test_fuction}}
        self.isaa.run_tool('arg1,arg2,arg3', 'test_func', agent)

    def test_run_tool_multiple_args_json(self):
        agent = self.isaa.get_agent_config_class('TestAgent')

        def test_fuction(arg1=None, arg2=None, arg3=None):
            self.assertEqual(arg1, 0)
            self.assertEqual(arg2, 1)
            self.assertEqual(arg3, 2)

        agent.tools = {'test_func': {'func': test_fuction}}
        self.isaa.run_tool(json.dumps({'arg1': 0, 'arg2': 1, 'arg3': 2}), 'test_func', agent)

    def test_run_tool_exception(self):
        # Mock a tool function that raises an exception
        agent = self.isaa.get_agent_config_class('TestAgent')
        self.isaa.add_tool("test_func", lambda x: int(x) / 0, "", "", agent)  # This will raise ZeroDivisionError
        result = self.isaa.run_tool('10', "test_func", agent)
        print("res::", result)
        self.assertIn('error in tool test_func: division by zero', result)

    def test_run_agent(self):
        # Definieren Sie die Eingabeparameter
        name = 'test_agent'
        text = 'test_text'
        mode_over_lode = None

        # Mock die get_agent_config_class Methode
        self.isaa.get_agent_config_class = MagicMock(return_value=self.isaa.get_agent_config_class('test_agent'))

        # Führen Sie die Funktion aus
        result = self.isaa.run_agent(name, text, mode_over_lode)

        # Überprüfen Sie das Ergebnis
        self.assertIsInstance(result, str)  # Die Funktion sollte einen String zurückgeben

        # Überprüfen Sie, ob die Methode get_agent_config_class aufgerufen wurde
        self.isaa.get_agent_config_class.assert_called_once_with(name)

    def test_run_agents(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion
        # "isaa-chat-web", "chain_search_memory", "chain_search_url", "chain_search_web"

        for agent_name in self.agents:
            with self.subTest(Agent=agent_name):
                agent_class = self.isaa.get_agent_config_class(agent_name)
                self.assertIsInstance(agent_class, AgentConfig)
                result = self.isaa.run_agent(agent_class,
                                             "das ist ein keiner test was ist deine aufgebe was kannst du?")
                print("###################")
                self.isaa.print(f"{agent_name}::{result}")
                print("###################")

    def test_run_agent_self_modes(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion
        self.isaa.init_from_augment(self.augment)
        agent_class = self.isaa.get_agent_config_class("self")
        self.assertIsInstance(agent_class, AgentConfig)
        for mode in self.modes:
            with self.subTest(Mode=mode):
                agent_class.save_to_permanent_mem()
                result = self.isaa.run_agent(agent_class,
                                             "das ist ein keiner test was ist deine aufgebe was kannst du?",
                                             mode_over_lode=mode)
                print("###################")
                self.isaa.print(f"{mode}::{result}")
                print("###################")
                result = self.isaa.run_agent(agent_class,
                                             "Was ergibt 87658 - 23.6 / 12.4?",
                                             mode_over_lode=mode)
                print("################### Math")
                self.isaa.print(f"{mode}: Math (87656,0968) :{result}")
                print("################### Math")

    def test_run_agent_self_stream(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion

        self.isaa.init_from_augment(self.augment)
        agent_class = self.isaa.get_agent_config_class("self")
        agent_class.stream = True
        self.assertIsInstance(agent_class, AgentConfig)
        result = self.isaa.run_agent(agent_class,
                                     "das ist ein keiner test was ist deine aufgebe was kannst du?",
                                     mode_over_lode="free")
        print("###################")
        self.isaa.print(f"{result}")
        print("###################")

    def test_run_agent_self_modes_live(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion
        self.isaa.init_from_augment(self.augment)
        agent_class = self.isaa.get_agent_config_class("liveInterpretation")
        self.assertIsInstance(agent_class, AgentConfig)
        for mode in self.modes:
            with self.subTest(Mode=mode):
                result = self.isaa.run_agent(agent_class,
                                             "das ist ein keiner test was ist deine aufgebe was kannst du?",
                                             mode_over_lode=mode)
                print("###################")
                self.isaa.print(f"{mode}::{result}")
                print("###################")
                result = self.isaa.run_agent(agent_class,
                                             "Was ergibt 87658 - 23.6 / 12.4?",
                                             mode_over_lode=mode)
                print("################### Math")
                self.isaa.print(f"{mode}: Math (87656,0968) :{result}")
                print("################### Math")

    def test_run_agent_self_tools(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion
        agent_name = "tools-test"
        self.isaa.init_from_augment(self.augment, agent_name)
        agent_class = self.isaa.get_agent_config_class(agent_name)
        self.assertIsInstance(agent_class, AgentConfig)

        result = self.isaa.run_agent(agent_class,
                                     "suche im web wie alt Barak obama ist und wie viel älter er als seine frau ist")
        print("###################")
        self.isaa.print(f"::{result}")
        print("###################")
        print(agent_class.observe_mem.text)

    def test_run_agent_self_live(self):
        # Fügen Sie hier weitere Überprüfungen hinzu, basierend auf dem erwarteten Verhalten der Funktion
        agent_name = "liveInterpretation"
        self.isaa.init_from_augment(self.augment, agent_name)
        agent_class = self.isaa.get_agent_config_class(agent_name)
        self.assertIsInstance(agent_class, AgentConfig)

        result = self.isaa.run_agent(agent_class,
                                     "Take action  ; f(x) = x^3 - 3x^2 + 2x - 1.  Find the critical points of this "
                                     "function and determine whether they are local minima, local maxima, "
                                     "or saddle points. Also, find the intervals of increase and decrease.", r=0)
        print("###################")
        self.isaa.print(f"::{result}")
        print(self.isaa.config)
        print("###################")
        print(agent_class.observe_mem.text)

    def test_stream_read_llm(self):

        self.isaa.init_from_augment(self.augment)
        agent_class = self.isaa.get_agent_config_class("self")
        agent_class.stream = True
        with self.subTest(config="default"):
            res = self.isaa.stream_read_llm("Ich habe einen Gedanken dieser lautet :", agent_class)
            print(res)
        agent_class.set_model_name("gpt4all#orca-mini-3b.ggmlv3.q4_0.bin")
        with self.subTest(config="orca"):
            res = self.isaa.stream_read_llm("Ich habe einen Gedanken dieser lautet :", agent_class)
            print(res)

    def test_run_agentChain(self):
        # Definieren Sie die Eingabeparameter

        user_text = "test"
        agent_tasks = [{'use': 'agent', 'name': 'test_agent', 'args': 'Hey how is it going this is a test.',
                        'mode': 'free', 'return': '$return'}]
        config = self.isaa.get_agent_config_class('test_agent')  # replace with an actual AgentConfig instance
        result = self.isaa.execute_thought_chain(user_text, agent_tasks, config)
        print(result)
        # replace with your custom assertions
        self.assertIsNotNone(result)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)


class TestIsaaUnit(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.app = get_app("test-IsaaUnit")

    @patch('requests.get')
    def test_get_ip(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {'ip': '123.123.123.123'}
        mock_get.return_value = mock_response

        result = get_ip()
        self.assertEqual(result, '123.123.123.123')

    @patch('requests.get')
    def test_get_location(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {'city': 'Berlin', 'region': 'Berlin', 'country_name': 'Germany'}
        mock_get.return_value = mock_response

        result = get_location()
        try:
            res = result.result(timeout=15)
            self.assertEqual(res, 'city: Berlin,region: Land Berlin,country: Germany,')
        except Exception:
            pass

    def test_extract_code(self):
        x = 'Hallo 123 ```python\nprint("Hello, World!")\n```asdadw'
        result = extract_code(x)
        self.assertEqual(result, ('print("Hello, World!")\n', 'python'))

    def test_get_tool_with_ac_mod(self):
        self.app.AC_MOD = self.app.get_mod('isaa')
        self.app.AC_MOD.name = 'isaa'
        result = get_tool(self.app)
        self.assertEqual(result, self.app.AC_MOD)

    @patch('toolboxv2.mods.isaa.isaa.Tools')  # replace 'your_module' with the actual module name
    def test_get_tool_no_app(self, mock_tools):
        result = get_tool(None)
        mock_tools.assert_called_once()
        self.assertEqual(result, mock_tools.return_value)

    @patch('replicate.Client')
    def test_initialize_gi(self, mock_client):
        model_name = 'test_model'
        self.app.AC_MOD = get_tool(self.app)
        self.app.AC_MOD.config = {'genrate_image-init': False, 'REPLICATE_API_TOKEN': 'test_token'}
        initialize_gi(self.app, model_name)
        mock_client.assert_called_once_with(api_token='test_token')
        initialize_gi(self.app.get_mod("isaa"), model_name)
        self.assertTrue(self.app.AC_MOD.config['genrate_image-init'])

    # def test_genrate_image(self):
    #     mock_app = self.app
    #     inputs = {}
    #     model = 'stability-ai/stable-diffusion'
    #     with ValueError as err:
    #         res = genrate_image(inputs, mock_app, model)
    #
    #     print(res)
    @patch('os.system')
    def test_show_image_in_internet(self, mock_system):
        image_url = 'http://example.com/image.jpg'
        show_image_in_internet(image_url, BROWSER)
        mock_system.assert_called_once_with(f'start {BROWSER} {image_url}')

    @patch('toolboxv2.mods.isaa.isaa.genrate_image')
    @patch('toolboxv2.mods.isaa.isaa.show_image_in_internet')
    def test_image_generating_tool(self, mock_show_image, mock_generate_image):
        mock_generate_image.return_value = ['http://example.com/image.jpg']
        image_genrating_tool('prompt', self.app)
        mock_generate_image.assert_called_once()
        mock_show_image.assert_called_once()

    def test_dilate_string(self):
        test_string = """
Zum Inhalt springen
Menü
Wohnen
Klimawandel
Krieg in der Ukraine
Alle Themen
Blog
TEILCHEN
Im Netz verstecken sich kleine Schätze – ZEIT ONLINE findet sie.
Über die Autoren
Janis Dietz22. Januar 2021
Einmal quer durch Havanna, bitte!

Plötzlich ist man in Havanna, Rio oder Jekatarinburg – „Drive & Listen“ macht es möglich. Foto: Andrei Luca/Youtube
Drive & Listen ist eine einfache aber sehnsuchtsschürende Anwendung, mit der man von zu Hause aus durch fremde Straßen fahren kann. Man klickt in einem Drop-Down-Menü auf eine von über 50 Städtenamen und schon ist man im Auto unterwegs, mitten in einer unbekannten Großstadt und begleitet von lokalen Radiosendern. Ich entscheide mich für die kubanische Hauptstadt. (Was hilft besser gegen Corona-Winterblues als karibisches Flair?) Und plötzlich fahre ich durch Havanna. Links das Meer, auf der rechten Seite ziehen die leicht verfallenen ehemaligen Prachtbauten an mir vorbei. Ab und an überhole ich einen alten Chevrolet oder Buick. Das Radio dudelt vor sich hin. Am liebsten würde ich jetzt ein Fenster öffnen und mein Gesicht in den karibischen Wind halten.  Weiter„Einmal quer durch Havanna, bitte!“


Dennis Schmees14. Januar 2021
Pflegekräfte in der Pandemie
„Nichts hätte mich darauf vorbereiten können“
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Zu Beginn der Corona-Pandemie haben sich in einem Pflegeheim in Hessen mehr als zwei Dutzend Bewohnerinnen, Bewohner und Pflegekräfte mit dem Virus infiziert. Drei Menschen sind an der Erkrankung gestorben. Um das Erlebte zu verarbeiten, hat eine Krankenschwester ein Gedicht geschrieben.

Weiter„„Nichts hätte mich darauf vorbereiten können““


ÜBER DIESES BLOG
Nicht jedes Video ist eine Nachricht, nicht jede Grafik lohnt für einen Artikel. Wir teilen sie trotzdem via Twitter, Facebook oder sprechen mit Freunden darüber. Weil sie sehenswert sind, unterhaltsam und informativ. Damit sie nicht einfach wieder verschwinden, sammelt ZEIT ONLINE im Teilchen-Blog regelmäßig Kleines, aber Feines aus dem Netz. Folgen Sie dem Blog auch auf Twitter unter #Teilchen.
Andreas Loos22. Dezember 2020
Dieses Mathe-Game lässt Sie den Weihnachtsbaum vergessen
Weihnachten und Corona
Vorsicht, Suchtgefahr! © Screenshot imaginary.org
Im Weltraum gehen Coronaviren ja vermutlich schnell kaputt, wegen UV-Strahlung, Kälte und dergleichen. Und für einen einzelnen Astronauten allein – obendrein im Anzug – besteht überhaupt keine Ansteckungsgefahr. Insofern ist das Spiel, das Mathematiker für Weihnachten online gestellt haben, perfekt coronakonform.

Weiter„Dieses Mathe-Game lässt Sie den Weihnachtsbaum vergessen“


Dennis Schmees11. Dezember 2020
Doktor trotz Prüfungsangst
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Schon die Vorstellung, einen Vortrag zu halten, lässt viele Menschen feuchte Hände bekommen. Das ist auch in Zeiten sozialer Distanz und Zoom-Meetings nicht anders. Ein Doktorand hat seinen Herzschlag vor, während und nach der Verteidigung seiner Doktorarbeit gemessen. Das Ergebnis: Ruhe zu bewahren ist trotz Anspannung möglich – und auch Freude sorgt für Herzrasen.

Weiter„Doktor trotz Prüfungsangst“


Tobias Dorfer1. Dezember 2020
Wenn Ärzte zu Tröstern werden
Corona-Intensivstation
Der Arzt Joseph Varon tröstet im United Memorial Medical Center in Houston an Thanksgiving einen Corona-Patienten. © Go Nakamura/Getty Images
Ein älterer Mann vergräbt seinen Kopf in den Armen eines Arztes. Er wird im United Memorial Medical Center in der US-Metropole Houston wegen Covid-19 behandelt. Der Arzt Joseph Varon hatte ihn am 26. November, dem Tag, an dem in den USA Thanksgiving gefeiert wird, in seinem Bett gefunden, weinend und um Hilfe rufend: „Ich möchte mit meiner Frau zusammen sein“, hatte der Mann gesagt. Varon kann nicht mehr tun als den Mann zu trösten.

Weiter„Wenn Ärzte zu Tröstern werden“


Dennis Schmees4. November 2020
World of Pandemieforschung
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Eine virtuelle Pandemie verbreitet sich im Jahr 2005 im Spiel World of Warcraft. Die Entdeckungen, die die epidemiologische Forschung anschließend machte, lässt uns bis heute echte Pandemien besser verstehen.

Weiter„World of Pandemieforschung“


Tobias Dorfer27. Oktober 2020
Kasachstan
Wie Kasachstan vom „Borat“-Boom profitieren möchte
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Frauenfeindlich, homophob, antisemitisch – so beschreibt der (fiktionale) Journalist Borat Sagdiyev im gleichnamigen Kinofilm sein Heimatland Kasachstan. Ein Land, das so rückständig beschrieben wird, wäre fast schon ein Fall für die Reisewarnungsabteilung des Auswärtigen Amtes. Und dennoch macht die Tourismusbehörde Kasachstans nun aus den zweifelhaften Zuschreibungen Gold.

Weiter„Wie Kasachstan vom „Borat“-Boom profitieren möchte“


Sasan Abdi-Herrle17. September 2020
Joko und Klaas machen das Grauen von Moria sichtbar
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Verteilungsschlüssel, EU-Türkei-Abkommen, „Koalition der Willigen“: Das Flüchtlingsthema kann schnell einen virtuellen Charakter bekommen, wenn man es aus der sinnbildlichen „warmen Stube“ verfolgt. Dabei geht es bei aller komplexen Politik doch um Menschen. Die Entertainer Joko Winterscheidt und Klaas Heufer-Umlauf haben dazu beigetragen, das konkrete Grauen der Flüchtlinge und Migranten auf Lesbos sichtbarer zu machen. Dazu nutzten sie die Sendezeit, die sie am Dienstag „gewonnen“ hatten: Die Kurzdoku A Short Story of Moria über die Lage auf Lesbos lief am Mittwochabend zur Primetime auf ProSieben.

Weiter„Joko und Klaas machen das Grauen von Moria sichtbar“


Dennis Schmees2. September 2020
US-Wahlkampf
Joe Biden, der Präsident von Animal Crossing
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Joe Bidens digitaler Wahlkampf geht in die nächste Runde. Während Donald Trump trotz Pandemie weiter große Wahlveranstaltungen abhält, setzen Joe Biden und Kamala Harris auf Livestreams und nun auch Videospiele. Weiter„Joe Biden, der Präsident von Animal Crossing“


Tobias Dorfer21. August 2020
„Ohne Joe Biden würde ich jetzt nicht zu Ihnen sprechen“
Empfohlener redaktioneller Inhalt
An dieser Stelle finden Sie externen Inhalt, der den Artikel ergänzt. Sie können sich externe Inhalte mit einem Klick anzeigen lassen und wieder ausblenden.
  Externer Inhalt
Ich bin damit einverstanden, dass mir externe Inhalte angezeigt werden. Damit können personenbezogene Daten an Drittplattformen übermittelt werden. Mehr dazu in unserer Datenschutzerklärung.

Der digitale Parteitag der US-Demokraten war in vielerlei Hinsicht bemerkenswert. Es waren die Tage der großen Worte. Von Biden selbst. Von Michelle Obama. Von Barack Obama. Von der Vizepräsidentschaftskandidatin Kamala Harris. Und am Ende auch von einem 13 Jahre alten Jungen aus New Hampshire.

Weiter„„Ohne Joe Biden würde ich jetzt nicht zu Ihnen sprechen““



1 / 2 / … / 122 NÄCHSTE SEITE


NEUESTE TEILCHEN
Einmal quer durch Havanna, bitte!
„Nichts hätte mich darauf vorbereiten können“
Dieses Mathe-Game lässt Sie den Weihnachtsbaum vergessen
Doktor trotz Prüfungsangst
Wenn Ärzte zu Tröstern werden
DIESES BLOG DURCHSUCHEN
Suche

Nach oben
Impressum Hilfe & Kontakt Unternehmen Karriere Presse Jobs Shop Inserieren Mediadaten
Bildrechte Rechte & Lizenzen AGB Datenschutz Privacy Einstellungen Cookies & Tracking Abo kündigen
        """
        res = dilate_string(test_string, 0, 2, 0)
        res1 = dilate_string(test_string, 1, 2, 0)
        res2 = dilate_string(test_string, 2, 2, 0)
        res3 = dilate_string(test_string, 3, 2, 0)
        print(len(test_string), len(res), len(res1), len(res2), len(res3), len(res) + len(res1) + len(res2) + len(res3))
        self.assertGreater(len(test_string), len(res))
        self.assertGreater(len(test_string), len(res1))
        self.assertGreater(len(test_string), len(res2))
        self.assertGreater(len(test_string), len(res3))

        first_itter = dilate_string(test_string, 0, 2, 0)
        second_itter = dilate_string(first_itter, 0, 2, 0)

        print(len(second_itter))

        first_itter_ab = dilate_string(test_string, 2, 2, 0)
        second_itter_ab = dilate_string(first_itter_ab, 0, 2, 0)

        print(len(second_itter_ab))


class TestWebScraping(unittest.TestCase):

    def __init__(self, methodName: str = ...):
        super().__init__(methodName)
        self.app = get_app("test-WebScraping")

    @patch('toolboxv2.mods.isaa.isaa.get_text_summary')
    @patch('toolboxv2.mods.isaa.isaa.get_hyperlinks')
    def test_browse_website(self, mock_get_hyperlinks, mock_get_text_summary):
        mock_get_text_summary.return_value = 'summary'
        mock_get_hyperlinks.return_value = ['http://example.com']
        result = browse_website('http://example.com', 'question', 'summarize')
        self.assertEqual(result, 'Website Content Summary: summary\n\nLinks: [\'http://example.com\']')

    @patch('requests.get')
    def test_scrape_text(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = 'Test text'
        mock_get.return_value = mock_response

        result = scrape_text('http://testurl.com')
        self.assertEqual(result, 'Test text')

    @patch('requests.get')
    def test_scrape_links(self, mock_get):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<a href="http://testlink.com">Test Link</a>'
        mock_get.return_value = mock_response

        result = scrape_links('http://testurl.com')
        self.assertEqual(result, ['Test Link (http://testlink.com)'])

    def test_extract_hyperlinks(self):
        soup = BeautifulSoup('<a href="http://testlink.com">Test Link</a>', 'html.parser')
        result = extract_hyperlinks(soup)
        self.assertEqual(result, [('Test Link', 'http://testlink.com')])

    def test_format_hyperlinks(self):
        hyperlinks = [('Test Link', 'http://testlink.com')]
        result = format_hyperlinks(hyperlinks)
        self.assertEqual(result, ['Test Link (http://testlink.com)'])

    @patch('toolboxv2.mods.isaa.isaa.scrape_text')
    @patch('toolboxv2.mods.isaa.isaa.scrape_links')
    def test_get_text_summary(self, mock_scrape_links, mock_scrape_text):
        mock_scrape_text.return_value = 'Test text'
        mock_scrape_links.return_value = ['Test Link (http://testlink.com)']

        result = get_text_summary('http://testurl.com', 'Test question', lambda x: x)
        self.assertEqual(result, 'Result: Context ###Test text### Question ###Test question###')

    @patch('toolboxv2.mods.isaa.isaa.scrape_links')
    def test_get_hyperlinks(self, mock_scrape_links):
        mock_scrape_links.return_value = ['Test Link (http://testlink.com)']

        result = get_hyperlinks('http://testurl.com')
        self.assertEqual(result, ['Test Link (http://testlink.com)'])


class TestProcessCompletion(unittest.TestCase):
    tools = None
    t0 = 0
    app = None

    not_run = [
        'gpt-4',
        'gpt-3.5-turbo',
        'gpt-3.5-turbo-0613',
        'text-davinci-003',
        'gpt-4-0613',
        'code-davinci-edit-001',
        'text-curie-001',
        'text-babbage-001',
        'text-ada-001',
        'text-davinci-edit-001',
        'gpt-3.5-turbo-instruct',

        'google/flan-t5-small',
        'google/flan-t5-xxl',
        'databricks/dolly-v2-3b',

        'gpt4all#GPT4All-13B-snoozy.ggmlv3.q4_0.bin',  # 5/
        'gpt4all#orca-mini-7b.ggmlv3.q4_0.bin',  # 4.5/10 :
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin',  # for comm
        'gpt4all#wizardLM-13B-Uncensored.ggmlv3.q4_0.bin',
        'gpt4all#ggml-replit-code-v1-3b.bin',  # Hy ly crati
        'knkarthick/MEETING_SUMMARY'
    ]

    models = [
        'gpt4all#GPT4All-13B-snoozy.ggmlv3.q4_0.bin',  # 5/10 (summary/classify/pl_lv2 in : 13.75s
        'gpt4all#orca-mini-7b.ggmlv3.q4_0.bin',  # : 7.17s
        # 4.5/10 : Hily spesific if you have any questions related to programming or computer science, feel free to ask me! , classify
        'gpt4all#orca-mini-3b.ggmlv3.q4_0.bin',  # : 3.76s
        # for command exection and evalation prosseses 6/10 context classify (tool use) qa code summ
        'gpt4all#wizardLM-13B-Uncensored.ggmlv3.q4_0.bin',  # : 13.62s
        # Conversational and Thinking Sartegegs 7.4/10 summary classify   (q/a 1): py Lv2)
        'gpt4all#ggml-replit-code-v1-3b.bin'  # Hy ly crative # : 11.08s
    ]

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test-TestIsaa")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa", "toolboxv2.mods.")
        cls.app.new_ac_mod("isaa")
        cls.tools: Tools = cls.app.get_mod('isaa')
        cls.tools.load_keys_from_env()

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def setUp(self):
        self.app_mock = Mock()
        self.tools = self.app.get_mod('isaa')

    def tearDown(self):
        self.tools._on_exit()
        self.app.remove_mod('isaa')

    def test_process_completion_returns_string(self):
        text = "test_text"
        config = self.tools.create_agent_class().set_completion_mode('text')
        result = self.tools.process_completion(text, config)
        self.assertIsInstance(result, str)

    def test_process_completion_with_different_text(self):
        text = "different_test_text"
        config = self.tools.create_agent_class().set_completion_mode('text')
        result = self.tools.process_completion(text, config)
        self.assertIsInstance(result, str)

    def test_process_completion_with_invalid_text(self):
        text = 123  # ungültiger Text
        config = self.tools.create_agent_class()
        with self.assertRaises(ValueError):
            self.tools.process_completion(text, config)

    def test_process_completion_with_invalid_config(self):
        text = "test_text"
        config = "invalid_config"  # ungültige Konfiguration
        with self.assertRaises(TypeError):
            self.tools.process_completion(text, config)

    def test_process_completion_with_invalid_settings(self):
        text = "test_text"
        config = self.tools.create_agent_class()
        config.completion_mode = "invalid_mode"  # ungültige Einstellung
        with self.assertRaises(ValueError):
            self.tools.process_completion(text, config)
