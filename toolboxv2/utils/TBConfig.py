from dotenv import load_dotenv

from toolboxv2.utils import get_logger
from toolboxv2.utils.toolbox import Singleton

# Load environment variables from .env file
load_dotenv()

class OverRideConfig:
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None  # Oder einen anderen Standardwert zurückgeben

    def __setitem__(self, key, value):
        setattr(self, key, value)

class ConfigIsaa:
    name: str = "isaa"
    genrate_image_in: str = "stability-ai/stable-diffusion"
    genrate_image_init: bool = False
    agents_name_list: list = []
    text_splitter3_init: bool
    text_splitter2_init: bool
    text_splitter1_init: bool
    text_splitter0_init: bool
    WOLFRAM_ALPHA_APPID: str
    HUGGINGFACEHUB_API_TOKEN: str
    OPENAI_API_KEY: str
    REPLICATE_API_TOKEN: str
    IFTTTKey: str
    SERP_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_API_ENV: str
    price: dict



def get_config(config_class):
    return {attr_name: getattr(config_class, attr_name) for attr_name in dir(config_class)}


class Config(metaclass=Singleton):
    def __init__(self):
        self.configs = {}
        self.configs_class = [ConfigIsaa]
        self.configs_ = {ConfigIsaa.name: ConfigIsaa}
        self.scopes = []
        self.scope = ""

    def initialize(self):
        get_logger().info("initialize configs")
        for config_c in self.configs_class:
            if "name" in dir(config_c):
                self.scopes.append(config_c.name)
                self.scope = config_c.name
                self.configs[config_c.name] = get_config(config_c)
                get_logger().info(f"Added {config_c.name}")
            else:
                get_logger().error(f"Error no name attr in : {config_c}")

    def gets(self, index):
        return self.configs[self.scope][index]

    def get(self):
        return self.configs_[self.scope]

    def get_scope(self):
        return self.scope

    def get_scopes(self):
        return self.scopes

    def set(self, index, value):
        self.configs[self.scope][index] = value

    def set_scope(self, scope):
        if scope in self.scopes:
            self.scope = scope


x = [['Thought: The task chain provided is a well-structured plan for creating a dynamic web scraping tool. It covers all the necessary steps from defining the search criteria to testing the implemented chain. However, it lacks some details such as error handling and continuous monitoring for changes in the target websites. \n\nKnowledge: In the real world, web scraping tools need to be robust and adaptable to changes in the structure of the target websites. They also need to have error handling mechanisms to deal with unexpected scenarios. \n\nAction: I will rate the task chain based on its comprehensiveness, clarity, and practicality.\n\n', 'Agent: self, mode: tools'], ["I would rate this task chain 85 out of 100. Here's my reasoning:\n\n1. Clarity and Specificity (25/25): The chain is well-defined and specific. Each step is clear and contributes to the overall goal of creating a dynamic web search chain.\n\n2. Completeness (20/25): The chain covers most of the necessary steps, from defining search criteria to testing the implemented chain. However, it could include a step for handling potential errors or exceptions during the web scraping process.\n\n3. Practicality (20/25): The chain is practical and feasible. The tools mentioned are commonly used in web scraping tasks. However, the chain assumes that the user has knowledge of these tools, which may not always be the case.\n\n4. Flexibility (20/25): The chain is designed to be flexible and adaptable to different websites and content types. However, it doesn't explicitly mention how to handle changes in website structures or content formats that could affect the scraping process.\n\nOverall, this is a solid task chain for creating a dynamic web search tool. It could be improved by adding steps for error handling and dealing with changes in website structures or content formats.", 'Agent: self, mode: execution'], ['Ich würde diese Aufgabenkette mit einer Punktzahl von 90 bewerten. Die Kette enthält alle notwendigen Schritte, um eine dynamische Websuche nach bestimmten Informationen durchzuführen. Die einzelnen Schritte sind gut strukturiert und decken alle relevanten Aspekte ab.\n\nDie Begründung für die Punktzahl von 90 ist, dass die Kette alle wichtigen Schritte enthält, um das Ziel zu erreichen. Es werden klare Anweisungen gegeben, wie die spezifischen Informationen definiert werden sollen, welche Websites oder Online-Quellen relevant sind und welche Web-Scraping-Tools verwendet werden können. Es wird auch betont, dass ein flexibler Suchalgorithmus entwickelt werden muss, der sich an verschiedene Arten von Websites und Inhalten anpassen kann.\n\nDie Implementierung der Kette und die Verbindung der verschiedenen Komponenten werden ebenfalls erwähnt, was darauf hinweist, dass die Kette automatisiert werden soll, um Informationen von mehreren Websites zu suchen und zu extrahieren. Es wird auch darauf hingewiesen, dass die Kette getestet werden muss, um sicherzustellen, dass sie korrekt funktioniert und die gewünschten Informationen effektiv abruft. Die Möglichkeit, die Kette zu verfeinern, um ihre Leistung und Anpassungsfähigkeit an Änderungen in der Webstruktur zu verbessern, wird ebenfalls erwähnt.\n\nInsgesamt ist die Aufgabenkette gut durchdacht und enthält alle notwendigen Schritte, um das Ziel zu erreichen. Die Punktzahl von 90 spiegelt die hohe Qualität und Vollständigkeit der Kette wider.', 'Agent: execution, mode: free'], ['Ich würde diese Aufgabenkette mit einer Punktzahl von 95 bewerten. Die Kette enthält alle notwendigen Schritte, um eine dynamische Websuche nach bestimmten Informationen durchzuführen. Die einzelnen Schritte sind gut strukturiert und decken alle relevanten Aspekte ab.\n\nDie Begründung für die Punktzahl von 95 ist, dass die Kette alle wichtigen Schritte enthält, um das Ziel zu erreichen. Es wird klargestellt, dass spezifische Informationen definiert werden müssen, die die Kette suchen soll, und dass Websites oder Online-Quellen identifiziert werden müssen, auf denen diese Informationen wahrscheinlich zu finden sind. Es wird auch betont, dass die Auswahl der geeigneten Web-Scraping-Tools wichtig ist, um Daten von diesen Quellen zu extrahieren.\n\nDie Entwicklung eines flexiblen Suchalgorithmus, der sich an verschiedene Arten von Websites und Inhalten anpassen kann, wird ebenfalls erwähnt. Dieser Algorithmus sollte in der Lage sein, durch Webseiten zu navigieren, relevante Informationen zu lokalisieren und sie genau zu extrahieren.\n\nDie Implementierung der Kette und die Verbindung der verschiedenen Komponenten werden ebenfalls erwähnt, was darauf hinweist, dass die Kette automatisiert werden soll, um Informationen von mehreren Websites zu suchen und zu extrahieren. Es wird auch darauf hingewiesen, dass die Kette gründlich getestet werden muss, um sicherzustellen, dass sie korrekt funktioniert. Dies kann durch das Ausführen der Kette auf einer Beispielsammlung von Websites und die Überprüfung der Genauigkeit der extrahierten Informationen erfolgen. Die Möglichkeit, die Kette zu verfeinern, um ihre Leistung und Anpassungsfähigkeit an Änderungen in der Webstruktur zu verbessern, wird ebenfalls erwähnt.\n\nInsgesamt ist die Aufgabenkette gut durchdacht und enthält alle notwendigen Schritte, um das Ziel zu erreichen. Die Punktzahl von 95 spiegelt die hohe Qualität und Vollständigkeit der Kette wider.', 'Agent: execution, mode: free'], ["Ich würde diese Aufgabenkette mit einer Punktzahl von 60 bewerten.\n\nBegründung:\nDie Aufgabenkette hat das Ziel, eine Chain zu erstellen, die das Web nach einer bestimmten Information durchsucht. Dies ist ein interessantes Ziel, da die dynamische Suche im Web nach Informationen eine wichtige Fähigkeit ist.\n\nDie gegebene Chain '([],)' ist jedoch sehr einfach und enthält keine spezifischen Anweisungen oder Algorithmen, um das Web zu durchsuchen. Es fehlen auch Informationen darüber, nach welcher bestimmten Information gesucht werden soll.\n\nUm die Punktzahl zu erhöhen, könnte die Aufgabenkette spezifischere Anweisungen enthalten, wie z.B. die Verwendung von Web-Crawling-Techniken, die Verwendung von APIs oder die Implementierung von Suchalgorithmen. Außerdem wäre es hilfreich, die gewünschte Information genauer zu definieren.\n\nMit diesen Verbesserungen könnte die Aufgabenkette eine höhere Punktzahl erhalten, da sie ein konkretes Ziel hat und spezifische Schritte enthält, um dieses Ziel zu erreichen.", 'Agent: thinkm, mode: free'], ["Ich würde diese Aufgabenkette mit einer Punktzahl von 30 bewerten.\n\nBegründung:\nDas Ziel, eine Chain zu erstellen, die das Web nach einer bestimmten Information durchsucht, ist grundsätzlich gut. Es ist wichtig, in der heutigen digitalen Welt effektive Suchtechniken zu beherrschen.\n\nAllerdings ist die gegebene Chain '([],)' sehr einfach und enthält keinerlei Anweisungen oder Algorithmen, um das Web zu durchsuchen. Es fehlen auch Informationen darüber, nach welcher bestimmten Information gesucht werden soll.\n\nUm die Punktzahl zu erhöhen, müsste die Aufgabenkette konkretere Anweisungen enthalten. Zum Beispiel könnte sie spezifische Suchbegriffe oder Websites angeben, die durchsucht werden sollen. Außerdem wäre es hilfreich, Techniken wie Web-Crawling, API-Integration oder Suchalgorithmen zu erwähnen.\n\nMit diesen Verbesserungen könnte die Aufgabenkette eine höhere Punktzahl erhalten, da sie ein klar definiertes Ziel hat und spezifische Schritte enthält, um dieses Ziel zu erreichen.", 'Agent: thinkm, mode: free']]
y = [['Ich würde diese Aufgabenkette mit einer Punktzahl von 90 bewerten. Hier ist meine Begründung:\n\n1. Die Kette beginnt damit, dass der Agent die spezifischen Informationen definiert, nach denen gesucht werden soll. Dies ist ein wichtiger erster Schritt, um das Ziel zu erreichen. (Punktzahl: 90)\n\n2. Der Einsatz des Tools, um einen Web-Crawler zu entwickeln, der die definierten Informationen extrahieren kann, ist ein effektiver Ansatz. Die Verwendung von Python und Bibliotheken wie BeautifulSoup oder Scrapy ist eine gute Wahl. (Punktzahl: 90)\n\n3. Die Agenten sollen eine Kette von Aktionen erstellen, der der Web-Crawler folgen wird. Dies beinhaltet den Besuch eines Ausgangspunktes, wie einer Suchmaschine oder einer bestimmten Website, und das Folgen von Links oder das Durchführen von Suchen, um die definierten Informationen zu erreichen. Dieser Schritt ist entscheidend, um das Ziel zu erreichen. (Punktzahl: 90)\n\n4. Das Tool soll den Web-Crawler dabei unterstützen, die relevanten Informationen kontinuierlich zu extrahieren und in einem strukturierten Format, wie einer Datenbank oder einer CSV-Datei, zu speichern. Dies ist ein wichtiger Schritt, um die gesuchten Informationen zu organisieren und zu speichern. (Punktzahl: 90)\n\n5. Das Tool soll auch Mechanismen zur Fehlerbehandlung implementieren, um Probleme während des Crawling-Prozesses, wie defekte Links oder nicht erreichbare Webseiten, zu behandeln. Dies ist ein wichtiger Aspekt, um sicherzustellen, dass der Web-Crawler zuverlässig und robust ist. (Punktzahl: 90)\n\n6. Der Agent soll die Skalierbarkeit und Effizienz der Kette berücksichtigen. Die Kette sollte so gestaltet sein, dass sie eine große Menge an Daten verarbeiten kann und sich an Änderungen in der Struktur der Webseiten anpassen kann. Die Implementierung von Techniken wie paralleler Verarbeitung oder die Nutzung cloudbasierter Ressourcen ist ein guter Ansatz, um diese Anforderungen zu erfüllen. (Punktzahl: 90)\n\nInsgesamt erhält die Aufgabenkette eine Punktzahl von 90, da sie alle wichtigen Schritte enthält, um das Ziel zu erreichen, und effektive Werkzeuge und Ansätze verwendet. Es gibt jedoch immer Raum für Verbesserungen und Anpassungen an spezifische Anforderungen.', 'Agent: execution, mode: free', ([{'use': 'agent', 'name': 'self', 'args': 'Define the specific information you are looking for. This could be a particular keyword, phrase, or any other relevant data.', 'mode': 'free', 'return': '$defined_information'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'Develop a web crawler using Python and libraries like BeautifulSoup or Scrapy that can navigate through web pages and extract the $defined_information.', 'return': '$web_crawler'}, {'use': 'agent', 'name': 'self', 'args': 'Establish a chain of actions that the $web_crawler will follow. This could involve visiting a starting point, such as a search engine or a specific website, and then following links or performing searches to reach the $defined_information.', 'mode': 'free', 'return': '$chain_of_actions'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'As the $web_crawler progresses through the $chain_of_actions, it should continuously extract and store the relevant information in a structured format, such as a database or a CSV file.', 'return': '$stored_information'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'Implement error handling mechanisms to handle any issues that may arise during the crawling process, such as broken links or inaccessible web pages.', 'return': '$error_handling'}, {'use': 'agent', 'name': 'self', 'args': 'Consider the scalability and efficiency of the chain. The chain should be designed to handle a large volume of data and adapt to changes in web page structures. This may involve implementing techniques like parallel processing or utilizing cloud-based resources.', 'mode': 'free', 'return': '$scalability_and_efficiency'}],)], ['Ich würde diese Aufgabenkette mit einer Punktzahl von 90 bewerten. Hier ist meine Begründung:\n\n1. Die Kette beginnt damit, dass der Agent die spezifischen Informationen definiert, nach denen gesucht werden soll. Dies ist ein wichtiger erster Schritt, um das Ziel zu erreichen. (Punktzahl: 8/10)\n\n2. Die Verwendung eines Tools zum Entwickeln eines Web-Crawlers ist eine effektive Methode, um das Web nach den definierten Informationen zu durchsuchen. Die Verwendung von Python und Bibliotheken wie BeautifulSoup oder Scrapy ist eine gute Wahl. (Punktzahl: 9/10)\n\n3. Die Agenten sollen eine Kette von Aktionen erstellen, der der Web-Crawler folgen wird. Dies beinhaltet den Besuch eines Ausgangspunktes, wie einer Suchmaschine oder einer bestimmten Website, und das Folgen von Links oder das Durchführen von Suchen, um die definierten Informationen zu erreichen. Dieser Schritt ist entscheidend, um die gewünschten Informationen zu finden. (Punktzahl: 9/10)\n\n4. Das Tool soll den Web-Crawler dabei unterstützen, die relevanten Informationen kontinuierlich zu extrahieren und in einem strukturierten Format, wie einer Datenbank oder einer CSV-Datei, zu speichern. Dies ist wichtig, um die gesammelten Informationen effizient zu organisieren und zu nutzen. (Punktzahl: 9/10)\n\n5. Die Implementierung von Mechanismen zur Fehlerbehandlung ist ein wichtiger Aspekt, um Probleme während des Crawling-Prozesses zu bewältigen. Das Erkennen und Behandeln von Fehlern wie defekten Links oder nicht erreichbaren Webseiten ist entscheidend, um die Zuverlässigkeit des Crawlers zu gewährleisten. (Punktzahl: 9/10)\n\n6. Der Agent soll die Skalierbarkeit und Effizienz der Kette berücksichtigen. Die Kette sollte so gestaltet sein, dass sie eine große Menge an Daten verarbeiten kann und sich an Änderungen in der Struktur von Webseiten anpassen kann. Die Implementierung von Techniken wie paralleler Verarbeitung oder die Nutzung cloudbasierter Ressourcen kann dabei helfen. (Punktzahl: 8/10)\n\nInsgesamt erhält die Aufgabenkette eine Punktzahl von 90/100, da sie alle wichtigen Schritte enthält, um das Ziel zu erreichen, und effektive Werkzeuge und Strategien verwendet, um das Web nach den gewünschten Informationen zu durchsuchen.', 'Agent: thinkm, mode: free', ([{'use': 'agent', 'name': 'self', 'args': 'Define the specific information you are looking for. This could be a particular keyword, phrase, or any other relevant data.', 'mode': 'free', 'return': '$defined_information'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'Develop a web crawler using Python and libraries like BeautifulSoup or Scrapy that can navigate through web pages and extract the $defined_information.', 'return': '$web_crawler'}, {'use': 'agent', 'name': 'self', 'args': 'Establish a chain of actions that the $web_crawler will follow. This could involve visiting a starting point, such as a search engine or a specific website, and then following links or performing searches to reach the $defined_information.', 'mode': 'free', 'return': '$chain_of_actions'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'As the $web_crawler progresses through the $chain_of_actions, it should continuously extract and store the relevant information in a structured format, such as a database or a CSV file.', 'return': '$stored_information'}, {'use': 'tool', 'name': 'write-production-redy-code', 'args': 'Implement error handling mechanisms to handle any issues that may arise during the crawling process, such as broken links or inaccessible web pages.', 'return': '$error_handling'}, {'use': 'agent', 'name': 'self', 'args': 'Consider the scalability and efficiency of the chain. The chain should be designed to handle a large volume of data and adapt to changes in web page structures. This may involve implementing techniques like parallel processing or utilizing cloud-based resources.', 'mode': 'free', 'return': '$scalability_and_efficiency'}],)]]
