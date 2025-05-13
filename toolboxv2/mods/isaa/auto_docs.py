import ast
import os
from collections import Counter, defaultdict

import networkx as nx
import nltk
from nltk.corpus import stopwords

# Initialize NLTK components
from nltk.tokenize import word_tokenize


def create_import_usage_graph(folder_path):
    graph = nx.DiGraph()
    import_count = Counter()
    import_usage = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding='utf-8') as f:
                    node_name = os.path.relpath(file_path, folder_path)
                    tree = ast.parse(f.read(), filename=node_name)

                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import | ast.ImportFrom):
                            for alias in node.names:
                                imported_module = alias.name
                                graph.add_edge(imported_module, node_name)
                                import_count[imported_module] += 1

                                # Aufzeichnen, in welchen Dateien der Import auftritt
                                if imported_module not in import_usage:
                                    import_usage[imported_module] = []
                                import_usage[imported_module].append(node_name)

    # Sortiere Dateien nach Importhäufigkeit
    sorted_files = sorted(import_count.items(), key=lambda x: x[1], reverse=True)
    return graph, sorted_files, import_usage


class ClassUsageAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.class_relations = nx.DiGraph()
        self.class_usage_count = Counter()
        self.class_usage_locations = {}
        self.class_sources = {}

    def visit_ClassDef(self, node):
        class_name = node.name
        self.class_relations.add_node(class_name, type='class')

        # Speichere Datei und ggf. übergeordnete Klasse
        current_file = self.current_file
        parent_class = None

        if hasattr(self, 'current_class'):
            parent_class = self.current_class

        # Speichern der Quelle (Datei und übergeordnete Klasse)
        self.class_sources[class_name] = {
            'file': current_file,
            'parent_class': parent_class
        }

        # Überprüfe Methoden, die andere Klassen instanziieren
        self.current_class = class_name
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                for subnode in ast.walk(stmt):
                    if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                        called_class = subnode.func.id
                        self.class_relations.add_edge(class_name, called_class, relation='uses')
                        self.class_usage_count[called_class] += 1

                        # Aufzeichnen, wo diese Klasse verwendet wurde
                        if called_class not in self.class_usage_locations:
                            self.class_usage_locations[called_class] = []
                        self.class_usage_locations[called_class].append(self.current_file)

        self.generic_visit(node)

    def visit_Module(self, node):
        self.current_file = self.file_name
        self.generic_visit(node)


def analyze_class_usage(folder_path):
    analyzer = ClassUsageAnalyzer()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding='utf-8') as f:
                    tree = ast.parse(f.read(), filename=file_path)
                    analyzer.file_name = file_path
                    analyzer.visit(tree)

    # Sortiere Klassen nach Verwendungsanzahl
    sorted_classes = sorted(analyzer.class_usage_count.items(), key=lambda x: x[1], reverse=True)
    return analyzer.class_relations, sorted_classes, analyzer.class_usage_locations, analyzer.class_sources


class FunctionUsageAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_relations = nx.DiGraph()
        self.function_usage_count = Counter()
        self.function_usage_locations = {}
        self.function_sources = {}

    def visit_FunctionDef(self, node):
        function_name = node.name
        self.function_relations.add_node(function_name, type='function')

        # Speichern der Datei und des Quellcodes der Funktion
        current_file = self.current_file
        function_code = ast.get_source_segment(self.source_code, node)

        self.function_sources[function_name] = {
            'file': current_file,
            'source_code': function_code
        }

        # Suche Funktionsaufrufe innerhalb der Funktionsdefinition
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                called_function = subnode.func.id
                self.function_relations.add_edge(function_name, called_function, relation='calls')
                self.function_usage_count[called_function] += 1

                # Aufzeichnen, wo die Funktion aufgerufen wurde
                if called_function not in self.function_usage_locations:
                    self.function_usage_locations[called_function] = []
                self.function_usage_locations[called_function].append(self.current_file)

        self.generic_visit(node)

    def visit_Module(self, node):
        self.current_file = self.file_name
        self.generic_visit(node)


def analyze_function_usage(folder_path):
    analyzer = FunctionUsageAnalyzer()

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, encoding='utf-8') as f:
                    analyzer.source_code = f.read()
                    tree = ast.parse(analyzer.source_code, filename=file_path)
                    analyzer.file_name = file_path
                    analyzer.visit(tree)

    # Sortiere Funktionen nach Verwendungsanzahl
    sorted_functions = sorted(analyzer.function_usage_count.items(), key=lambda x: x[1], reverse=True)
    return analyzer.function_relations, sorted_functions, analyzer.function_usage_locations, analyzer.function_sources


def combine_usage(import_graph, class_graph, function_graph, sorted_files, sorted_classes, sorted_functions,
                  import_usage, class_usage_locations, class_sources, function_usage_locations, function_sources):
    nx.compose_all([import_graph, class_graph, function_graph])

    # Erstelle eine detaillierte Ausgabe für Dateien, Klassen und Funktionen mit Verwendungsstellen und Quellen
    relevance_report = {
        'files': {
            'sorted': sorted_files,
            'usage': import_usage,
        },
        'classes': {
            'sorted': sorted_classes,
            'usage': class_usage_locations,
            'sources': class_sources  # Fügt die Dateiquelle und übergeordnete Klasse hinzu
        },
        'functions': {
            'sorted': sorted_functions,
            'usage': function_usage_locations,
            'sources': function_sources  # Fügt Dateiquelle und Quellcode der Funktion hinzu
        }
    }

    return relevance_report


def generate_usage_documentation(folder_path):
    # Erzeuge Verwendungsgraphen und sortierte Listen mit Verwendungsstellen und Quellen
    import_graph, sorted_files, import_usage = create_import_usage_graph(folder_path)
    class_graph, sorted_classes, class_usage_locations, class_sources = analyze_class_usage(folder_path)
    function_graph, sorted_functions, function_usage_locations, function_sources = analyze_function_usage(folder_path)

    # Kombiniere die Ergebnisse
    usage_report = combine_usage(import_graph, class_graph, function_graph, sorted_files, sorted_classes,
                                 sorted_functions, import_usage, class_usage_locations, class_sources,
                                 function_usage_locations, function_sources)

    return usage_report


def create_linear_analysis_path(usage_report):
    analyzed_files = set()
    linear_path = []

    # 1. Beginne mit den wichtigsten Dateien (nach Importhäufigkeit sortiert)
    for file, _ in usage_report['files']['sorted']:
        if file not in analyzed_files:
            linear_path.append({
                'type': 'core',
                'name': file,
                'references': usage_report['files']['usage'].get(file, []),
                'context': 'initial'
            })
            analyzed_files.add(file)

            # 2. Analysiere Klassen in der Datei
            for class_name, class_info in usage_report['classes']['sources'].items():
                if class_info['file'] == file:
                    linear_path.append({
                        'type': 'class',
                        'name': class_name,
                        'file': class_info['file'],
                        'parent_class': class_info['parent_class'],
                        'usage': usage_report['classes']['usage'].get(class_name, []),
                        'importance': 'class',
                        'context': 'file_analysis'
                    })

                    # 3. Analysiere Funktionen in der Klasse
                    for function_name, function_info in usage_report['functions']['sources'].items():
                        if function_info['file'] == file:
                            linear_path.append({
                                'type': 'function',
                                'name': function_name,
                                'file': function_info['file'],
                                'source_code': function_info['source_code'],
                                'usage': usage_report['functions']['usage'].get(function_name, []),
                                'importance': 'function',
                                'context': 'class_analysis'
                            })

    # 4. Weiterführende Referenzierung und Backreferencing
    for class_name, class_info in usage_report['classes']['sources'].items():
        if class_name not in [step['name'] for step in linear_path]:
            linear_path.append({
                'type': 'class',
                'name': class_name,
                'file': class_info['file'],
                'parent_class': class_info['parent_class'],
                'usage': usage_report['classes']['usage'].get(class_name, []),
                'importance': 'class',
                'context': 'backreference'
            })

    # 5. Kernaspekte des Codes
    for function_name, function_info in usage_report['functions']['sources'].items():
        if function_name not in [step['name'] for step in linear_path]:
            linear_path.append({
                'type': 'function',
                'name': function_name,
                'file': function_info['file'],
                'source_code': function_info['source_code'],
                'usage': usage_report['functions']['usage'].get(function_name, []),
                'importance': 'function',
                'context': 'backreference'
            })

    return linear_path


def format_core_object(data):
    return f"""
### Documentation (python code) ###
Object Type: {data['type']}
Object Name: {data['name']}

Context of Analysis: {data.get('context', 'N/A')}

### Task ###
Analyze the file `{data['name']}`. Describe its main purpose, key imports, and relationships to other files or objects.
"""


def format_class_object(data):
    return f"""
### Documentation (python code) ###
Object Type: {data['type']}
Class Name: {data['name']}

File Location: {data.get('file', 'N/A')}
Parent Class: {data.get('parent_class', 'None')}
Context of Analysis: {data.get('context', 'N/A')}

### Task ###
Analyze the class `{data['name']}`. Describe its role, inheritance structure, and how it is used in the code.
Backreference to other classes or modules if applicable.
"""


def format_function_object(data):
    return f"""
### Documentation (python code) ###
Object Type: {data['type']}
Function Name: {data['name']}

File Location: {data.get('file', 'N/A')}
Context of Analysis: {data.get('context', 'N/A')}

### Task ###
Analyze the function `{data['name']}`. Provide an explanation of its purpose, its usage, and how it interacts with other parts of the code.
Backreference to related functions or classes.

### Code Snippet ###
{data.get('source_code', 'Source code not available')}

### Expected Output ###
Explain the function's purpose, its parameters, and how it is used.
"""


def compare_with_generated_doc(obj, generated_doc, get_ref=False):
    # Füge Notizen und Kontext in die ursprüngliche Dokumentation ein
    formatted_obj = ""

    if get_ref:
        obj['context'] = get_ref(obj['context']) if 'context' in obj else get_ref(obj['name'])

    if obj['type'] == 'core':
        formatted_obj = format_core_object(obj)
    elif obj['type'] == 'class':
        formatted_obj = format_class_object(obj)
    elif obj['type'] == 'function':
        formatted_obj = format_function_object(obj)

    doc = {}
    # Prüfe, ob die generierte Doku zu diesem Objekt passt (basierend auf Namen, Typ etc.)
    if obj['type'] == 'core':  # Handle file-level documentation
        doc = generated_doc['files']['usage'].get(obj['name'], {})

    elif obj['type'] == 'class':  # Handle class-level documentation
        doc = generated_doc['classes']['usage'].get(obj['name'], {})

    elif obj['type'] == 'function':  # Handle function-level documentation
        doc = generated_doc['functions']['usage'].get(obj['name'], {})

        # If no documentation is found, return a fallback message
    if not doc:
        return formatted_obj

        # Extract relevant notes and additional context
    doc = set(doc)
    relevant_notes = doc if len(doc) > 0 else 'No relevant notes found.'
    additional_context = ""
    if get_ref:
        additional_context = "### Additional Context from Backreferencing ###\n"
        for ref in doc:
            additional_context += get_ref(ref)

    return f"""
{formatted_obj}

### Relevant Notes from Previous Documentation ###
{relevant_notes}

{additional_context}
"""


def flatten_data(relevance_report):
    # Flatten the data for easy processing
    elements = []

    # Add files
    for file, _ in relevance_report['files']['sorted']:
        elements.append({'name': file, 'type': 'file', 'references': relevance_report['files']['usage'].get(file, [])})

    # Add classes
    for class_name, _ in relevance_report['classes']['sorted']:
        class_info = relevance_report['classes']['sources'].get(class_name, {})
        elements.append({'name': class_name, 'type': 'class',
                         'references': relevance_report['classes']['usage'].get(class_name, []),
                         'file': class_info.get('file', ''), 'parent_class': class_info.get('parent_class', '')})

    # Add functions
    for func_name, _ in relevance_report['functions']['sorted']:
        function_info = relevance_report['functions']['sources'].get(func_name, {})
        elements.append({'name': func_name, 'type': 'function',
                         'references': relevance_report['functions']['usage'].get(func_name, []),
                         'file': function_info.get('file', ''), 'source_code': function_info.get('source_code', '')})

    return elements


def retrieve_most_relevant_combined(elements, target_objects=None, max_results=2):
    # Sort elements by popularity if no target objects are provided
    if not target_objects:
        sorted_elements = sorted(elements, key=lambda x: x.get('popularity', 0), reverse=True)
        return sorted_elements[:max_results]

    # Calculate relevance based on the combination of files, classes, and functions
    def calculate_relevance(element, target_objects):
        relevance_score = 0
        for target in target_objects:
            # Matching element name
            if element['name'] == target['name']:
                relevance_score += 10
            # Matching references or relationships between elements
            if target['name'] in element.get('references', []):
                relevance_score += 5
        return relevance_score

    # Rank elements by relevance
    ranked_elements = sorted(elements, key=lambda x: calculate_relevance(x, target_objects), reverse=True)
    return ranked_elements[:max_results]


def build_combined_tree_with_removal(elements, target_objects=None, max_depth=3, current_depth=0):
    # Base case: if max depth is reached or no more elements are relevant
    if current_depth >= max_depth or not elements:
        return []

    # Get the most relevant elements (0, 1, or 2 items)
    relevant_elements = retrieve_most_relevant_combined(elements, target_objects)

    # Remove the relevant elements from the list
    for elem in relevant_elements:
        elements.remove(elem)

    # Recursively build the tree
    tree = []
    for elem in relevant_elements:
        tree.append({
            'element': elem,
            'children': build_combined_tree_with_removal(elements, [elem], max_depth, current_depth + 1)
        })

    return tree


def build_combined_tree_without_removal(elements, target_objects=None, max_depth=3, current_depth=0, names=None):
    # Base case: if max depth is reached or no more elements are relevant

    if names is None:
        names = []

    if current_depth >= max_depth or not elements:
        return []

    # Get the most relevant elements (0, 1, or 2 items)
    relevant_elements = retrieve_most_relevant_combined(elements, target_objects, max_results=len(elements))

    # Recursively build the tree
    tree = []
    for elem in relevant_elements:
        if elem.get('name', str(elem)) in names:
            continue
        names.append(elem.get('name', str(elem)))
        tree.append({
            'element': elem,
            'children': build_combined_tree_without_removal(elements, [elem], max_depth, current_depth + 1, names)
        })

    return tree


def build_code_tree(folder_path, recursive=False, max_depth=None):

    documentation_data = generate_usage_documentation(folder_path)

    data_row = flatten_data(documentation_data)

    if max_depth is None:
        max_depth = len(data_row)

    if not recursive:
        tree_simpel = build_combined_tree_with_removal(data_row, None, max_depth)
        return tree_simpel

    tree_recursive = build_combined_tree_without_removal(data_row, None, max_depth)
    tree_recursive_mac = []
    for elem in tree_recursive:
        tree_recursive_mac.extend(
            build_combined_tree_with_removal(data_row.copy(), [elem.get("element")], max_depth * 2))
    return tree_recursive_mac


# Mini-task completion mock function to simulate LLM interaction
def mini_task_completion_moc(prompt):
    """
    This is a mock function to simulate the behavior of an LLM completion process.
    In reality, this would interact with an actual LLM API (like OpenAI, etc.).
    """
    return prompt


# Function to extract keywords and entities using NLTK
def extract_keywords_and_entities(text):
    """
    Extracts important keywords, named entities, and concepts from a text.
    """
    words = word_tokenize(text.lower())  # Tokenizing text
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words


# Function to format prompts for LLM from the relevant information
def generate_llm_prompt(data_chunk, context):
    """
    Formats a prompt for an LLM based on the provided data and context.
    """
    extract_keywords_and_entities(data_chunk)
    context_str = " ".join(context)
    prompt = f"Analyze the following codebase context: {context_str}. Then analyze this chunk of data: {data_chunk}. Identify key concepts, relations in the knowledge base."
    return prompt


# Function to analyze missing information (to be used in top-down traversal)
def analyze_missing_information(data_chunk, current_context):
    """
    Analyzes what information is missing from the current chunk and suggests improvements or additions.
    """
    # For simplicity, let's assume missing info is any reference to a non-existing concept in the context
    keywords = extract_keywords_and_entities(data_chunk)
    missing_information = [keyword for keyword in keywords if keyword not in current_context]
    return missing_information


def top_down_traversal(tree, current_context=None, missing_info=None, depth=0, mini_task_completion=None):
    if mini_task_completion is None:
        mini_task_completion = mini_task_completion_moc
    if current_context is None:
        current_context = set()  # Store the context as a set for fast lookups
    if missing_info is None:
        missing_info = defaultdict(list)

    # Traverse each element in the tree
    for node in tree:
        element = node['element']
        children = node['children']

        # Extract key points and analyze using small context window
        context_window = current_context.copy()  # Work with a copy to avoid mutation during analysis
        data_chunk = str(element)  # Here you would usually process code, but we mock it as a string

        # Analyze the element's data and extract missing information
        missing_data = analyze_missing_information(data_chunk, context_window)

        # If missing information is found, add it to the missing_info collection
        if missing_data:
            missing_info[element.get('name' 'root')].extend(missing_data)

        # Simulate using LLM to fill missing info (mock interaction)
        if missing_data:
            prompt = generate_llm_prompt(data_chunk, list(current_context))
            response = mini_task_completion(prompt)
            node['response'] = response
            node['missing_data'] = missing_data

        # Recursively process children
        missing_info.update(top_down_traversal(children, current_context | set(missing_data), missing_info, depth + 1,
                                               mini_task_completion=mini_task_completion))

    return missing_info


def bottom_up_traversal(tree, current_context=None, depth=0, collected_docs=None, mini_task_completion=None):
    """
    Traverse the tree from bottom-up, filling in missing information and collecting documentation.
    """
    if current_context is None:
        current_context = set()
    if mini_task_completion is None:
        mini_task_completion = mini_task_completion_moc
    if collected_docs is None:
        collected_docs = []

    for node in tree:
        element = node['element']
        children = node['children']

        # Step 1: Recursively process the children (bottom-up)
        child_context = set()
        for child in children:
            child_docs = bottom_up_traversal([child], current_context | child_context, depth + 1,
                                             mini_task_completion=mini_task_completion)
            collected_docs.extend(child_docs)
            # Merge children's information into the current context
            for doc in child_docs:
                child_context.update(doc.get('missing_data', []))  # Get missing data from children

        # Step 2: Process current node after all children are processed
        context_window = current_context | child_context  # Combine parent and child contexts

        # Analyze the current element to detect missing information
        data_chunk = str(element) + node.get('response', '')
        missing_data = analyze_missing_information(data_chunk, context_window)

        # If missing data exists, format the prompt and use mini_task_completion for enrichment
        prompt = generate_llm_prompt(data_chunk, list(context_window))
        llm_response = mini_task_completion(prompt)
        # Append the result for the current element to the documentation
        collected_docs.append({
            'element': element,
            'response': llm_response,
            'missing_data': missing_data,
            'references': list(context_window),  # List of references to other elements
            'context_window': list(context_window)
        })
        if missing_data:
            # Update current context with missing data (enrich it)
            current_context.update(missing_data)

        # Optionally store source code or other node-related data
        if 'source_code' not in node:
            node['source_code'] = ""
        node['source_code'] += '\n' + element.get('source_code', '')

    return collected_docs


def process_tree_for_knowledge_base(tree, mini_task_completion=None):
    # First run: top-down traversal to gather missing information
    print("Top-down traversal: Gathering missing information...")
    missing_info = top_down_traversal(tree, mini_task_completion=mini_task_completion)

    # Second run: bottom-up traversal to enrich and complete the knowledge base
    print("Bottom-up traversal: Enriching knowledge base...")
    final_docs = bottom_up_traversal(tree, mini_task_completion=mini_task_completion)

    final_docs.reverse()

    # After both traversals, print out the enriched knowledge base

    return final_docs, missing_info


def python_code_to_tree_represent(folder_path, row=False):
    documentation_data = generate_usage_documentation(folder_path)
    docs_linear_path = create_linear_analysis_path(documentation_data)
    def retiver(doc, get_ref):
        return compare_with_generated_doc(doc, documentation_data,
                                                                  get_ref) if doc in docs_linear_path else None

    def helper(get_ref=None):
        for l_doc in docs_linear_path:
            if l_doc.get('name').lower() in ['os', 'print', 'async', 'asyncio', 'type', 'sys']:
                continue
            yield retiver(l_doc, get_ref=get_ref)

    if row:
        return helper, retiver, docs_linear_path, documentation_data

    return helper, docs_linear_path


def setup():
    nltk.download('punkt')
    nltk.download('stopwords')


def combine_elements(data):
    combined_results = []
    current_item = None

    for i in data:
        element_name = i['element']['name']

        # If no current item exists, start a new one
        if current_item is None:
            current_item = {
                'element': i['element'],
                'response': i.get('response', '')+'\n',
                'missing_data': i.get('missing_data', ['']),
                'source_code': i['element'].get('source_code', '')+'\n',
                'file': i['element'].get('file', '')+'\n',
                'references': set(i['element'].get('references', [])),
                'context_window': i.get('context_window', [])
            }

        # If the name matches the current item, we merge references
        if current_item['element']['name'] == element_name:
            current_item['references'].update(i['element'].get('references', []))
            current_item['missing_data'].extend(i.get('missing_data', []))
            current_item['response'] += i.get('response', '')+'\n'
            current_item['source_code'] += i['element'].get('source_code', '')+'\n'
            if current_item['file'] == '':
                current_item['file'] += i['element'].get('file', '')

        # If the name changes, we append the current item to results and start a new one
        else:
            combined_results.append(current_item)
            current_item = {
                'element': i['element'],
                'response': i.get('response', '')+'\n',
                'missing_data': i.get('missing_data', ['']),
                'source_code': i['element'].get('source_code', '')+'\n',
                'file': i['element'].get('file', '')+'\n',
                'references': set(i['element'].get('references', [])),
                'context_window': i.get('context_window', [])
            }

    # Add the last item after the loop
    if current_item is not None:
        combined_results.append(current_item)

    return combined_results


def dir_process(folder_path="utils", mini_task_completion=None, recursive=False, max_depth=None):
    if mini_task_completion is None:
        mini_task_completion = mini_task_completion_moc
    tree_data = build_code_tree(folder_path, recursive=recursive, max_depth=max_depth)
    data = process_tree_for_knowledge_base(tree_data, mini_task_completion=mini_task_completion)
    return data, combine_elements(data[0])


if __name__ == '__main__':
    folder_path_docs = r'C:\Users\Markin\Workspace\ToolBoxV2\toolboxv2\utils'
    for i in dir_process(folder_path_docs)[1]:
        print(i['element']['name'], i.keys())

