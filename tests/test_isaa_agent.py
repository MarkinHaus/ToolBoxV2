import json
import os
import time
import unittest

from langchain.vectorstores.base import VectorStoreRetriever

from toolboxv2 import App, get_logger
from toolboxv2.mods.isaa.AgentUtils import AIContextMemory, ObservationMemory, ShortTermMemory, MemoryModel, \
    PyEnvEval, get_token_mini, get_max_token_fom_model_name, get_price, AgentConfig, anything_from_str_to_dict, \
    parse_json_with_auto_detection, AgentChain


class TestAgentChain(unittest.TestCase):

    def setUp(self):
        print("testing agent chain")
        self.agent_chain = AgentChain()

    def test_add(self):
        self.agent_chain.add('test_chain', [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])
        self.assertIn('test_chain', self.agent_chain.chains)

    def test_remove(self):
        self.agent_chain.add('test_chain', [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])
        self.agent_chain.remove('test_chain')
        self.assertNotIn('test_chain', self.agent_chain.chains)

    def test_add_task(self):
        self.agent_chain.add('test_chain', [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])
        self.agent_chain.add_task('test_chain', {'use': 'test_use2', 'name': 'test_name2', 'args': 'test_args2'})
        self.assertEqual(len(self.agent_chain.chains['test_chain']), 2)

    def test_remove_task(self):
        self.agent_chain.add('test_chain', [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])
        self.agent_chain.add_task('test_chain', {'use': 'test_use2', 'name': 'test_name2', 'args': 'test_args2'})
        self.agent_chain.remove_task('test_chain', 0)
        self.assertEqual(len(self.agent_chain.chains['test_chain']), 1)

    def test_get_chain(self):
        self.agent_chain.add('test_chain', [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])
        self.assertEqual(self.agent_chain.get('test_chain'),
                         [{'use': 'test_use', 'name': 'test_name', 'args': 'test_args'}])


class TestAIContextMemory(unittest.TestCase):

    def setUp(self):
        self.ai_context_memory = AIContextMemory()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(".config/system.infos"):
            os.remove(".config/system.infos")

    def test_init(self):
        self.assertIsInstance(self.ai_context_memory.memory, dict)
        self.assertIsInstance(self.ai_context_memory.vector_store, dict)
        self.assertEqual("", self.ai_context_memory.extra_path)

    def test_get_sto_bo(self):
        result = self.ai_context_memory.get_sto_bo('test')
        self.assertIsInstance(result, dict)
        self.assertEqual(result['text'], [])
        self.assertEqual(result['full-text-len'], 0)
        self.assertEqual(result['vectors'], [])
        self.assertIsNone(result['db'])
        self.assertEqual(result['len-represent'], 0)
        self.assertEqual(result['represent'], [])

    def test_cleanup_list(self):
        data = ['   ', 'test', '   test   ', '']
        result = self.ai_context_memory.cleanup_list(data)
        self.assertEqual(result, [])
        data = ['   ', 'test123456789', '   test123456789   ', '']
        result = self.ai_context_memory.cleanup_list(data)
        self.assertEqual(result, ['test123456789', 'test123456789'])

    def test_add_data(self):
        self.ai_context_memory.add_data('test', 'data')
        self.ai_context_memory.add_data('test', ['data'])
        self.assertEqual([], self.ai_context_memory.vector_store['test']['vectors'])
        self.ai_context_memory.add_data('test', ['data1234567890'])
        self.assertIn('test', self.ai_context_memory.vector_store.keys())
        self.assertNotEqual([], self.ai_context_memory.vector_store['test']['vectors'])

    def test_get_retriever(self):
        self.ai_context_memory.add_data('test', ['data'])
        result = self.ai_context_memory.get_retriever('test')
        self.assertIsInstance(result, VectorStoreRetriever)

    def test_search(self):
        self.ai_context_memory.add_data('test2', ['data1234567890'])
        get_logger().info(f"test_search {self.ai_context_memory.vector_store['test2']['vectors']=}")
        result = self.ai_context_memory.search('test2', 'data')
        get_logger().info(f"test_search {result=}")
        get_logger().info(f"test_search {self.ai_context_memory.vector_store['test2']['full-text-len'] =}")
        print(result)
        self.assertEqual(result[0][0].page_content, 'data1234567890')

    def test_get_context_for(self):
        self.ai_context_memory.add_data('test3', ['data1234567890'])
        get_logger().info(f"test_get_context_for {self.ai_context_memory.vector_store['test3']['vectors']=}")
        result = self.ai_context_memory.get_context_for('data')
        get_logger().info(f"test_get_context_for {result=}")
        print(result)
        self.assertIn('data1234567890', result)


class TestObservationMemory(unittest.TestCase):
    t0 = None
    observation_memory = None
    isaa = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-ObservationMemory')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.load_mod('isaa')
        cls.isaa = cls.app.get_mod('isaa')
        cls.app.new_ac_mod('isaa')
        cls.observation_memory = ObservationMemory(cls.isaa)

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')
        del cls.isaa
        del cls.observation_memory

    def test_info(self):
        info = self.observation_memory.info()
        self.assertIsInstance(info, str)
        self.assertIn(str(self.observation_memory.max_length), info)

    def test_text_property(self):
        self.observation_memory.max_length = 0
        self.observation_memory.cut()
        # self.assertIn("No memory data", self.observation_memory.text) spooky
        self.observation_memory.max_length = 0
        self.observation_memory.text = "This is a test text"
        self.assertGreater(len(self.observation_memory.text), 0)
        self.observation_memory.max_length = 1000
        self.observation_memory.text = "This is a test text"
        # self.assertEqual(self.observation_memory.text, "This is a test text\n")

    def test_cut(self):
        self.observation_memory.max_length = 1
        self.observation_memory.cut()
        self.assertLessEqual(self.observation_memory.tokens, self.observation_memory.max_length)
        self.observation_memory.tokens = 0


class TestShortTermMemory(unittest.TestCase):
    short_term_memory = None
    t0 = None
    name = "TestShortTermMemory"
    isaa = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-ShortTermMemory')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.load_mod('isaa')
        cls.isaa = cls.app.get_mod('isaa')
        cls.app.new_ac_mod('isaa')
        cls.short_term_memory = ShortTermMemory(cls.isaa, cls.name)

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')
        del cls.isaa
        del cls.short_term_memory

    def test_init(self):
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        self.assertEqual(self.short_term_memory.isaa, self.isaa)
        self.assertEqual(self.short_term_memory.name, self.name)
        self.assertEqual(self.short_term_memory.tokens, 0)
        self.assertEqual(self.short_term_memory.max_length, 2000)
        self.assertEqual(self.short_term_memory.model_name, MemoryModel)

    def test_set_name(self):
        new_name = 'new_test_name'
        self.short_term_memory.set_name(new_name)
        self.assertEqual(self.short_term_memory.name, new_name)
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)

    def test_info(self):
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        info = self.short_term_memory.info()
        self.assertIn('tokens=0', info)
        self.assertIn('max_length=2000', info)
        self.assertIn(f"model_name='{MemoryModel}'", info)

    def test_cut(self):
        ShortTermMemory.add_to_static = []
        ShortTermMemory.memory_data = []
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        self.short_term_memory.tokens = 3000
        self.short_term_memory.memory_data = [{'token-count': 10, 'data': 'test_data'}]
        self.short_term_memory.cut()
        self.assertEqual(self.short_term_memory.tokens, 2990)
        self.assertEqual(self.short_term_memory.memory_data, [])
        self.assertEqual(self.short_term_memory.add_to_static, [{'token-count': 10, 'data': 'test_data'}])
        self.short_term_memory.tokens = 0

    def test_clear_to_collective(self):
        ShortTermMemory.add_to_static = []
        ShortTermMemory.memory_data = []
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        self.short_term_memory.tokens = 30
        self.short_term_memory.memory_data = [{'token-count': 10, 'data': 'test_data'}]
        self.short_term_memory.clear_to_collective()
        self.assertEqual(self.short_term_memory.tokens, 20)
        self.assertEqual(self.short_term_memory.memory_data, [])
        self.assertEqual(self.short_term_memory.add_to_static, [{'token-count': 10, 'data': 'test_data'}])
        self.short_term_memory.tokens = 0

    def test_text(self):
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        self.short_term_memory.memory_data = [{'data': 'test_data1'}, {'data': 'test_data2'}]
        self.assertEqual(self.short_term_memory.text, 'test_data1\ntest_data2\n')
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)

    def test_text_setter(self):
        self.short_term_memory = ShortTermMemory(self.isaa, self.name)
        self.short_term_memory.text = 'test_data'
        self.assertEqual(self.short_term_memory.memory_data[0]['data'], 'test_data')
        self.assertGreater(self.short_term_memory.tokens, 0)


class TestPyEnvEval(unittest.TestCase):

    def setUp(self):
        self.py_env_eval = PyEnvEval()

    def test_eval_code(self):
        code = '1 + 1'
        result = self.py_env_eval.eval_code(code)
        self.assertEqual(result, 'Ergebnis: 2')

    def test_get_env(self):
        self.py_env_eval.eval_code('x = 10')
        env = self.py_env_eval.get_env()
        self.assertIn('x: 10', env)

    def test_format_output(self):
        output = self.py_env_eval.format_output('Hello, World!')
        self.assertEqual(output, 'Ergebnis: Hello, World!')

    def test_format_env(self):
        self.py_env_eval.local_env = {'x': 10, 'y': 20}
        env = self.py_env_eval.format_env(self.py_env_eval.local_env)
        self.assertEqual(env, 'x: 10\ny: 20')

    def test_run_and_display(self):
        code = 'x = 10\ny = 20\n_ =x + y'
        result = self.py_env_eval.run_and_display(code)
        self.assertIn('Startzustand:', result)
        self.assertIn('Endzustand:', result)
        self.assertIn('Ausführungsergebnis:', result)
        self.assertIn('x: 10', result)
        self.assertIn('y: 20', result)
        self.assertIn('30', result)

    def tearDown(self):
        del self.py_env_eval


class TestAgentUtilFunctions(unittest.TestCase):

    def test_get_token_mini(self):
        text = "Hello, world!"
        model_name = "gpt-3.5-turbo-0613"
        result = get_token_mini(text, model_name)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 0)

    def test_get_max_token_fom_model_name(self):
        model = "gpt-3.5-turbo-0613"
        result = get_max_token_fom_model_name(model)
        self.assertIsInstance(result, int)
        self.assertGreater(result, 1000)

    def test_get_price(self):
        fit = 2048
        result = get_price(fit)
        self.assertIsInstance(result, list)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

        self.assertGreater(result[0], 0)
        self.assertGreater(result[1], 0)


class TestAgentConfig(unittest.TestCase):
    agent_config = None
    t0 = None
    isaa = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App('test-ObservationMemory')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.load_mod('isaa')
        cls.isaa = cls.app.get_mod('isaa')
        cls.app.new_ac_mod('isaa')
        cls.agent_config = AgentConfig(cls.isaa)

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.time() - cls.t0}')
        del cls.isaa
        del cls.agent_config

    def test_init(self):
        self.assertEqual(self.agent_config.isaa, self.isaa)
        self.assertEqual(self.agent_config.name, 'agentConfig')
        self.assertEqual(self.agent_config.mode, 'talk')
        self.assertEqual(self.agent_config.model_name, 'gpt-3.5-turbo-0613')

    def test_set_mode(self):
        self.agent_config.set_mode('planning')
        self.assertEqual(self.agent_config.mode, 'planning')

    def test_set_completion_mode(self):
        self.agent_config.set_completion_mode('text')
        self.assertEqual(self.agent_config.completion_mode, 'text')

    def test_set_temperature(self):
        self.agent_config.set_temperature(0.5)
        self.assertEqual(self.agent_config.temperature, 0.5)

    def test_add_task(self):
        self.agent_config.add_task('new task')
        self.assertIn('new task', self.agent_config.task_list)

    def test_mark_task_done(self):
        self.agent_config = AgentConfig(self.isaa)
        self.agent_config.add_task('new task')
        self.assertIn('new task', self.agent_config.task_list)
        self.assertNotIn('new task', self.agent_config.task_list_done)

        self.agent_config.mark_task_done('new task')
        self.assertIn('new task', self.agent_config.task_list_done)
        self.assertNotIn('new task', self.agent_config.task_list)

    def test_set_short_term_memory(self):
        short_mem = ShortTermMemory(self.isaa, 'test')
        self.agent_config.set_short_term_memory(short_mem)
        self.assertEqual(self.agent_config.short_mem, short_mem)

    def test_set_context(self):
        context = ShortTermMemory(self.isaa, 'test')
        self.agent_config.set_context(context)
        self.assertEqual(self.agent_config.context, context)

    def test_set_observation_memory(self):
        obser_mem = ObservationMemory(self.isaa)
        self.agent_config.set_observation_memory(obser_mem)
        self.assertEqual(self.agent_config.observe_mem, obser_mem)

    def test_set_model_name(self):
        self.agent_config.set_model_name('gpt-3')
        self.assertEqual(self.agent_config.model_name, 'gpt-3')


class TestAnythingFromStrToDict(unittest.TestCase):
    def test_json_string(self):
        data = '{"key": "value", "expected_key": "expected_value"}'
        expected_keys = {"expected_key": "expected_value2"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [{"expected_key": "expected_value", "key": "value"}])

    def test_json_string_in_list(self):
        data = '[{"key": "value", "expected_key": "expected_value"}]'
        expected_keys = {"expected_key": "expected_value2"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [{"expected_key": "expected_value", "key": "value"}])

    def test_json_string_in_list2(self):
        data = ('[{"key": "value", "expected_key": "expected_value"}, {"key": "value", "expected_key": '
                '"expected_value"}]')
        expected_keys = {"expected_key": "expected_value2"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [{"expected_key": "expected_value", "key": "value"},
                                  {"key": "value", "expected_key": "expected_value"}])

    def test_non_json_string(self):
        data = "This is not a JSON string"
        expected_keys = {"expected_key": "expected_value"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [{"expected_key": "This is not a JSON string"}])

    def test_empty_string(self):
        data = ""
        expected_keys = {"expected_key": "expected_value"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [])

    def test_string_mini_task(self):
        data = ""
        mini_task = lambda x: "{'x':0}"
        expected_keys = {"expected_key": "expected_value"}
        result = anything_from_str_to_dict(data, expected_keys, mini_task)
        self.assertEqual(result, [])

    def test_string_with_multiple_json_objects(self):
        data = '{"key1": "value1"} {"key2": "value2"}'
        expected_keys = {"expected_key": "expected_value"}
        result = anything_from_str_to_dict(data, expected_keys)
        self.assertEqual(result, [{"expected_key": "expected_value", "key1": "value1"},
                                  {"expected_key": "expected_value", "key2": "value2"}])

    def test_case_1(self):
        # Arrange
        input_data = '{"key": "value"}'
        expected_keys = {"expected_key": "expected_value"}
        expected_output = [{"key": "value", "expected_key": "expected_value"}]

        # Act
        actual_output = anything_from_str_to_dict(input_data, expected_keys)

        # Assert
        self.assertEqual(actual_output, expected_output)

    def test_case_2(self):
        # Arrange
        input_data = 'This is not a JSON string'
        expected_keys = {"expected_key": "expected_value"}
        expected_output = [{"expected_key": "This is not a JSON string"}]

        # Act
        actual_output = anything_from_str_to_dict(input_data, expected_keys)

        # Assert
        self.assertEqual(actual_output, expected_output)

    def test_case_3(self):
        # Arrange
        input_data = '{"key": "value"} {"key2": "value2"}'
        expected_keys = {"expected_key": "expected_value"}
        expected_output = [{"key": "value", "expected_key": "expected_value"},
                           {"key2": "value2", "expected_key": "expected_value"}]

        # Act
        actual_output = anything_from_str_to_dict(input_data, expected_keys)

        # Assert
        self.assertEqual(actual_output, expected_output)

    def test_case_4(self):
        input_data = " {'Action':'find','Inputs':{'directory':'.','pattern':'mods//isaa'}}"
        expected_output = [{'Action': 'find', 'Inputs': {'directory': '.', 'pattern': 'mods//isaa'}}]

        # Act
        actual_output = anything_from_str_to_dict(input_data)

        # Assert
        self.assertEqual(actual_output, expected_output)

    def test_live_input(self):
        expected_keys = {"Action": "", "Inputs": ""}
        test_json = """{'Action':'spawn_agent','Inputs':'{"Name": "Environment_Exploration_Agent","Personal":"Resourceful, Collaborative, Inquisitive", "Goals": "Explore the environment and the source code folder \'toolboxv2\'", "Capabilities": "Skills in file and directory exploration"}'}"""
        expected_result = [{'Action': 'spawn_agent', 'Inputs': {"Name": "Environment_Exploration_Agent",
                                                                "Personal": "Resourceful, Collaborative, Inquisitive",
                                                                "Goals": "Explore the environment and the source code folder \'toolboxv2\'",
                                                                "Capabilities": "Skills in file and directory exploration"}}]
        actual_result = anything_from_str_to_dict(test_json, expected_keys)

        print(actual_result, type(actual_result))

        # Assert
        self.assertEqual(expected_result, actual_result)

    def test_live_input_python_file(self):
        expected_keys = {"Action": "", "Inputs": ""}
        test_json = """{'Action':'write','Inputs':{'filename':'list_files.py','text':'import os\n\ndef list_files(path):\n    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]\n\ndef list_folders(path):\n    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]\n\ndef list_hidden(path):\n    return [f for f in os.listdir(path) if f.startswith(".")]\n'}}"""
        expected_result = [{'Action': 'write', 'Inputs': {'filename': 'list_files.py',
                                                          'text': 'import os\n\ndef list_files(path):\n    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]\n\ndef list_folders(path):\n    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]\n\ndef list_hidden(path):\n    return [f for f in os.listdir(path) if f.startswith(\'.\')]\n'}}]
        actual_result = anything_from_str_to_dict(test_json, expected_keys)

        print(actual_result, type(actual_result))

        # Assert
        self.assertEqual(expected_result[0]['Action'], actual_result[0]['Action'])
        self.assertEqual(expected_result[0]['Inputs']['text'], actual_result[0]['Inputs']['text'])
        self.assertEqual(expected_result[0]['Inputs']['filename'], actual_result[0]['Inputs']['filename'])


class TestParseJsonWithAutoDetection(unittest.TestCase):
    def test_dictionary(self):
        json_string = '{"name": "John", "age": 30, "city": "New York"}'
        expected_result = {"name": "John", "age": 30, "city": "New York"}
        self.assertEqual(parse_json_with_auto_detection(json_string), expected_result)

    def test_dictionary_in_d(self):
        json_string = '{"name": "John", "age": 30, "citys": {"city": "New York"}}'
        expected_result = {"name": "John", "age": 30, "citys": {"city": "New York"}}
        self.assertEqual(parse_json_with_auto_detection(json_string), expected_result)

    def test_list(self):
        json_string = '["apple", "banana", "cherry"]'
        expected_result = ["apple", "banana", "cherry"]
        self.assertEqual(parse_json_with_auto_detection(json_string), expected_result)

    def test_single_value(self):
        json_string = "hello"
        expected_result = "hello"
        self.assertEqual(parse_json_with_auto_detection(json_string), expected_result)

    def test_non_json_string(self):
        non_json_string = "This is a normal string"
        expected_result = "This is a normal string"
        self.assertEqual(parse_json_with_auto_detection(non_json_string), expected_result)
