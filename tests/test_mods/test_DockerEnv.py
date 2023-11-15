import unittest
import time

from toolboxv2 import App
from toolboxv2.utils.toolbox import ApiOb


class TestRestrictor(unittest.TestCase):
    t0 = None
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.perf_counter()
        cls.app = App('test-DockerEnv')
        cls.app.mlm = 'I'
        cls.app.debug = True
        cls.app.inplace_load('dockerEnv')
        cls.tool = cls.app.get_mod('dockerEnv')
        cls.app.new_ac_mod('dockerEnv')

    @classmethod
    def tearDownClass(cls):
        cls.app.logger.info('Closing APP')
        cls.app.config_fh.delete_file()
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f'Accomplished in {time.perf_counter() - cls.t0}')

    def test_show_version(self):
        command = ApiOb(
            data={'username': 'test', 'password': 'test', 'email': 'test@test.com', 'invitation': 'test'},
            token='')
        res = self.app.run_function('Version', [command, ])
        self.assertEqual(res, '0.0.2')

    def test_show_image(self):
        for image in self.tool.client.images.list():
            print(image.id)

    def test_create_container(self):
        # Teste die "create_container"-Methode
        image = "ubuntu:latest"
        name = "my-container"
        container_id = self.tool.create_container(image, name)
        print(container_id)
        self.assertIsNotNone(container_id)
        # container_st = self.tool.start_container(name)
        # print(container_st)
        # self.assertIsNotNone(container_st)
        # self.assertEqual(container_st.id, container_id)

        self.tool.stop_container(name)

    def test_run_command(self):
        # Teste die "run_command_in_container"-Methode
        name = "my-container"
        container_id = self.tool.get_container(name)
        print(container_id)
        self.assertIsNotNone(container_id)

        # time.sleep(2)

        command = "echo 'Hello, World!'"
        self.tool.run_command_in_container(container_id.id, command)

        self.tool.pause_container(name)

        # Überprüfe, ob die Ausgabe des Befehls im Container erwartet wird

    def test_commit_container(self):
        # Teste die "commit_container"-Methode
        image = "ubuntu"
        name = "my-container"
        container_id = self.tool.create_container(image, name)
        self.assertIsNotNone(container_id)

        command = "echo 'Hello, World!' >> test.txt"
        self.tool.run_command_in_container(container_id, command)

        self.tool.commit_container(self.tool.get_container(name), "")

    def test_commit_container_stram(self):
        # Teste die "commit_container"-Methode
        image = "ubuntu"
        name = "my-container"
        container_id = self.tool.create_container(image, name)
        self.assertIsNotNone(container_id)

        command = "ls /bin"
        self.tool.run_command_in_container(container_id, command, stream=True)

        self.tool.commit_container(self.tool.get_container(name), "")
    def test_NONELIVE(self):
        # Teste die "commit_container"-Methode
        image = "ubuntu:latest"
        name = "my-container7"
        container_id = self.tool.create_container(image, name)
        self.assertIsNotNone(container_id)

        # self.tool.session_cli_user(container_id)

        self.tool.commit_container(self.tool.get_container(name), "")

    def test_reset_container_to_commit(self):
        # Teste die "reset_container_to_commit"-Methode
        #image = "ubuntu"
        #name = "my-container"
        #container_id = self.tool.create_container(image, name)
        #self.assertIsNotNone(container_id)
#
        #command = "echo 'Hello, World!' >> test.txt"
        #self.tool.run_command_in_container(container_id, command)
#
        #message = "Test commit"
        #commit_id = self.tool.commit_container(container_id, message)
        #self.assertIsNotNone(commit_id)
#
        #new_container_id = self.tool.reset_container_to_commit(container_id, commit_id)
        #self.assertIsNotNone(new_container_id)
        pass

    def test_create_docker_image_from_folder(self):
        # Teste die "create_docker_image_from_folder"-Methode
        #folder_path = r"E:\Markin\D\project_py\ToolBoxV2\toolboxv2\utils"
        #image_name = "my-docker-image-utils"
        #self.tool.create_docker_image_from_folder(folder_path, image_name)

        # Überprüfe, ob das Bild erfolgreich erstellt wurde
        pass

    def test_overwrite_folder_from_container(self):
        # Teste die "overwrite_folder_from_container"-Methode
        #image = "ubuntu"
        #name = "my-container"
        #container_id = self.tool.create_container(image, name)
        #self.assertIsNotNone(container_id)
#
        #source_path = r"C:\Users\Markin\Isaa\Prototyp"
        #target_path = r"C:\Users\Markin\Isaa\Neuer Ordner"
        #command = "echo 'Hello, World!' >> test.txt"
        #self.tool.run_command_in_container(container_id, command)
        #self.tool.overwrite_folder_from_container(container_id, source_path, target_path)
        pass

        # Überprüfe, ob das Verzeichnis im Container erfolgreich überschrieben wurde

