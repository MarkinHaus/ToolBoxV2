import time
import unittest

from toolboxv2 import App


class TestIsaaIDE(unittest.TestCase):
    t0 = 0
    app = None

    @classmethod
    def setUpClass(cls):
        # Code, der einmal vor allen Tests ausgeführt wird
        cls.t0 = time.time()
        cls.app = App("test")
        cls.app.mlm = "I"
        cls.app.debug = True
        cls.app.inplace_load("isaa_ide", "toolboxv2.mods_dev.")
        cls.app.new_ac_mod("isaa_ide")
        cls.fm = cls.app.AC_MOD
        cls.file_name = "test_file.txt"
        cls.folder_name = "test_folder"

    @classmethod
    def tearDownClass(cls):
        cls.app.remove_all_modules()
        cls.app.save_exit()
        cls.app.exit()
        cls.app.logger.info(f"Accomplished in {time.time() - cls.t0}")

    def test_create(self):
        # Test file creation
        self.assertIsNotNone(self.fm.create([self.file_name]))

        # Test folder creation
        self.assertIsNotNone(self.fm.create([self.folder_name]))

    def test_delete(self):
        # Test file deletion
        self.fm.create([self.file_name])
        self.assertEqual(self.fm.delete([self.file_name]), f"File deleted at {self.fm.scope + self.file_name}")

        # Test folder deletion
        self.fm.create([self.folder_name])
        self.assertEqual(self.fm.delete([self.folder_name]), f"Directory deleted at {self.fm.scope + self.folder_name}")

    def test_list(self):
        self.fm.create([self.file_name])
        # Test listing of current directory
        self.assertIn(self.file_name, self.fm.list([""]))

    def test_move(self):
        # Test file move
        new_path = "new_folder/" + self.file_name
        self.fm.create([self.file_name])
        self.assertEqual(self.fm.move([self.file_name, new_path]),
                         f"{self.fm.scope + self.file_name} moved to {self.fm.scope + new_path}")
        self.fm.delete(new_path)

        # Test folder move
        new_path = "new_folder/" + self.folder_name
        self.fm.create([self.folder_name])
        self.assertEqual(self.fm.move([self.folder_name, new_path]),
                         f"{self.fm.scope + self.folder_name} moved to {self.fm.scope + new_path}")
        self.fm.delete(new_path)

    def test_insert_edit(self):
        # Test file insert-edit
        self.fm.create(self.file_name)
        self.assertEqual(self.fm.insert_edit([self.file_name, 0, 1, "Hello, World!"]), "File content updated")
        self.assertEqual(self.fm.read([self.file_name]), "Hello, World!\n")
        self.fm.delete(self.file_name)

    def test_search(self):
        # Test file search
        self.fm.create(self.file_name)
        self.fm.insert_edit([self.file_name, 0, 1, "Hello, World!"])
        print("<||>", self.fm.search([self.file_name, "Hel"]))
        self.assertIn("Found", self.fm.search([self.file_name, "Hel"]))
        self.fm.delete([self.file_name])

    def test_copy(self):
        # Test file copy
        new_path = "new_folder/" + self.file_name
        self.fm.create(self.file_name)
        self.assertEqual(self.fm.copy([self.file_name, new_path]), f"Copied {self.file_name} to {new_path}")
        self.fm.delete([self.file_name])
        self.fm.delete([new_path])
