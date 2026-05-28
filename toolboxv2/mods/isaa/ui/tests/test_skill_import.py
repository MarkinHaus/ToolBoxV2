"""Tests for skill import/export helpers (zip round-trip).

These exercise the pure helpers in routes/skills.py without a live FastTB app.
"""
from __future__ import annotations

import io
import json
import zipfile
import unittest

from toolboxv2.mods.isaa.ui.routes import skills as skills_routes


class FakeSkill:
    def __init__(self, sid, name="n", triggers=None, instruction=""):
        self.id = sid
        self.name = name
        self.triggers = triggers or []
        self.instruction = instruction
        self.tools_used = []
        self.tool_groups = []
        self.source = "predefined"

    def to_dict(self):
        return {
            "id": self.id, "name": self.name, "triggers": self.triggers,
            "instruction": self.instruction, "tools_used": self.tools_used,
            "tool_groups": self.tool_groups, "source": self.source,
        }


class FakeSkillsManager:
    def __init__(self):
        self.skills = {}
        self._skill_embeddings_dirty = False


class ZipImportHelpers(unittest.TestCase):
    def _make_zip(self, entries: dict) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for fn, obj in entries.items():
                zf.writestr(fn, json.dumps(obj))
        return buf.getvalue()

    def test_file_bytes_from_dict(self):
        b = skills_routes._file_bytes({"data": b"hello"})
        self.assertEqual(b, b"hello")
        b2 = skills_routes._file_bytes({"data": "world"})
        self.assertEqual(b2, b"world")

    def test_file_bytes_from_string_and_bytes(self):
        self.assertEqual(skills_routes._file_bytes("abc"), b"abc")
        self.assertEqual(skills_routes._file_bytes(b"xyz"), b"xyz")

    def test_import_zip_skips_manifest(self):
        sm = FakeSkillsManager()
        data = self._make_zip({
            "_manifest.json": {"format": "isaa-skills-v1"},
            "s1.json": {"id": "s1", "name": "Skill 1", "triggers": ["a"], "instruction": "do x"},
        })
        # Patch Skill import: monkeypatch _skill_from_dict to use FakeSkill
        orig = skills_routes._skill_from_dict
        def fake_from_dict(mgr, d):
            if not d.get("id"):
                return False
            mgr.skills[d["id"]] = FakeSkill(d["id"], d.get("name", ""))
            return True
        skills_routes._skill_from_dict = fake_from_dict
        try:
            imported, errors = skills_routes._import_zip_bytes(sm, data, overwrite=True)
        finally:
            skills_routes._skill_from_dict = orig
        self.assertEqual(imported, ["s1"])
        self.assertEqual(errors, [])
        self.assertIn("s1", sm.skills)

    def test_import_zip_invalid_zip(self):
        sm = FakeSkillsManager()
        imported, errors = skills_routes._import_zip_bytes(sm, b"not a zip", overwrite=True)
        self.assertEqual(imported, [])
        self.assertTrue(errors and "invalid zip" in errors[0])

    def test_import_zip_list_of_skills(self):
        sm = FakeSkillsManager()
        data = self._make_zip({
            "bundle.json": [
                {"id": "a", "name": "A"},
                {"id": "b", "name": "B"},
            ],
        })
        orig = skills_routes._skill_from_dict
        def fake_from_dict(mgr, d):
            if not d.get("id"):
                return False
            mgr.skills[d["id"]] = FakeSkill(d["id"])
            return True
        skills_routes._skill_from_dict = fake_from_dict
        try:
            imported, errors = skills_routes._import_zip_bytes(sm, data, overwrite=True)
        finally:
            skills_routes._skill_from_dict = orig
        self.assertEqual(sorted(imported), ["a", "b"])


class GithubUrlParsing(unittest.TestCase):
    def test_unsupported_url_raises(self):
        sm = FakeSkillsManager()
        with self.assertRaises(ValueError):
            skills_routes._import_github(sm, "https://example.com/foo", overwrite=True)


if __name__ == "__main__":
    unittest.main()
