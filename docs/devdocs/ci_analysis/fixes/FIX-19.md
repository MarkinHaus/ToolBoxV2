# FIX-19 – aus Karte K-API-13

Paket: **P4-ENTSCHEIDUNG** (Braucht Entscheidung des Owners) · Fix-Typ: **A** · Karte: GETESTET · betrifft 3 Test-IDs

```
Fix:           FIX-19  (aus Karte K-API-13)
Voraussetzung: FIX-01, FIX-02 (ehrliches Signal und editierbare Installation), sonst keine
Dateien:       toolboxv2/utils/system/session.py; toolboxv2/tests/test_mini/test_system/test_session.py (nur Zeilen 49, 59, 80, 104); toolboxv2/tests/test_mini/test_cli_session.py (nur `test_long_username_truncated`)   (nur diese ändern)
Vorher lesen:  `session.py:100-215`; `toolboxv2/utils/toolbox.py:220-230, 476-487`; `toolboxv2/utils/extras/blobs.py` (`BlobFile`, Namensregel)
```

**Schritte:**

1. Patch aus diesem Auftrag anwenden (s. „Patch“). Gelingt `git apply --check` nicht, von Hand nach dem Patch.
2. Szenario-Prüfskript aus „Zusätzliche Abnahme“ ausführen.

**Nicht tun:** Die alte Datei nicht löschen, nur leeren (wie der Patch). Sitzung eines anderen Nutzers nie übernehmen. `test_login_valid_token_sets_session_valid` und `test_clear_session_token_deletes_blob_and_clears_attributes` nicht anpassen – offene Frage. Allgemein: kein `skip`, `xfail`, `continue-on-error`; keine Assertion abschwächen.

**Abnahme:**

- Repro → genau die 3 roten IDs aus „Nachher“; Szenario-Skript gibt die vier Zeilen aus „Nachher“ aus.
- Gate: `python docs/devdocs/ci_analysis/gate.py` → `neue rote IDs: 0` (keine rote ID, die nicht in `catalog.csv` steht).

**Rückfall:** wenn rot: stoppen, Ausgabe melden, nichts weiter ändern.

## Patch (getestet, `git apply`)

Als Datei außerhalb des Repos speichern, im Repo-Wurzelordner `git apply --check <datei>` und dann `git apply <datei>`.

```diff
diff --git a/toolboxv2/tests/test_mini/test_cli_session.py b/toolboxv2/tests/test_mini/test_cli_session.py
index dbecb36..c276a02 100644
--- a/toolboxv2/tests/test_mini/test_cli_session.py
+++ b/toolboxv2/tests/test_mini/test_cli_session.py
@@ -126,7 +126,8 @@ class TestSessionPathSafety(unittest.TestCase):
         sess = self._make_session("a" * 200)
         path = sess._get_blob_name()
         # Filename part (without extension) should be <= 32 + len("_session.json")
-        name_part = path.replace("_session.json", "")
+        # path is "user/<name>_session.json" (BlobFile needs "<blob_id>/<file>")
+        name_part = path.rsplit("/", 1)[-1].replace("_session.json", "")
         self.assertLessEqual(len(name_part), 32)
 
 
diff --git a/toolboxv2/tests/test_mini/test_system/test_session.py b/toolboxv2/tests/test_mini/test_system/test_session.py
index e5bc348..d63cf4d 100644
--- a/toolboxv2/tests/test_mini/test_system/test_session.py
+++ b/toolboxv2/tests/test_mini/test_system/test_session.py
@@ -46,7 +46,7 @@ class TestSessionStorage(unittest.TestCase):
         blob_name = self.sess._get_blob_name()
 
         # Assert
-        self.assertEqual(blob_name, "user_evil_______name__session.json")
+        self.assertEqual(blob_name, "user/user_evil_______name__session.json")
 
     def test_get_blob_name_empty_username_uses_default(self):
         # Arrange
@@ -56,7 +56,7 @@ class TestSessionStorage(unittest.TestCase):
         blob_name = self.sess._get_blob_name()
 
         # Assert
-        self.assertEqual(blob_name, "default_session.json")
+        self.assertEqual(blob_name, "user/default_session.json")
 
     @patch('toolboxv2.utils.system.session.BlobFile')
     @patch('toolboxv2.utils.system.session.Code')
@@ -77,7 +77,7 @@ class TestSessionStorage(unittest.TestCase):
         self.assertEqual(self.sess.access_token, "access_123")
 
         mock_blob_file.assert_called_with(
-            "testuser_session.json",
+            "user/testuser_session.json",
             mode="w",
             key="fake-device-key",
             storage=self.sess._storage
@@ -101,7 +101,7 @@ class TestSessionStorage(unittest.TestCase):
         }
 
         # Temporarily clear base to test restoration
-        del os.environ["TOOLBOXV2_REMOTE_BASE"]
+        os.environ.pop("TOOLBOXV2_REMOTE_BASE", None)
 
         # Act
         data = self.sess._load_session_token()
diff --git a/toolboxv2/utils/system/session.py b/toolboxv2/utils/system/session.py
index fa91905..1bae938 100644
--- a/toolboxv2/utils/system/session.py
+++ b/toolboxv2/utils/system/session.py
@@ -116,13 +116,41 @@ class Session(metaclass=Singleton):
             storage_directory=str(storage_path)
         )
 
+    # Until 2026-11 every user on a device shared this one file; read once and migrated.
+    _LEGACY_BLOB_NAME = "user/_session.json"
+
     def _get_blob_name(self) -> str:
-        """Determines the blob name (filename)."""
+        """Determines the blob name (filename): one session file per user."""
         # Default to 'default' if no username, allows basic session persistence without login
         safe_name = self.username or "default"
         # Sanitize filename
         safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_name)[:32]
-        return f"user/_session.json"
+        # BlobFile needs "<blob_id>/<file>"; the blob is "user", one file per user inside it
+        return f"user/{safe_name}_session.json"
+
+    def _migrate_legacy_session(self) -> Optional[dict]:
+        """Move a session from the old device-wide file to this user's file (once).
+
+        Only when the old file belongs to this user (or has no owner recorded);
+        another user's old session is left untouched.
+        """
+        key = self._get_encryption_key()
+        try:
+            with BlobFile(self._LEGACY_BLOB_NAME, mode="r", key=key, storage=self._storage) as blob:
+                data = blob.read_json()
+        except Exception:
+            return None
+        if not data:
+            return None
+        owner = data.get("username") or ""
+        if self.username and owner and owner != self.username:
+            return None
+        with BlobFile(self._get_blob_name(), mode="w", key=key, storage=self._storage) as blob:
+            blob.write_json(data)
+        with BlobFile(self._LEGACY_BLOB_NAME, mode="w", key=key, storage=self._storage) as blob:
+            blob.write_json({})
+        get_logger().info(f"Session migrated from {self._LEGACY_BLOB_NAME} to {self._get_blob_name()}")
+        return data
 
     def _get_encryption_key(self) -> str:
         """
@@ -163,9 +191,14 @@ class Session(metaclass=Singleton):
         try:
             blob_name = self._get_blob_name()
 
-            with BlobFile(blob_name, mode="r", key=self._get_encryption_key(), storage=self._storage) as blob:
-                session_data = blob.read_json()
+            try:
+                with BlobFile(blob_name, mode="r", key=self._get_encryption_key(), storage=self._storage) as blob:
+                    session_data = blob.read_json()
+            except Exception:
+                session_data = None
 
+            if not session_data:
+                session_data = self._migrate_legacy_session()
             if not session_data:
                 return None
 
```

## Test-IDs dieses Auftrags

```
toolboxv2/tests/test_mini/test_cli_session.py::TestSessionPathSafety::test_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_empty_username_uses_default
toolboxv2/tests/test_mini/test_system/test_session.py::TestSessionStorage::test_get_blob_name_sanitizes_special_characters
```
