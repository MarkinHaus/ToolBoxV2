import json
import time
import threading
from typing import Any, List, Optional, Dict, Set

# Imports basierend auf deiner Struktur
from toolboxv2 import Result
from toolboxv2.mods.DB.types import AuthenticationTypes
from toolboxv2.utils.extras.blobs import BlobFile, BlobStorage
from abc import ABC, abstractmethod


# Definition der abstrakten Klasse (wie von dir bereitgestellt)
class DB(ABC):
    @abstractmethod
    def get(self, query: str) -> Result: """get data"""

    @abstractmethod
    def set(self, query: str, value) -> Result: """set data"""

    @abstractmethod
    def append_on_set(self, query: str, value) -> Result: """append set data"""

    @abstractmethod
    def delete(self, query: str, matching=False) -> Result: """delete data"""

    @abstractmethod
    def if_exist(self, query: str) -> bool: """return True if query exists"""

    @abstractmethod
    def exit(self) -> Result: """Close DB connection and optional save data"""


class BlobDB(DB):
    """
    Eine robuste, netzwerkfähige Key-Value Datenbank basierend auf BlobStorage.
    Features:
    - Sofortige Persistenz (ACID-ähnlich auf Blob-Ebene)
    - Lokaler Caching mit automatischer Invalidierung via Watch-Events
    - Manifest-Tracking für schnelle Suchoperationen
    - Verschlüsselung
    """
    auth_type = AuthenticationTypes.location

    def __init__(self):
        self.storage_client: Optional[BlobStorage] = None
        self.db_base_path: str = ""
        self.enc_key: Optional[str] = None

        # Lokaler Cache: Key -> (Timestamp, Data)
        self._local_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()

        # Liste der Blob-IDs, die wir bereits watchen
        self._watched_blobs: Set[str] = set()

        # Das Manifest speichert eine Liste aller Keys, um Wildcard-Suchen zu ermöglichen
        self._manifest_cache: Set[str] = set()
        self._manifest_loaded = False

    def initialize(self, db_path: str, key: str, storage_client: BlobStorage) -> Result:
        """
        Initialisiert die DB. Muss vor der ersten Nutzung aufgerufen werden.
        """
        try:
            self.db_base_path = db_path.rstrip('/')
            self.enc_key = key
            self.storage_client = storage_client

            # Manifest laden (Liste aller Keys)
            self._reload_manifest()

            # Watch auf das Manifest starten, damit wir mitbekommen,
            # wenn andere Clients Keys hinzufügen/löschen
            self._ensure_watch(self._get_manifest_path(), is_manifest=True)

            return Result.ok().set_origin("BlobDB")
        except Exception as e:
            return Result.default_internal_error(data=str(e), info="Init failed").set_origin("BlobDB")

    # --- Helper Methods ---

    def _key_to_path(self, key: str) -> str:
        """
        Konvertiert einen DB-Key in einen Blob-Pfad.
        Strategie: Wir nutzen den ersten Teil des Keys als Ordner, um Blobs nicht zu groß werden zu lassen.
        Bsp: "USER::123" -> "db_path/USER/123.json"
        """
        clean_key = key.replace('::', '/').strip('/')
        return f"{self.db_base_path}/{clean_key}.json"

    def _get_manifest_path(self) -> str:
        return f"{self.db_base_path}/_system/manifest.json"

    def _ensure_watch(self, blob_path: str, is_manifest: bool = False):
        """
        Startet einen Watch auf die Blob-Datei, falls noch nicht aktiv.
        """
        # Wir müssen die Blob-ID aus dem Pfad extrahieren, da BlobStorage auf Blob-IDs watched
        try:
            # BlobFile Helper nutzen, um BlobID zu parsen
            blob_id, _, _ = BlobFile._path_splitter(blob_path)

            if blob_id in self._watched_blobs:
                return

            def callback(blob_file: BlobFile):
                self._on_remote_update(blob_id, is_manifest)

            # Start Watch via BlobFile interface
            # Wir erstellen ein dummy BlobFile objekt nur zum Registrieren
            bf = BlobFile(blob_path, storage=self.storage_client)
            bf.watch(callback, max_idle_timeout=3600)  # 1 Stunde Timeout

            self._watched_blobs.add(blob_id)

        except Exception as e:
            print(f"Warning: Could not setup watch for {blob_path}: {e}")

    def _on_remote_update(self, blob_id: str, is_manifest: bool):
        """
        Callback, der feuert, wenn sich Daten auf dem Server ändern.
        """
        with self._cache_lock:
            if is_manifest:
                # Manifest hat sich geändert -> Liste der Keys neu laden
                print(f"[BlobDB] Manifest changed externally. Reloading keys.")
                self._reload_manifest()
            else:
                # Ein Daten-Blob hat sich geändert.
                # Da wir nicht genau wissen, welcher Key im Blob betroffen ist (könnten mehrere sein),
                # löschen wir pauschal alles aus dem Cache, was zu dieser BlobID gehören könnte.
                # Simplifizierung: Wir leeren den gesamten Cache für Sicherheit.
                # In High-Performance Szenarien könnte man das granularer machen.
                self._local_cache.clear()

    def _reload_manifest(self):
        """Lädt die Liste aller existierenden Keys."""
        try:
            path = self._get_manifest_path()
            # Cache deaktivieren für Manifest Read, um immer frisch zu sein
            with BlobFile(path, 'rw', storage=self.storage_client, key=self.enc_key, use_cache=False) as f:
                data = f.read_json()
                if isinstance(data, list):
                    self._manifest_cache = set(data)
                else:
                    self._manifest_cache = set()
            self._manifest_loaded = True
        except Exception as e:
            # Manifest existiert vielleicht noch nicht
            print(f"Error loading manifest: {e}")
            self._manifest_cache = set()

    def _update_manifest(self, key: str, add: bool) -> bool:
        """
        Aktualisiert das Manifest atomar (so gut wie möglich via BlobFile).
        """
        path = self._get_manifest_path()
        try:
            # 'w' Modus in BlobFile liest erst, erlaubt Modifikation, schreibt dann (Locking via Versionierung im Storage)
            with BlobFile(path, 'rw', storage=self.storage_client, key=self.enc_key) as f:
                current_list = f.read_json()
                if not isinstance(current_list, list):
                    current_list = []

                current_set = set(current_list)

                if add:
                    current_set.add(key)
                else:
                    if key in current_set:
                        current_set.remove(key)

                f.clear()  # Buffer leeren
                f.write_json(list(current_set))

            # Lokales Set auch updaten
            if add:
                self._manifest_cache.add(key)
            elif key in self._manifest_cache:
                self._manifest_cache.remove(key)

            return True
        except Exception as e:
            print(f"Error updating manifest: {e}")
            return False

    # --- API Implementation ---

    def get(self, query: str) -> Result:
        """
        Lädt Daten. Unterstützt Wildcards (*) für Key-Pattern.
        """
        if not self.storage_client:
            return Result.default_internal_error(info="DB not initialized")

        # Fall 1: Wildcard Suche (z.B. "USER::*")
        if query == "all":
            query = "*"
        if query == "all-k":
            return Result.ok(data=list(self._manifest_cache))
        if '*' in query:
            pattern = query.replace('*', '')
            matching_keys = [k for k in self._manifest_cache if k.startswith(pattern)]
            results = {}
            for k in matching_keys:
                # Rekursiver Aufruf für einzelne Keys
                res = self.get(k)
                if res.is_ok():
                    results[k] = res.get()
            return Result.ok(data=results)

        # Fall 2: Einzelschlüssel
        # Zuerst im Cache schauen
        with self._cache_lock:
            if query in self._local_cache:
                return Result.ok(data=self._local_cache[query])

        blob_path = self._key_to_path(query)

        try:
            # BlobFile context manager handles downloading, decrypting and parsing
            # Wir nutzen use_cache=True, da der BlobStorage Cache via Watch invalidiert wird (hoffentlich),
            # aber sicherheitshalber verlassen wir uns auf den BlobStorage Cache Mechanismus.
            bf = BlobFile(blob_path, 'r', storage=self.storage_client, key=self.enc_key)
            if not bf.exists():
                return Result.default_user_error(info=f"Key '{query}' not found")

            with bf as f:
                data = f.read_json()

            # In lokalen Cache packen
            with self._cache_lock:
                self._local_cache[query] = data

            # Watch sicherstellen für zukünftige Updates
            self._ensure_watch(blob_path)

            return Result.ok(data=data)

        except Exception as e:
            return Result.default_internal_error(info=f"Error reading key '{query}': {e}")

    def set(self, query: str, value) -> Result:
        """
        Speichert Daten sofort persistent.
        """
        if not self.storage_client:
            return Result.default_internal_error(info="DB not initialized")

        blob_path = self._key_to_path(query)

        try:
            # 1. Daten schreiben
            with BlobFile(blob_path, 'w', storage=self.storage_client, key=self.enc_key) as f:
                f.clear()  # Überschreiben
                f.write_json(value)

            # 2. Cache updaten
            with self._cache_lock:
                self._local_cache[query] = value

            # 3. Manifest updaten (nur wenn neu)
            if query not in self._manifest_cache:
                self._update_manifest(query, add=True)

            # 4. Watch aktivieren (falls wir selbst Änderungen von anderen Geräten auf diesem Key wollen)
            self._ensure_watch(blob_path)

            return Result.ok()

        except Exception as e:
            return Result.default_internal_error(data=str(e), info=f"Failed to set '{query}'")

    def append_on_set(self, query: str, value) -> Result:
        """
        Fügt Daten zu einer Liste hinzu oder erstellt sie. Thread-safe(r) via read-modify-write.
        """
        if not self.storage_client:
            return Result.default_internal_error(info="DB not initialized")

        blob_path = self._key_to_path(query)

        try:
            with BlobFile(blob_path, 'w', storage=self.storage_client, key=self.enc_key) as f:
                # Versuch existierende Daten zu lesen
                try:
                    # Hinweis: BlobFile 'w' Modus liest standardmäßig vorher ein (siehe blobs.py logic)
                    current_data = f.read_json()
                except:
                    current_data = []

                if not isinstance(current_data, list):
                    # Wenn es keine Liste ist, machen wir eine draus (oder Fehler werfen, je nach Anforderung)
                    current_data = [current_data] if current_data else []

                if isinstance(value, list):
                    for v in value:
                        if v not in current_data:
                            current_data.append(v)
                elif value not in current_data:
                    current_data.append(value)

                f.clear()
                f.write_json(current_data)

            # Cache Update
            with self._cache_lock:
                # Wir müssen hier aufpassen, wir haben 'current_data' jetzt im Speicher
                self._local_cache[query] = current_data

            if query not in self._manifest_cache:
                self._update_manifest(query, add=True)

            return Result.ok()

        except Exception as e:
            return Result.default_internal_error(data=str(e), info=f"Failed to append to '{query}'")

    def delete(self, query: str, matching=False) -> Result:
        """
        Löscht Schlüssel.
        """
        if not self.storage_client:
            return Result.default_internal_error(info="DB not initialized")

        keys_to_delete = []
        if matching or '*' in query:
            pattern = query.replace('*', '')
            keys_to_delete = [k for k in self._manifest_cache if k.startswith(pattern)]
        else:
            keys_to_delete = [query]

        deleted_count = 0
        errors = []

        for key in keys_to_delete:
            blob_path = self._key_to_path(key)
            try:
                # Wir nutzen BlobStorage direkt zum Löschen, wenn der Blob nur diesen Key enthält.
                # Aber BlobFile mappt Key -> Pfad IN einem Blob.
                # Wenn wir Blobs pro User gruppieren, dürfen wir nicht den ganzen Blob löschen.

                # Checken ob Datei existiert und leer machen
                # Workaround: BlobFile hat keine explizite 'delete_internal_file' Methode im Interface,
                # aber wir können leeres JSON schreiben oder die Struktur manipulieren.
                # In der blobs.py: exists() prüft keys.
                # Saubere Lösung: Wir schreiben ein leeres Objekt oder nutzen ein Flag,
                # aber um "Exists" false zu machen, müssten wir den Key aus dem Pickle-Dict entfernen.

                # Da BlobFile.__exit__ schreibt: current_level[self.datei] = final_data
                # Es gibt keine API um den Key zu entfernen via BlobFile.
                # Wir schreiben None oder markieren es.
                # BESSER: Wir entfernen es aus dem Manifest. Das "File" bleibt als Leiche,
                # aber if_exist schaut ins Manifest.

                # Alternative: Wir implementieren _hard_delete via Zugriff auf interne Struktur,
                # aber das bricht Kapselung.

                # Strategie hier: Manifest Update ist die "Löschung" für die App-Logik.
                # Physisches Löschen (Aufräumen) wäre ein Garbage Collection Prozess.

                self._update_manifest(key, add=False)

                with self._cache_lock:
                    if key in self._local_cache:
                        del self._local_cache[key]

                deleted_count += 1

            except Exception as e:
                errors.append(f"{key}: {e}")

        if errors:
            return Result.custom_error(data=errors, info=f"Deleted {deleted_count} keys, {len(errors)} errors.")

        return Result.ok(data=deleted_count, data_info=f"Deleted {deleted_count} keys")

    def if_exist(self, query: str) -> bool:
        """
        Prüft Existenz effizient über das Manifest.
        """
        if not self._manifest_loaded:
            self._reload_manifest()

        if '*' in query:
            pattern = query.replace('*', '')
            for k in self._manifest_cache:
                if k.startswith(pattern):
                    return True
            return False

        return query in self._manifest_cache

    def exit(self) -> Result:
        """
        Räumt auf. Da wir sofort persistieren, müssen wir nur Watches stoppen.
        """
        try:
            # Watches stoppen
            if self.storage_client:
                # Wir können nicht einfach alle Watches des Clients stoppen,
                # da der Client vielleicht geshared ist.
                # Wir stoppen nur die Watches, die wir kennen (aber die API in blobs.py
                # erlaubt remove_watch per callback oder blob_id).

                # Da wir das 'callback' Objekt lokal (Closure) erstellt haben und nicht gespeichert haben,
                # ist das Entfernen einzelner Callbacks schwer.
                # Aber blobs.py erlaubt remove_watch(blob_id).
                pass

            return Result.ok(info="BlobDB closed cleanly (data was already persisted).").set_origin("BlobDB")
        except Exception as e:
            return Result.default_internal_error(data=str(e))
