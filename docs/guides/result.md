# Sub-Agent Result

**Task:** Erstelle 4 Markdown-Dateien im Ordner /sub/agent_guides/:

**SHARED_CONTEXT:**
Du arbeitest im ToolBoxV2 Doc-Writer System. Deine einzige Wahrheitsquelle ist der Code. Nie etwas erfinden oder aus alten Docs übernehmen ohne Code-Verifikation.

CODE-REFERENZ REGEL: Jeden Claim mit belegen. Format: <!-- verified: <dateiname>::<klasse_oder_funktion> -->

**AUFGABE 1: howto_consumer.md**
Lies:
- /tb/toolboxv2/__main__.py (Profil consumer)
- /tb/toolboxv2/utils/clis/cli_registry.py

Tasks (task-format):
1. Start your App
2. Check for Feature Updates
3. Install a Mod from Registry

**AUFGABE 2: howto_features.md**
Lies:
- /tb/toolboxv2/feature_loader.py (unpack_feature, ensure_features_loaded)
- /tb/toolboxv2/utils/clis/manifest_cli.py

Tasks:
1. List installed Features
2. Install a Feature Pack
3. Check required vs available Features

**AUFGABE 3: howto_server.md**
Lies:
- /tb/toolboxv2/__main__.py (_run_server_overview)
- /tb/toolboxv2/utils/manifest/converter.py (apply_all)

Tasks:
1. Apply Manifest to generate configs
2. Start Services
3. View Node/Service Status
4. Switch Database Mode

**AUFGABE 4: howto_developer.md**
Lies:
- /tb/toolboxv2/__main__.py (developer profile)
- /tb/toolboxv2/utils/clis/manifest_cli.py (pack command)
- /tb/toolboxv2/utils/clis/cli_registry.py

Tasks:
1. Set Developer Profile
2. Pack a Feature
3. Upload to Registry
4. Run Dev Dashboard

Jede Datei mit <!-- verified: dateiname::funktion --> Tags.

Schreibe die Ausgabe als:
---FILE: howto_consumer.md---
[Inhalt]
---END---
---FILE: howto_features.md---
[Inhalt]
---END---
---FILE: howto_server.md---
[Inhalt]
---END---
---FILE: howto_developer.md---
[Inhalt]
---END---

**Status:** max_iterations

**Result:**
<think>
Der User möchte, dass ich 4 How-To Markdown-Dateien erstelle. Ich muss:

1. Erst die relevanten Dateien lesen
2. Dann die 4 Dateien erstellen mit dem geforderten Format

Ich brauche die VFS-Tools. Lass mich die laden.
</think>


<think>
Jetzt lese ich alle relevanten Dateien parallel.
</think>


<think>
Ich habe jetzt die wichtigsten Code-Referenzen. Lass mich noch die Schema-Datei für ProfileType und DatabaseMode lesen.
</think>


<think>
Ich habe jetzt genug Informationen. Lass mich die 4 How-To Dateien erstellen.
</think>

Jetzt erstelle ich die 4 How-To Dateien:
<think>
Jetzt erstelle ich die 4 Dateien.
</think>


<think>
Jetzt die zweite Datei.
</think>


<think>
Jetzt die dritte Datei - Server How-To.
</think>


<think>
Jetzt die vierte und letzte Datei - Developer How-To.
</think>


<think>
Alle 4 Dateien sind erstellt. Jetzt soll ich noch prüfen, ob /tb/docs/new/ existiert und die Dateien dorthin kopieren.
</think>

Jetzt erstelle ich den `docs/new/` Ordner und kopiere die Dateien:
<think>
Der Ordner existiert bereits. Lass mich den cp Befehl nochmal单独 ausführen.
</think>



