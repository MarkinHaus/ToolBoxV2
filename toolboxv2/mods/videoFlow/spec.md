Absolut. Basierend auf Ihren detaillierten Anforderungen und den neuen Vorgaben (Implementierung als `toolboxv2`-Modul, strenge Entwicklungsregeln) habe ich den ursprünglichen Plan grundlegend überarbeitet.

Dieser neue Plan ist eine präzise Roadmap, die auf eine professionelle, wartbare und skalierbare Produktentwicklung ausgelegt ist und die Kollaboration zwischen Ihnen und einer KI optimiert.

---

### **Projekt: "Video Flow" – Vom CLI zum fertigen Produkt**

**Vision:** Eine hochmoderne, modulare Web-Plattform innerhalb des `toolboxv2`-Ökosystems, die es Benutzern ermöglicht, über ein Credit-System mühelos Videos zu erstellen. Die Plattform bietet eine nahtlose Erfahrung von der vollautomatischen Generierung bis hin zur detaillierten manuellen Bearbeitung jedes einzelnen Schrittes.

---

### **Phase 0: Regeln & Entwicklungsprinzipien**

Bevor wir beginnen, etablieren wir die Grundregeln, die für das gesamte Projekt gelten. Dies stellt Konsistenz und höchste Qualität sicher.

1.  **Maximale Dateigröße (1000-Zeilen-Regel):** Keine einzelne Code-Datei darf mehr als 1000 Zeilen umfassen. Dies erzwingt Modularität und Lesbarkeit.
2.  **Spezifikations- & Testgetriebene Entwicklung (Spec/Test-Driven Development):**
    *   **Docs-First:** Bevor eine Funktion implementiert wird, wird ihre Spezifikation in einer Markdown-Datei (`.md`) definiert. Diese beschreibt den Zweck, die Parameter, den Rückgabewert und das Verhalten.
    *   **Tests-First:** Nach der Spezifikation wird ein Testfall (z.B. mit `pytest`) geschrieben, der die in der `.md`-Datei definierte Funktionalität überprüft. Der Test wird fehlschlagen.
    *   **Implementierung:** Erst dann wird der Code geschrieben, um den Test erfolgreich zu machen.
3.  **Strikte Typensicherheit:** Das gesamte Projekt, sowohl Python-Backend als auch TypeScript/JSDoc-Frontend, muss vollständig typensicher sein. `Mypy` wird für das Backend verwendet.
4.  **Perfekte Projektstruktur:** Das Projekt wird in eine logische und tief verschachtelte Ordnerstruktur unterteilt, die klar zwischen Engine, API, Web-Interface und Dokumentation trennt.
5.  **Umfassende Dokumentation:** Jede Funktion, jede Klasse und jeder API-Endpunkt erhält eine eigene `.md`-Datei im `docs/`-Verzeichnis, die die Spezifikation enthält.
6.  **Einfache & Direkte Lösungen:** Komplexe Abstraktionen werden vermieden. Wo immer möglich, werden die von `toolboxv2` und `tbjs` bereitgestellten Werkzeuge genutzt, um einfache und wartbare Lösungen zu schaffen.
7.  **Produktionsfertiger Code:** Es wird kein Prototypen-Code geschrieben. Jede Implementierung muss robust, fehlerbehandelt und für den Produktionseinsatz bereit sein.

---

### **Phase 1: Fundament, Spezifikation & Architektur (Der Docs-First-Ansatz)**

**Ziel:** Ein felsenfestes Fundament durch sorgfältige Planung und Dokumentation schaffen, bevor die erste Zeile produktiven Codes geschrieben wird.

#### **Schritt 1.1: Projekt-Scaffolding als `toolboxv2`-Modul**

*   **Anleitung für den Menschen:** Erstellen Sie die grundlegende Ordnerstruktur für Ihr neues `toolboxv2`-Modul. Dies ist das Skelett Ihres gesamten Projekts.

*   **Prompt für die KI:**
    ```
    Erstelle die Verzeichnisstruktur für ein neues, komplexes `toolboxv2`-Modul namens "videoFlow". Die Struktur muss die "Perfekte Projektstruktur"-Regel befolgen und klar zwischen den verschiedenen Anliegen trennen.

    **Verzeichnisstruktur:**
    /mods/videoFlow/
    ├── __init__.py         # Haupt-Moduldatei, lädt die Tools
    ├── tools.py            # MainTool-Klasse, die die Sub-Module initialisiert
    │
    ├── engine/             # Kernlogik für die Videoerstellung
    │   ├── __init__.py
    │   ├── models/         # Pydantic-Datenmodelle (Story, Scene, etc.)
    │   ├── generators/     # Die einzelnen Generatoren (Image, Audio, etc.)
    │   ├── pipeline/       # Steuerung der schrittweisen Pipeline
    │   └── project_manager.py # Verwaltung von Projektzuständen
    │
    ├── api/                # API-Funktionen, die mit @export(api=True) dekoriert sind
    │   ├── __init__.py
    │   ├── auth.py         # Benutzer-Authentifizierung
    │   ├── credits.py      # Credit-System
    │   ├── projects.py     # Projekt-Management Endpunkte
    │   └── generation.py   # Endpunkte zum Auslösen der Generierung
    │
    ├── web/                # Statische Frontend-Dateien (HTML, CSS, JS)
    │   ├── assets/
    │   ├── css/
    │   ├── js/             # Hier wird die tbjs-Logik platziert
    │   └── index.html      # Haupt-Startseite
    │
    ├── docs/               # Markdown-Dokumentation (Specs)
    │   ├── engine/
    │   ├── api/
    │   └── models/
    │
    └── tests/              # Pytest-Tests
        ├── engine/
        ├── api/
        └── test_pipeline.py
    ```

#### **Schritt 1.2: Spezifikation der Datenmodelle und der Engine (Docs-First)**

*   **Anleitung für den Menschen:** Definieren Sie die Struktur Ihrer Daten und die Signaturen Ihrer Kernfunktionen in Markdown. Dies ist der Bauplan für die Engine.

*   **Prompt für die KI:**
    ```
    Erstelle die Spezifikations-Dokumente (.md Dateien) für die Datenmodelle und die Kernfunktionen der `videoFlow`-Engine. Platziere sie in den entsprechenden `docs/`-Unterordnern.

    **1. Datenmodelle (in `docs/models/`):**
    - `StoryData.md`: Beschreibe alle Felder des Pydantic-Modells `StoryData`.
    - `Scene.md`: Beschreibe alle Felder des `Scene`-Modells.
    - `Character.md`: Beschreibe alle Felder des `Character`-Modells.
    - `Timeline.md`: Beschreibe die Struktur eines neuen JSON-Objekts, das die interaktive Timeline mit Clips, Audio-Spuren und Effekten repräsentiert.

    **2. Engine-Funktionen (in `docs/engine/`):**
    - `project_manager.md`: Definiere die Funktionen `create_project`, `get_project_state`, `update_project_data`, `save_asset`.
    - `pipeline.md`: Definiere die Funktionen für jeden Schritt der Pipeline, z.B. `run_story_generation_step`, `run_image_generation_step`. Beschreibe, wie jeder Schritt den Projektstatus ändert.
    ```

#### **Schritt 1.3: API-Spezifikation (Docs-First)**

*   **Anleitung für den Menschen:** Definieren Sie Ihre API-Endpunkte in Markdown. Dies ist der Vertrag zwischen Ihrem Frontend und Backend.

*   **Prompt für die KI:**
    ```
    Basierend auf den Engine-Spezifikationen, erstelle die Markdown-Dokumente für die API-Endpunkte in `docs/api/`. Jeder Endpunkt sollte den HTTP-Methodentyp, die URL, erwartete Eingaben (Parameter, Body) und die möglichen Antworten (Erfolg, Fehler) beschreiben.

    **Beispiele für Endpunkte:**
    - `POST /api/videoFlow/register`: Benutzerregistrierung.
    - `POST /api/videoFlow/login`: Benutzer-Login.
    - `POST /api/videoFlow/create_project`: Erstellt ein neues leeres Projekt.
    - `PUT /api/videoFlow/update_story/{project_id}`: Speichert die Story-Daten.
    - `POST /api/videoFlow/run_step/{project_id}/{step_name}`: Löst einen Generierungsschritt aus.
    - `GET /api/videoFlow/project_status/{project_id}`: Fragt den Status eines Projekts ab.
    ```

---

### **Phase 2: Implementierung der Core-Engine (Testgetrieben)**

**Ziel:** Die dokumentierte Engine mit robustem, getestetem und modularem Code zum Leben erwecken.

#### **Schritt 2.1: Implementierung der Datenmodelle und Tests**

*   **Anleitung für den Menschen:** Setzen Sie die in Phase 1 definierten Pydantic-Modelle um und schreiben Sie Tests, die deren Validierung prüfen.

*   **Prompt für die KI:**
    ```
    Erstelle basierend auf den `.md`-Spezifikationen in `docs/models/` die Python-Dateien mit den Pydantic-Modellen in `mods/videoFlow/engine/models/`. Implementiere alle Modelle mit strikter Typisierung.

    Schreibe anschließend `pytest`-Tests in `tests/engine/test_models.py`, die sicherstellen, dass die Modelle korrekte Daten validieren und bei inkorrekten Daten wie erwartet `ValidationError` auslösen.
    ```

#### **Schritt 2.2: Refactoring des bestehenden Codes & Implementierung der Generatoren**

*   **Anleitung für den Menschen:** Zerlegen Sie Ihr ursprüngliches CLI-Skript in die neuen, modularen Generator-Dateien. Halten Sie sich strikt an die 1000-Zeilen-Regel.

*   **Prompt für die KI:**
    ```
    Analysiere das bereitgestellte "Multimedia Story Generator v5.0"-Skript. Refaktoriere die Klassen `StoryGenerator`, `ImageGenerator`, `AudioGenerator`, `VideoGenerator` und `ClipGenerator` in separate Dateien innerhalb von `mods/videoFlow/engine/generators/`.

    **Anweisungen:**
    - Jede Datei darf maximal 1000 Zeilen haben. Teile größere Klassen bei Bedarf in kleinere Hilfsklassen oder Funktionen auf.
    - Passe alle Importe an die neue modulare Struktur an.
    - Entferne jegliche direkte Ausführungslogik (wie die `run()`-Funktion). Die Klassen sollten nur die reine Generierungslogik enthalten.
    - Schreibe für jede Kernfunktion eines Generators (z.B. `ImageGenerator.generate_all_images`) einen grundlegenden `pytest`-Test in `tests/engine/generators/`, der prüft, ob die Funktion ohne Fehler durchläuft (Mocking von externen API-Aufrufen ist hierbei erforderlich).
    ```

#### **Schritt 2.3: Implementierung der schrittweisen Pipeline und des Project Managers**

*   **Anleitung für den Menschen:** Dies ist das Herzstück. Implementieren Sie die Logik, die den Zustand eines Projekts verwaltet und die Generierungsschritte einzeln ausführen kann.

*   **Prompt für die KI:**
    ```
    Implementiere `ProjectManager` und die Pipeline-Logik gemäß den Spezifikationen.

    **1. In `mods/videoFlow/engine/project_manager.py`:**
    - Implementiere die in `docs/engine/project_manager.md` definierten Funktionen.
    - Nutze die `toolboxv2.FileHandler` oder eine einfache JSON-Speicherung, um den Zustand von Projekten auf der Festplatte zu verwalten.

    **2. In `mods/videoFlow/engine/pipeline/`:**
    - Erstelle eine Datei `steps.py`, die die einzelnen Generierungs-Tasks implementiert (z.B. `run_story_generation_step`).
    - Jede dieser Funktionen nimmt eine `project_id` entgegen, lädt den Projektzustand mit dem `ProjectManager`, ruft den entsprechenden Generator auf, und speichert das Ergebnis (Dateipfade, Metadaten) wieder mit dem `ProjectManager`.
    - Nutze `app.run_bg_task` aus `toolboxv2`, um die Generatoren in einem Hintergrund-Thread auszuführen, damit die API nicht blockiert wird.
    - Schreibe Tests in `tests/test_pipeline.py`, die den gesamten Ablauf eines Projekts (Erstellen -> Story -> Bilder -> ... -> Fertig) simulieren und den korrekten Zustandsübergang nach jedem Schritt überprüfen.
    ```

---

### **Phase 3: API-Schicht & Backend-Integration**

**Ziel:** Eine sichere und funktionale API schaffen, die als Schnittstelle zwischen Frontend und Engine dient.

#### **Schritt 3.1: API-Endpunkte mit `toolboxv2` implementieren**

*   **Anleitung für den Menschen:** Setzen Sie die API-Endpunkte um, indem Sie die Engine-Funktionen mit dem `@export(api=True)`-Decorator von `toolboxv2` versehen.

*   **Prompt für die KI:**
    ```
    Implementiere die API-Endpunkte gemäß den Spezifikationen in `docs/api/` in den entsprechenden Dateien unter `mods/videoFlow/api/`.

    **Anweisungen:**
    - Nutze den `@export(api=True, mod_name="videoFlow", ...)`-Decorator von `toolboxv2`.
    - `auth.py`: Implementiere `register` und `login`. Nutze `toolboxv2.Code` oder eine ähnliche Methode zur Passwort-Verschlüsselung. Die Login-Funktion sollte ein JWT zurückgeben, das vom `toolboxv2`-Session-Management verwaltet wird.
    - `projects.py`: Implementiere `create_project`, `get_project_status`, etc. Diese Funktionen müssen auf die `session` zugreifen, um die `user_id` zu erhalten und die Berechtigung zu prüfen.
    - `generation.py`: Implementiere `run_step`. Diese Funktion sollte das Credit-System aufrufen, um Credits abzubuchen, bevor sie die Pipeline-Task im Hintergrund startet.
    - Schreibe API-Tests in `tests/api/`, die jeden Endpunkt aufrufen und die erwartete Antwort (sowohl bei Erfolg als auch bei Fehlern wie fehlender Authentifizierung) überprüfen.
    ```

---

### **Phase 4: Frontend-Entwicklung mit `tbjs`**

**Ziel:** Eine intuitive, reaktionsschnelle und visuell ansprechende Benutzeroberfläche schaffen.

#### **Schritt 4.1: Aufbau des Generator-Studios und der Komponenten**

*   **Anleitung für den Menschen:** Entwickeln Sie die grundlegende Struktur Ihrer Web-Anwendung und die Hauptkomponente für die Videoerstellung.

*   **Prompt für die KI:**
    ```
    Erstelle die grundlegende JavaScript-Logik für das "Generator-Studio" in `mods/videoFlow/web/js/studio.js` unter Verwendung von `tbjs`.

    **Features:**
    1.  **Initialisierung:** Nutze `TB.init`, um die App zu konfigurieren (API-URL, initialer State).
    2.  **State Management:** Verwalte den Zustand des aktuellen Projekts (Story-Daten, generierte Assets, etc.) in `TB.state`.
    3.  **Kommunikation:** Erstelle Funktionen, die `TB.api.request` verwenden, um mit den `videoFlow`-API-Endpunkten zu kommunizieren.
    4.  **UI-Updates:** Nutze `TB.events`, um auf Zustandsänderungen zu reagieren und die UI zu aktualisieren. Erstelle eine Funktion, die den Projektstatus regelmäßig vom Backend abfragt und einen Ladebalken anzeigt.
    5.  **Komponenten:** Implementiere die UI für die Eingabe der Story- und Charakterdetails. Nutze `TB.ui.Toast` für Benachrichtigungen und `TB.ui.Loader` für Ladeanzeigen.
    ```

#### **Schritt 4.2: Entwicklung des interaktiven Timeline-Editors**

*   **Anleitung für den Menschen:** Dies ist die komplexeste UI-Komponente. Implementieren Sie die visuelle Timeline, die Drag-and-Drop, Bearbeitung und das Hinzufügen/Entfernen von Clips ermöglicht.

*   **Prompt für die KI:**
    ```
    Entwirf und implementiere den interaktiven Timeline-Editor in `mods/videoFlow/web/js/timeline.js`.

    **Funktionen:**
    1.  **Rendering:** Schreibe eine Funktion, die das `timeline`-Objekt aus `TB.state` entgegennimmt und es als visuelle Blöcke (Clips, Audio) in HTML rendert.
    2.  **Interaktivität:**
        - Implementiere Drag-and-Drop-Funktionalität (z.B. mit der `Drag and Drop API` des Browsers), um Clips neu anzuordnen.
        - Bei einem Klick auf einen Clip soll ein `TB.ui.Modal` geöffnet werden, das die Bearbeitung der Clip-Details ermöglicht.
        - Das Speichern im Modal ruft den API-Endpunkt zum Aktualisieren des Assets und zum erneuten Auslösen der Generierung für diesen Clip auf.
    3.  **Zustandssynchronisation:** Jede Änderung an der Timeline (Neuanordnen, Bearbeiten, Löschen) muss den Zustand in `TB.state` aktualisieren und an das Backend gesendet werden, um dort gespeichert zu werden.
    ```

---

### **Phase 5: Finale Integration, Tests & Deployment**

**Ziel:** Das fertige Produkt bereitstellen und seine Stabilität und Performance sicherstellen.

#### **Schritt 5.1: End-to-End-Tests**

*   **Anleitung für den Menschen:** Testen Sie den gesamten Anwendungsfluss aus der Perspektive eines Benutzers.

*   **Prompt für die KI:**
    ```
    Erstelle einen detaillierten End-to-End-Testplan in einer Datei `docs/E2E_Test_Plan.md`.

    **Szenarien:**
    1.  **"Happy Path":** Registrierung -> Login -> Neues Projekt -> Vollautomatische Generierung -> Download des Videos.
    2.  **"Editor Path":** Login -> Projekt erstellen -> Story generieren -> Pause -> Charakterbild ändern -> Bilder neu generieren -> Fortsetzen -> Video herunterladen.
    3.  **Fehlerfälle:** Versuch, ohne Credits zu generieren; Versuch, auf ein fremdes Projekt zuzugreifen; Eingabe ungültiger Daten.
    ```

#### **Schritt 5.2: Deployment-Strategie**

*   **Anleitung für den Menschen:** Planen und dokumentieren Sie den Prozess, um Ihre `toolboxv2`-Anwendung auf einem Server bereitzustellen.

*   **Prompt für die KI:**
    ```
    Erstelle eine Deployment-Anleitung in `docs/Deployment.md` für die `videoFlow`-Anwendung.

    **Anweisungen:**
    - Beschreibe, wie man die `toolboxv2`-Anwendung als Systemdienst (z.B. mit `systemd` auf Linux) unter Verwendung der `--sm`-Option des `tb`-CLI einrichtet.
    - Gib eine Beispielkonfiguration für einen Reverse-Proxy (z.B. Nginx), der Anfragen an den `actix_web`-Server von `toolboxv2` weiterleitet und sich um HTTPS/SSL kümmert.
    - Erkläre, welche Umgebungsvariablen für die Produktion gesetzt werden müssen (Datenbank-URL, API-Schlüssel für externe Dienste, etc.).
    ```
