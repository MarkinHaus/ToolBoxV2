### **Projektregeln & Entwicklungsprinzipien für "Video Flow"**

Diese Grundregeln gelten für das gesamte Projekt, um Konsistenz, Modularität und höchste Code-Qualität sicherzustellen.

1.  **Maximale Dateigröße (1000-Zeilen-Regel)**
    *   Keine einzelne Code-Datei darf mehr als 1000 Zeilen umfassen. Dies erzwingt Modularität, fördert die Lesbarkeit und zwingt zur klaren Trennung von Verantwortlichkeiten.

2.  **Spezifikations- & Testgetriebene Entwicklung (Spec/Test-Driven Development)**
    *   **Docs-First:** Bevor eine Funktion oder ein Modul implementiert wird, muss dessen Spezifikation in einer Markdown-Datei (`.md`) definiert werden. Diese Spezifikation beschreibt den Zweck, die Parameter, den Rückgabewert und das erwartete Verhalten.
    *   **Tests-First:** Nach der Erstellung der Spezifikation wird ein Testfall (z.B. mit `pytest`) geschrieben, der die definierte Funktionalität validiert. Dieser Test muss zunächst fehlschlagen.
    *   **Implementierung:** Erst nachdem der fehlschlagende Test existiert, wird der eigentliche Code geschrieben, mit dem Ziel, den Test erfolgreich zu machen.

3.  **Strikte Typensicherheit**
    *   Das gesamte Projekt muss vollständig typensicher sein.
    *   Für das Python-Backend wird `Mypy` zur statischen Typüberprüfung eingesetzt.
    *   Für das Frontend wird TypeScript oder JavaScript mit JSDoc zur Gewährleistung der Typensicherheit verwendet.

4.  **Perfekte Projektstruktur**
    *   Das Projekt muss in eine logische und tief verschachtelte Ordnerstruktur unterteilt werden.
    *   Es muss eine klare Trennung zwischen den Hauptkomponenten geben: **Engine** (Kernlogik), **API** (Schnittstelle), **Web-Interface** (Benutzeroberfläche) und **Dokumentation**.

5.  **Umfassende Dokumentation**
    *   Jede Funktion, jede Klasse und jeder API-Endpunkt muss eine eigene begleitende Markdown-Datei (`.md`) im `docs/`-Verzeichnis haben.
    *   Diese Datei dient als "Single Source of Truth" und enthält die Spezifikation der jeweiligen Code-Einheit.

6.  **Einfache & Direkte Lösungen**
    *   Komplexe und unnötige Abstraktionen sind zu vermeiden.
    *   Es sollen bevorzugt die von den Frameworks (`toolboxv2`, `tbjs`) bereitgestellten Werkzeuge und Muster genutzt werden, um einfache, direkte und wartbare Lösungen zu schaffen.

7.  **Produktionsfertiger Code**
    *   Es wird kein temporärer oder Prototypen-Code geschrieben.
    *   Jede Implementierung muss von Anfang an robust, mit angemessener Fehlerbehandlung versehen und für den Einsatz in einer Produktionsumgebung ausgelegt sein.
