# Workshop Page — Creation Steps

> Workflow für die Erstellung von interaktiven Workshop-Seiten.
> Dieses Dokument beschreibt den kompletten Ablauf von Roh-Script bis fertige HTML-Seite.

---

## Überblick

```
Roh-Script ──→ Step 1: Blaupause ──→ User Review ──→ Step 2: HTML-Generierung ──→ Fertig
   (Text/MD)      (blaupause.md)      (Überarbeitung)    (seite.html)
```

Es gibt **immer** zwei Schritte. Nie direkt von Script zu HTML.

---

## Voraussetzungen

Bevor du startest, brauchst du:

1. **Das Roh-Script** — der Kursinhalt als Text/Markdown. Enthält typischerweise:
   - Aufgabenstellungen mit "Aufgabe:" Markierungen
   - Beispiellösungen / Code-Blöcke
   - Tipps und Hinweise
   - Zeitangaben
   - Lernziele oder Konzepte

2. **Den workshop-page Skill** — installiert oder als SKILL.md verfügbar

3. **Optional: den Paper Style Guide** (`nbpaper_style.md`) — für die vollen Design-Tokens.
   Ohne ihn werden die im Skill eingebetteten Fallback-Tokens verwendet.

---

## Step 1: Blaupause erstellen

### Input an Claude
```
Hier ist das Script für eine Coding-Session: [Script einfügen oder als Datei hochladen]
Erstelle eine Blaupause.
```

### Was Claude tut
1. Liest den Skill (`workshop-page/SKILL.md`)
2. Analysiert das Roh-Script:
   - Zählt die Aufgaben
   - Identifiziert Konzepte pro Aufgabe
   - Erkennt Schwierigkeitsgrad
   - Findet Bonus/Zusatzaufgaben
3. Generiert `blaupause.md` nach dem Template (`references/blueprint_template.md`)
4. Füllt alle Felder aus — inkl. Vorschläge für:
   - Analogie-Beispiele (Hint 2)
   - Platzhalter-Auswahl (Hint 3)
   - Easter Eggs
   - Halfway-Motivation
   - Aufteilungs-Empfehlung

### Output
Eine `.md`-Datei: die Blaupause. Claude präsentiert sie zum Download + zeigt die Aufgaben-Tabelle inline.

---

## User Review: Blaupause überarbeiten

### Was du als Kursleiter prüfst und anpasst

**Muss geprüft werden:**
- [ ] Stimmt die Aufgaben-Reihenfolge? Ist der Schwierigkeitsanstieg logisch?
- [ ] Sind die Kern-Konzepte pro Aufgabe korrekt identifiziert?
- [ ] Stimmen die Zeitschätzungen?
- [ ] Ist die Aufteilungs-Empfehlung (1 vs 2 Dateien) sinnvoll?

**Sollte angepasst werden:**
- [ ] **Erklärungs-Stil pro Aufgabe** — Passen die Analogien? Gibt es bessere aus deiner Erfahrung?
- [ ] **Erklärungs-Patterns** — DAS ist der wichtigste Abschnitt. Schreib ihn in deinem eigenen Stil um. Er wird als Lern-Pattern für zukünftige Generierungen verwendet.
- [ ] **Easter Eggs** — Passen sie zur Zielgruppe? Sind sie lustig genug?
- [ ] **Bonus/Zusatzaufgaben** — Zu leicht? Zu schwer? Richtig platziert?

**Kann angepasst werden:**
- [ ] Theme-Farben
- [ ] Hint-2 Analogie-Ideen (andere Beispiele die besser passen)
- [ ] Platzhalter-Auswahl für Hint 3 (welche Stellen im Code werden zu Lücken)
- [ ] Halfway-Motivation Text und Platzierung

### Dauer
15–30 min. Je mehr du hier investierst, desto besser wird die Seite.

---

## Step 2: HTML generieren

### Input an Claude
```
Hier ist die überarbeitete Blaupause: [blaupause.md hochladen]
Und hier das Original-Script: [script.md hochladen, falls nicht im Chatverlauf]
Generiere die HTML-Seite(n).
```

### Was Claude tut
1. Liest den Skill + die überarbeitete Blaupause
2. Liest den Paper Style Guide (oder Fallback-Tokens)
3. Generiert pro Datei (1 oder 2):
   - Volle HTML-Struktur nach Page Architecture (Skill)
   - Theme-Layer basierend auf Blueprint Meta
   - Alle Task-Sektionen mit 4-Hint-System
   - Hint-3 Lückentext mit collapsierbaren Lösungen (`<details>`)
   - Halfway-Motivation an der im Blueprint definierten Stelle
   - Easter Eggs an den im Blueprint definierten Stellen
   - Klickbare Progress-Dots im Nav
   - Celebration am Ende
4. Läuft die Quality Checklist durch
5. Präsentiert die Datei(en) zum Download

### Output
1–2 `.html`-Dateien, komplett self-contained.

---

## Nach der Erstellung

### Testen
- Seite im Browser öffnen
- Alle Hints durchklicken — stimmt der progressive Reveal?
- Alle Checkboxen abhaken — kommt die Celebration?
- Progress-Dots anklicken — scrollen sie korrekt?
- `<details>` in Hint 3 auf/zuklappen — funktioniert es?
- Easter Eggs finden (HTML-Kommentar am Ende der Datei listet sie)
- Auf dem Handy testen — responsive?

### Blaupause archivieren
Die überarbeitete `blaupause.md` aufbewahren. Sie enthält:
- Deine Erklärungs-Patterns (Institutional Knowledge)
- Die Aufgaben-Struktur (Wiederverwendbar für ähnliche Sessions)
- Easter Egg Dokumentation

Beim nächsten Mal kannst du mit "Nimm diese Blaupause als Basis für eine neue Session über [Thema]" starten.

---

## Schnell-Referenz: Dateien

```
workshop-page/
├── SKILL.md                          ← Der Skill (Regeln, Architektur, Tokens)
├── references/
│   ├── blueprint_template.md         ← Leeres Blaupause-Template
│   └── nbpaper_style.md             ← Verweis auf Paper Style Guide
│
Arbeitsverzeichnis (pro Session):
├── roh_script.md                     ← Input: Das Kurs-Script
├── blaupause.md                      ← Step 1 Output → User überarbeitet
├── blaupause_reviewed.md             ← Überarbeitete Version → Step 2 Input
└── session_seite.html                ← Step 2 Output: Fertige Seite
    (oder: teil1.html + teil2.html)
```

---

## Deployment in Kurse (statt Upload)

Die fertige `.html` wird auf der Coach-Seite eingesetzt, nicht als Datei hochgeladen:

1. Coach-Seite öffnen → Kurs wählen (oder anlegen).
2. **+ Neue Datei** → Namen vergeben (z.B. "Session 3 — Lotto").
3. Datei anklicken → HTML in das Textfeld einfügen → **Speichern**.
4. **Testen ✓** prüft Inhalt + ob das `reveal()`/`data-kurse`-Naming vorhanden ist.
5. **Vorschau** rendert die Seite mit eingehängtem `kurse.js` in einem neuen Tab.
6. Unter **Link erstellen**: von/bis Session + Einstieg (Anker) → Link teilen.

Tracking (welche Aufgabe, offene Tipps, Zeiten) läuft automatisch, sobald die
Seite das Standard-Naming-Modell dieses Skills nutzt — siehe
`SKILL.md → Kurse-Kompatibilität`.
