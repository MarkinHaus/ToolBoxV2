# workshop-page — Kursinhalte erstellen (Kurse-kompatibel)

> In Claude einfügen, dann: "Hier ist ein Session-Script: … erstelle eine Blaupause."

======================================================================
# SKILL.md
======================================================================
---
name: workshop-page
description: >
  Create interactive, single-file HTML workshop/tutorial pages for coding sessions aimed at teens (13–16).
  Transforms raw lesson scripts (Markdown/text with tasks, hints, and solutions) into polished, self-contained
  HTML pages using the TBJS Paper Neo-Brutalism design system. Use this skill whenever the user asks to build
  a "workshop page", "lesson page", "tutorial page", "session page", "Kursseite", "Aufgabenseite", or any
  interactive step-by-step coding guide for young learners. Also trigger when the user provides a lesson
  script with tasks/solutions and asks to make it into a website, or says "make an interactive page from this
  curriculum". The output is always a single .html file with everything inline (CSS + JS), no external
  dependencies except Google Fonts.
---

# Workshop Page Builder

Build interactive, single-file HTML tutorial pages that guide teens (13–16) through coding exercises.
The pages follow the **TBJS Paper Neo-Brutalism** design system and have a specific pedagogical structure.

## When to read the style reference

Before generating any HTML, check if `/mnt/user-data/uploads/nbpaper_style.md` exists and read it.
If it does not exist, use the **Inline Style Tokens** section below. The Paper style is non-negotiable —
every workshop page MUST use it.

---

## Page Architecture

Every workshop page has these structural layers, in order:

### 1. Sticky Nav Bar
- Left: brand/title in `--font-display`, weight 700, with optional themed icon (colored square with 2px ink border)
- Right: **clickable progress dots** (numbered 1–N), each an `<a href="#task-N">` that scrolls to the task
- Dots have three states: `.active` (current, themed accent color), `.done` (green/grass, shows ✓), default (white)
- Clicking a dot scrolls smoothly to that task section

### 2. Hero Section
- Eyebrow: uppercase mono, `--text-xs`, letter-spacing 2px, `--ink-muted`
- H1: `--font-display`, `--text-display`, weight 700. Can have a highlight span with themed accent background + ink border + offset shadow
- 1–2 paragraphs explaining what the session is about, max 68ch width
- **Game/Project Preview Box**: dark background card (`--mc-obsidian` or `#1a1a2e` or similar dark tone matching the theme), 2px ink border, 6px offset shadow. Lists the final result in numbered steps with accent-colored numbers. This gives learners the "big picture" before diving in.

### 3. Halfway Motivation Block
- Insert after approximately 50% of the tasks
- A full-width callout card with themed accent background (e.g. grass-green, diamond-cyan, warning-gold)
- Bold heading like "💪 Halbzeit!", "⚡ Schon über die Hälfte!", or similar
- 1–2 encouraging sentences acknowledging what they've built so far and teasing what's coming
- Use `--font-display` for the heading, keep it energetic but not cringe

### 4. Task Sections (the core)
Each task section contains, in order:

#### a) Step Banner
Black background, white text, mono uppercase with letter-spacing: "AUFGABE N VON M" or "SCHRITT N VON M"

#### b) Task Header
H2 in `--font-display`, weight 600, describing what this task accomplishes

#### c) "Das machst du jetzt" Box
Accent-colored background (gold/warning works well), 2px ink border, 4px offset shadow.
Contains a `task-now-label` (uppercase tiny mono "DAS MACHST DU JETZT") and a bold sentence
describing the concrete action. Uses `--font-display` weight 600.

#### d) Task Description
1–3 paragraphs of prose explaining the concepts, referencing what learners already know,
giving context. Max 68ch width. Can include inline `<code>` snippets.

#### e) The 4-Hint Progressive Reveal Grid
A 2-column grid (stacks to 1 col on mobile) with 4 hint cards. Only Hint 1 is visible initially.
Each subsequent hint is revealed by clicking the button on the previous card.

**Hint 1 — "WO & WAS"**
- Always visible (not hidden)
- Explains WHERE in the code to add things and WHAT concepts are needed
- No code yet — just orientation
- Button: "Weiter →" (reveals Hint 2)

**Hint 2 — "CODE-BEISPIEL"**
- Hidden initially, class `hidden-content`
- Shows an ANALOGOUS code example — similar pattern but different context (different variable names, different use case)
- This is NOT the solution — it's a transferable pattern the learner adapts
- Button: "Mehr Hilfe" (reveals Hint 3)

**Hint 3 — "LÜCKENTEXT" (Fill-the-blanks)**
- Hidden initially
- Shows the actual target code but with key parts replaced by `<span class="blank">#PLACEHOLDER#</span>` in gold/yellow
- Placeholders use descriptive ALL-CAPS names like `#FUNKTION#`, `#VARIABLE#`, `#BLOCK_ID#`
- **Collapsible solution hints**: Below the code block, a `<details><summary>` element contains the mapping of each placeholder to its answer. The summary text is something like "🔍 Platzhalter-Lösungen ein-/ausklappen". This way learners can try first, then peek if stuck.
- Button: "Lösung zeigen" (reveals Hint 4)

**Hint 4 — "LÖSUNG"**
- Hidden initially, has additional class `solution-card` (green left border, green hint-num)
- Shows the complete, correct code
- Includes a `where-tag` (black background, white text, uppercase mono) saying exactly where to insert: "→ Einfügen in: datei.py · nach der countdown() Funktion"
- May include a `callout` with an explanation or "So liest du das:" paragraph
- May include a **Bonus ⭐** suggestion (e.g. "Ändere die Zahl und schau was passiert")

#### f) Checkbox Row + Strukturübersicht (optional)
After the hint grid: a styled checkbox + label. Checking it marks the progress dot as done,
scrolls to the next task, and activates the next dot.

**Checkpoint Structure Overview:** At tasks where it makes sense — typically after a conceptual
milestone, after the first testable version, or before a major new topic — show a collapsible
`<details>` block BELOW the checkbox row that visualizes the code structure built so far vs.
the full target structure.

When to include it (use judgment, NOT on every task):
- After setup/boilerplate is complete (learner can verify: "do I have all the imports?")
- After the first runnable version (learner can compare their file structure)
- Before a section that adds to an existing function (learner needs to know WHERE they are)
- After the halfway point (good orientation checkpoint)

When NOT to include it:
- After trivial tasks (single-line additions)
- When the previous task already had one
- When the next task is a direct continuation with no structural change

Format: A `<details class="checkpoint">` with summary "📋 Strukturübersicht: Wo stehst du gerade?"
Inside: a simplified pseudo-code or outline showing the full target structure, with the parts
built so far in normal text and the parts still to come greyed out / marked with `# kommt noch...`.

```html
<details class="checkpoint">
  <summary>📋 Strukturübersicht: Wo stehst du gerade?</summary>
  <pre class="structure-map"><code>import time          ← ✅ fertig
import random        ← ✅ fertig

def countdown():     ← ✅ fertig
    ...

def zufaelliger_Block():   ← # kommt in Aufgabe 6
    ...

def parkourSpiel():  ← ✅ Grundgerüst steht
    ende = True      ← ✅ fertig
    # Teleport       ← ✅ fertig
    # Countdown      ← ✅ fertig
    while ende:      ← ✅ fertig
        # Block prüfen   ← # kommt in Aufgabe 5
        # Neuer Block    ← # kommt in Aufgabe 6
        # Punkte         ← # kommt in Aufgabe 8

parkourSpiel()</code></pre>
</details>
```

Style the checkpoint:
```css
.checkpoint { margin-top: 0.75rem; }
.checkpoint summary {
  font-family: var(--font-display);
  font-size: var(--text-sm);
  cursor: pointer;
  color: var(--ink-muted);
  padding: 0.5rem 0;
}
.checkpoint summary:hover { color: var(--ink); }
.structure-map {
  font-size: var(--text-sm);
  background: var(--paper-sunken);
  border: 2px solid var(--ink);
  box-shadow: 3px 3px 0 var(--ink);
  padding: 1rem;
  margin-top: 0.5rem;
  line-height: 1.6;
}
.structure-map .done { color: var(--ink); }
.structure-map .upcoming { color: var(--ink-faint); font-style: italic; }
```

### 5. Easter Eggs (minimum 3)
Hide at least 3 small surprises in the page. These are fun discoveries for curious learners who poke around. Examples:
- A hidden message that appears when you click a specific element 7 times
- A `<span>` with `title` attribute containing a joke or fun fact (shows on hover/long-press)
- A Konami code listener that triggers a fun CSS animation or chat message
- A hidden pixel art or emoji that only appears at a certain scroll position
- A secret CSS class that inverts colors when a specific key combo is pressed
- A counter that tracks how many hints were revealed, with a hidden achievement message if all are opened

Be creative. The easter eggs should feel rewarding to find, not annoying. They should not interfere with the learning flow. Document them in an HTML comment at the bottom of the file: `<!-- EASTER EGGS: 1. ... 2. ... 3. ... -->` so the session leader knows about them.

### 6. Final Celebration
Hidden by default, appears when all checkboxes are checked:
- Themed accent background (green/grass), white text, 2px ink border, 8px offset shadow
- Big heading with emoji: "⛏ PARKOUR KOMPLETT!" or "🐸 ALLE AUFGABEN GESCHAFFT!"
- Paragraph summarizing what was built
- "Ideen zum Weiterbauen" — 2–3 extension ideas

---

## Theming

The base is always TBJS Paper. On top of that, add a **theme layer** that matches the session's topic.
The theme adds 3–6 extra CSS custom properties and uses them for accent colors on specific elements.

**Examples of theme layers:**
- Minecraft: `--mc-diamond: #4AEDD9; --mc-grass: #5D8C3E; --mc-gold: #FCDB05; --mc-obsidian: #1B1028;`
- Space: `--space-nebula: #6B3FA0; --space-star: #FFD700; --space-void: #0a0a1a;`
- Frog game: `--frog-green: #4CAF50; --frog-fire: #FF5722; --frog-ruby: #E91E63;`
- Web dev: `--web-html: #E44D26; --web-css: #264DE4; --web-js: #F7DF1E;`

Use theme colors for: progress dot active state, task-now box background, game preview box background,
highlight spans in headings, bonus tags, celebration card, halfway motivation block.
Keep `--ink`, `--paper-bg`, `--paper-surface`, `--paper-sunken` from Paper — do NOT override structural tokens.

---

## Inline Style Tokens (Fallback)

If the full Paper style guide is not available, use these tokens:

```css
:root {
  --paper-bg: #f4f1ea;
  --paper-surface: #ffffff;
  --paper-sunken: #ebe7dc;
  --ink: #111111;
  --ink-muted: #555555;
  --ink-faint: #888888;
  --rule: #111111;

  --font-display: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
  --font-body: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;

  --text-display: clamp(28px, 5vw, 44px);
  --text-h1: clamp(24px, 3.5vw, 32px);
  --text-h2: clamp(20px, 2.5vw, 26px);
  --text-h3: clamp(17px, 2vw, 21px);
  --text-base: 16px;
  --text-sm: 14px;
  --text-xs: 12px;
}
```

**Paper rules (non-negotiable):**
- `border-radius: 0` everywhere (except `border-radius: 50%` for avatar dots)
- All borders: `2px solid var(--ink)` minimum
- All interactive surfaces: offset shadow `Npx Npx 0 var(--ink)` (4px buttons, 6px cards, 8px hover)
- Hover: `transform: translate(-2px, -2px)` + shadow grows. Active: `translate(2px, 2px)` + shadow zeroes
- No gradients, no backdrop-filter, no blur shadows, no border-radius
- Headlines: `--font-display` (mono). Body: `--font-body` (sans)
- Google Fonts import: IBM Plex Mono (400,500,600,700) + IBM Plex Sans (400,500,600)

---

## JavaScript Behavior

### Progressive Reveal
```javascript
function reveal(btn, taskNum, hintNum) {
  const target = document.getElementById('hint-' + taskNum + '-' + hintNum);
  if (target) {
    target.classList.add('revealed');
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
  btn.disabled = true;
  btn.style.opacity = '0.35';
  btn.style.cursor = 'default';
  btn.textContent = '✓ Aufgedeckt';
}
```

### Progress Tracking
```javascript
function markDone(n) {
  // Mark dot as done (green, ✓)
  // Activate next dot
  // Scroll to next task
  // Check if ALL done → show celebration
}
```

### Clickable Progress Dots
Each dot is an `<a>` (or has an `onclick`) that scrolls to `#task-N`. Make sure the scroll offset
accounts for the sticky nav height.

### Collapsible Placeholder Solutions (Hint 3)
Use native `<details><summary>` — no JS needed. Style the summary with `--font-display`, cursor pointer.
```html
<details class="placeholder-reveal">
  <summary>🔍 Platzhalter-Lösungen ein-/ausklappen</summary>
  <div class="placeholder-answers">
    <code>#FUNKTION#</code> = <code>spawnFeind</code><br>
    <code>#VARIABLE#</code> = <code>randi(50, 450)</code>
  </div>
</details>
```

Style the details/summary:
```css
.placeholder-reveal { margin-top: 0.75rem; }
.placeholder-reveal summary {
  font-family: var(--font-display);
  font-size: var(--text-sm);
  cursor: pointer;
  color: var(--ink-muted);
  padding: 0.4rem 0;
}
.placeholder-reveal summary:hover { color: var(--ink); }
.placeholder-answers {
  padding: 0.75rem 1rem;
  background: var(--paper-sunken);
  border: 1px solid var(--ink);
  margin-top: 0.5rem;
  line-height: 1.8;
}
```

---

## Creation Process — 2-Step Working Agreement

NEVER go straight from raw script to final HTML. Always do Step 1 first, wait for user approval,
then proceed to Step 2. This gives the user creative control over structure, pacing, and teaching style.

### Step 1: Blueprint (Blaupause)

Read the raw lesson script + the SKILL.md. Then generate a **Blueprint Markdown file** (`blaupause.md`).
This file is a lightweight plan — NOT the final page. Present it to the user for review and editing.

The blueprint follows this exact template (see `references/blueprint_template.md` for the full pattern):

```markdown
# Blaupause: [Session Title]

## Meta
- **Thema:** [was wird gebaut]
- **Sprache:** [Python / JavaScript / etc.]
- **Theme:** [Minecraft / Space / Frosch / etc.]
- **Altersgruppe:** [13–14 / 14–16]
- **Geschätzte Dauer:** [X min gesamt]
- **Dateien:** 1 HTML | oder 2 HTML (Teil 1 + Teil 2) — mit Begründung

## Aufgaben-Übersicht

| # | Titel | Konzept | Dauer | Schwierigkeit |
|---|-------|---------|-------|---------------|
| 0 | Setup | Verbindung + Editor starten | 3 min | ☆ |
| 1 | ... | ... | ... | ☆☆ |

## Pro Aufgabe — Detailplan

### Aufgabe 1: [Titel]
**Kern-Konzept:** [Was lernen sie? z.B. "Funktionen definieren"]
**Erklärungs-Stil:**
- Analogie/Metapher: [z.B. "Funktion = Rezept das man immer wieder kochen kann"]
- Bezug zu Vorwissen: [z.B. "Wie feuerball() im letzten Spiel"]
- Fachbegriffe die vorkommen: [z.B. "def, return, Parameter"]

**Hint-2 Analogie-Idee:** [Welches andere Beispiel zeigen wir? z.B. "Eine spawnStern() Funktion"]
**Hint-3 Platzhalter:** [Welche 3–6 Stellen werden ersetzt? z.B. "#FUNKTIONSNAME#, #SPRITE#, #X_POS#"]
**Bonus ⭐:** [Optionale Erweiterung, z.B. "Bereich der Zufallszahlen ändern"]
**Zusatzaufgabe 🌟:** [Für Schnelle, z.B. "Zweiten Feind-Typ mit anderer Farbe"]

### Aufgabe 2: [Titel]
...

## Halfway Motivation
- **Nach Aufgabe:** [N]
- **Stimmung:** [z.B. "Energetisch — sie haben gerade das erste Mal getestet"]
- **Was wurde schon erreicht:** [z.B. "Countdown + Teleport funktioniert"]
- **Was kommt noch:** [z.B. "Jetzt kommt der Kern: zufällige Blöcke + Punkte"]

## Easter Eggs (min. 3)
1. **Typ:** [z.B. "Klick-Counter"] **Wo:** [z.B. "Auf den Hero-Titel"] **Was passiert:** [z.B. "Nach 5x Klicks erscheint ein versteckter Spruch"]
2. **Typ:** [...] **Wo:** [...] **Was passiert:** [...]
3. **Typ:** [...] **Wo:** [...] **Was passiert:** [...]

## Erklärungs-Patterns (Learning Style)
> Dieser Abschnitt wird vom User überarbeitet und dient als Lern-Pattern
> für zukünftige Generationen. Beschreibt WIE Konzepte erklärt werden sollen.

- **Neue Konzepte einführen:** [z.B. "Immer erst fragen 'Erinnert ihr euch an X?' → Bezug herstellen → Dann neues Konzept als Erweiterung davon"]
- **Code erklären:** [z.B. "Zeile für Zeile durchgehen, jede Zeile in einem Satz auf Deutsch erklären"]
- **Fehler ansprechen:** [z.B. "Häufige Fehler proaktiv nennen: 'Achtung: hier vergessen viele das Komma'"]
- **Motivation:** [z.B. "Nach jedem erfolgreichen Test: kurze Bestätigung 'Nice, das läuft!'"]
- **Schwierigkeitsbalance:** [z.B. "Aufgaben 1–3 geführt, ab Aufgabe 4 mehr Eigenarbeit"]

## Aufteilungs-Empfehlung
[Wenn > 6 Aufgaben: Empfehlung ob 1 oder 2 HTML-Dateien. Begründung basierend auf
natürlichem Pausenpunkt, Aufmerksamkeitsspanne der Altersgruppe, und thematischem Bruch.
z.B. "Nach Aufgabe 4 (erster lauffähiger Test) → Teil 1 abschließen. Teil 2 beginnt mit
zufälligem Block als neues Feature."]
```

**What the user does with the blueprint:**
- Reviews and adjusts task count, order, difficulty
- Edits the "Erklärungs-Stil" per task to match their teaching voice
- Rewrites the "Erklärungs-Patterns" section in their own style — this becomes the learning pattern for future generations
- Decides on 1 vs 2 files
- Approves, adds, or changes easter eggs
- Adds/removes bonus tasks

**Return the edited blueprint to Claude → Step 2.**

### Step 2: HTML Generation

Take the approved blueprint and the raw script. Now generate the final HTML page(s).

For each file to generate:
1. Read the Paper style guide (or use inline tokens)
2. Pick theme colors based on blueprint Meta
3. Build the page following the Page Architecture (see above)
4. Apply the user's Erklärungs-Patterns from the blueprint for all descriptions and hint text
5. Use the user's Hint-2 ideas, Platzhalter lists, and Bonus suggestions from the blueprint
6. Place halfway motivation where the blueprint says
7. Plant easter eggs where the blueprint says
8. Run the Quality Checklist

**If the blueprint says 2 files:**
- File 1: Tasks up to the split point, its own celebration ("Teil 1 geschafft!") with a teaser for Teil 2
- File 2: Remaining tasks, hero references "Du hast in Teil 1 schon X gebaut", own celebration
- Both files are independent (full CSS/JS inline in each)

### Why this 2-step process matters

1. **Creative control** — the user shapes the pedagogy, not just the content
2. **Learning patterns** — the Erklärungs-Patterns section becomes institutional knowledge. Each edited blueprint teaches future Claude instances how this specific instructor teaches.
3. **Quality** — catching structural issues (too many tasks, wrong split point, missing concept) is cheap in a markdown file, expensive in a 500-line HTML file
4. **Reusability** — blueprints can be stored and adapted for similar sessions

---

## Language

Default language is **German** (the target audience is German-speaking teens in coding workshops).
Use "du" (informal singular). Keep technical terms in English where standard (function, loop, variable,
import, etc.) but explain them in German context. Emoji are OK in headings and motivation blocks,
but don't overdo it — max 1–2 per section header.

If the user provides content in English or asks for English, switch accordingly.

---

## Quality Checklist (verify before delivering)

- [ ] Single .html file, no external dependencies except Google Fonts
- [ ] TBJS Paper style: 0 border-radius, 2px borders, offset shadows, mono headlines, sans body
- [ ] Theme layer matches the session topic (3–6 custom properties)
- [ ] Sticky nav with clickable numbered progress dots that scroll to tasks
- [ ] Hero with eyebrow + display heading + preview box
- [ ] Every task has: step banner, h2, "Das machst du jetzt" box, description, 4-hint grid, checkbox
- [ ] Hint 1 always visible, Hints 2–4 progressively revealed
- [ ] Hint 3 has collapsible `<details>` for placeholder answers
- [ ] Checkpoint structure overviews at milestone tasks (not every task — only where blueprint says)
- [ ] Halfway motivation block between ~50% tasks
- [ ] Minimum 3 easter eggs, documented in HTML comment at bottom
- [ ] Final celebration appears when all checkboxes checked
- [ ] Mobile responsive (hint grid stacks, shadows shrink, touch targets ≥ 44px)
- [ ] All code examples use correct syntax for the session's language (Python, JS, etc.)
- [ ] German "du" form, technical terms in English where standard

---

## Kurse-Kompatibilität (Deployment über die Kurse-Mod)

Die fertige HTML-Datei wird **nicht hochgeladen**, sondern auf der Coach-Seite
(`kurse.simplecore.app`) in ein Datei-Feld eingefügt und gespeichert. Beim
Ausliefern an Teilnehmer rendert die Kurse-Shell die Seite in einem `iframe`
und injiziert automatisch `<script src="/static/kurse.js">` davor.

**Für Tracking sind KEINE Änderungen nötig** — `kurse.js` hakt das
Standard-Naming-Modell dieses Skills automatisch ein:

- `reveal(btn, taskNum, hintNum)` → meldet Tipp-Öffnung (mit serverseitiger Offenzeit)
- `markDone(n)` → meldet Fortschritt/Abschluss
- `id="task-N"` → meldet, welche Aufgabe erreicht wurde (per IntersectionObserver)
- `input[type=checkbox]` → Abschluss-Fallback, wenn alle angehakt sind

**Regeln, damit es sauber läuft (minimal, meist ohnehin erfüllt):**
1. Behalte die Funktionsnamen `reveal(...)` und `markDone(...)` bei (nicht umbenennen).
2. Behalte `id="task-0"`, `id="task-1"`, … für die Task-Sections (0-basiert).
3. Die Seite läuft in einem iframe — verlasse dich nicht auf `window.top`,
   und nutze normalen Dokument-Scroll (kein `position:fixed` Vollbild-Overlay,
   das den iframe sprengt).
4. Optionale, explizite Meldungen aus der Seite heraus sind erlaubt:
   `Kurse.task(i)`, `Kurse.hint(task, level)`, `Kurse.done()`.
   Der Neu-Bau-Weg `Kurse.tasks(el, spec)` erzeugt bereits verdrahtete Karten.

Die äußere Blatt-zu-Blatt-Navigation (◀ ▶ + Dots über die freigegebenen
Sessions) liefert die Kurse-Shell. Die seiteninternen `progress-dot`s scrollen
weiterhin nur innerhalb der eigenen Tasks — kein Konflikt.

======================================================================
# references/blueprint_template.md
======================================================================
# Blaupause: [Session-Titel hier]

> **Erstellt von:** Claude (workshop-page skill)
> **Zu überarbeiten von:** [Kursleiter-Name]
> **Datum:** [Heute]
> **Roh-Script:** [Dateiname des Input-Scripts]

---

## Meta

- **Thema:** [Was wird gebaut? z.B. "Parkour-Minispiel in Minecraft"]
- **Sprache:** [Python / JavaScript / Kaboom.js / etc.]
- **Theme:** [Minecraft / Space / Frosch / Web / Roboter / etc.]
- **Theme-Farben:** [z.B. "Diamond-Cyan #4AEDD9, Grass-Grün #5D8C3E, Gold #FCDB05"]
- **Altersgruppe:** [13–14 / 14–16 / gemischt]
- **Geschätzte Dauer:** [X min gesamt, davon Y min Setup]
- **Vorkenntnisse:** [Was müssen TN schon können? z.B. "Variablen, for-Schleife, Funktionsaufrufe"]
- **Dateien:** 1 HTML | oder 2 HTML (Teil 1: Aufgabe 1–N, Teil 2: Aufgabe N+1–M)

---

## Aufgaben-Übersicht

| #  | Titel                           | Kern-Konzept                | Dauer  | Schwierigkeit | Neuer Code-Block | Checkpoint? |
|----|---------------------------------|-----------------------------|--------|---------------|------------------|-------------|
| 0  | Setup                           | Editor + Verbindung starten | 3 min  | ☆             | —                | —           |
| 1  | [Titel]                         | [Konzept]                   | [X] min| ☆☆            | [z.B. "def ..."] | Nein        |
| 2  | [Titel]                         | [Konzept]                   | [X] min| ☆☆            | [z.B. "while ..."]| Ja — Grundgerüst steht |
| 3  | [Titel]                         | [Konzept]                   | [X] min| ☆☆☆           | [...]            | Nein        |
| ...| ...                             | ...                         | ...    | ...           | ...              | ...         |

**Schwierigkeit:** ☆ = geführt, copy-paste reicht · ☆☆ = mit Tipps machbar · ☆☆☆ = echte Denkarbeit
**Checkpoint?:** Strukturübersicht nach dieser Aufgabe zeigen? Nur bei Meilensteinen (Setup fertig, erster Test, vor neuem Thema, nach Halbzeit). "—" = kein Checkpoint.

---

## Pro Aufgabe — Detailplan

### Aufgabe 0: Setup
_Kein Hint-System nötig. Nur Checklist-Punkte._
- [ ] Editor/Notebook starten
- [ ] Verbindung herstellen / Server beitreten
- [ ] Datei umbenennen

---

### Aufgabe 1: [Titel]

**Kern-Konzept:** [Was lernen sie? Ein Satz. z.B. "Eine Funktion definieren die einen Sprite an zufälliger Position erzeugt"]

**Erklärungs-Stil:**
- **Analogie/Metapher:** [z.B. "Eine Funktion ist wie ein Rezept — einmal aufschreiben, beliebig oft kochen"]
- **Bezug zu Vorwissen:** [z.B. "Wie die feuerball()-Funktion im vorherigen Spiel"]
- **Fachbegriffe:** [z.B. "def, function, Parameter, return"]
- **Häufiger Fehler:** [z.B. "Einrückung vergessen → IndentationError"]
- **Erklär-Tiefe:** [KURZ: 1 Satz reicht | MITTEL: 2–3 Sätze + Verweis | AUSFÜHRLICH: Schritt für Schritt]

**4 Hints:**
- **Hint 1 (Wo & Was):** [Stichpunkte: Wo im Code einfügen? Was ist das Ziel? Welche Befehle braucht man?]
- **Hint 2 (Analogie-Beispiel):** [Welches ANDERE Beispiel zeigen? z.B. "spawnStern()-Funktion mit festen Koordinaten" — NICHT die Lösung]
- **Hint 3 (Platzhalter):** [Liste: #NAME# = ?, #SPRITE# = ?, #X# = ? — welche 3–6 Stellen ersetzen?]
- **Hint 4 (Lösung):** [Where-Tag: "Einfügen in: scene('game') nach den Rubinen"]

**Bonus ⭐:** [Optionale Erweiterung für alle, z.B. "Bereich der Zufallszahlen ändern"]
**Zusatzaufgabe 🌟:** [Nur für Schnelle, z.B. "Zweiten Feind-Typ mit anderer Farbe"]
**Checkpoint:** [Nein | Ja — was zeigen? z.B. "Imports + countdown() + parkourSpiel() Grundgerüst, while-Schleife noch grau"]

**Kern-Konzept:** [...]

**Erklärungs-Stil:**
- **Analogie/Metapher:** [...]
- **Bezug zu Vorwissen:** [...]
- **Fachbegriffe:** [...]
- **Häufiger Fehler:** [...]
- **Erklär-Tiefe:** [KURZ / MITTEL / AUSFÜHRLICH]

**4 Hints:**
- **Hint 1 (Wo & Was):** [...]
- **Hint 2 (Analogie-Beispiel):** [...]
- **Hint 3 (Platzhalter):** [...]
- **Hint 4 (Lösung):** [...]

**Bonus ⭐:** [...]
**Zusatzaufgabe 🌟:** [...]
**Checkpoint:** [Nein | Ja — was zeigen?]

---

_[Weitere Aufgaben nach demselben Pattern...]_

---

## Halfway Motivation

- **Nach Aufgabe:** [N — ungefähr 50%]
- **Stimmung:** [z.B. "Energetisch" / "Beruhigend" / "Hyped"]
- **Was wurde erreicht:** [z.B. "Countdown funktioniert, Spieler wird teleportiert, erster Block steht"]
- **Was kommt noch:** [z.B. "Jetzt wird's spannend: zufällige Blöcke + Punkte + Game Over"]
- **Vorgeschlagener Text:** [1–2 Sätze, z.B. "Schon über die Hälfte! Euer Spieler fliegt, der Countdown tickt — jetzt bauen wir das Herzstück."]

---

## Easter Eggs (min. 3)

| # | Typ               | Wo (Element)           | Was passiert                                       | Schwer zu finden? |
|---|-------------------|------------------------|----------------------------------------------------|--------------------|
| 1 | [z.B. Klick-Counter] | [z.B. Hero-Titel]     | [z.B. "Nach 5x Klick: '🎮 Secret Level: Du bist neugierig — gut so!'"] | Mittel |
| 2 | [z.B. Hover-Titel]   | [z.B. Celebration-Box] | [z.B. "title='Diamanten sind der beste Freund eines Coders'"] | Leicht |
| 3 | [z.B. Konami Code]   | [z.B. Global/Body]     | [z.B. "↑↑↓↓←→←→ löst Matrix-Rain CSS-Animation aus"] | Schwer |

---

## Erklärungs-Patterns (Learning Style)

> **WICHTIG:** Dieser Abschnitt ist das Herzstück des Blueprints.
> Überarbeite ihn in DEINEM Stil. Was hier steht wird als Lern-Pattern für
> zukünftige Generationen verwendet — es bestimmt, wie Claude Konzepte für
> deine Zielgruppe erklärt.

### Wie führe ich neue Konzepte ein?
[z.B. "Immer erst fragen: 'Erinnert ihr euch an X aus letzter Woche?' → Bezug herstellen → Neues Konzept als natürliche Erweiterung davon einführen → Konkretes Beispiel im Spiel zeigen"]

### Wie erkläre ich Code?
[z.B. "Nie den ganzen Block auf einmal. Zeile für Zeile, jede Zeile ein deutscher Satz. Bei komplizierten Zeilen: erst den Effekt beschreiben ('Das sorgt dafür, dass...'), dann den Code-Teil"]

### Wie gehe ich mit Fehlern um?
[z.B. "Häufige Fehler proaktiv nennen BEVOR sie passieren: 'Achtung: Hier vergessen viele das Komma nach dem letzten Element.' — Nicht nach dem Fehler erklären, sondern vorher warnen"]

### Wie motiviere ich?
[z.B. "Nach jedem erfolgreichen Test: 'Nice, das läuft!' / 'Stark!' — Kurz und echt, nicht übertrieben. Bei Schwierigkeiten: 'Das ist normal, das haben wir alle am Anfang falsch gemacht.'"]

### Wie balanciere ich Schwierigkeit?
[z.B. "Aufgaben 1–3: stark geführt, fast Copy-Paste mit kleinen Anpassungen. Ab Aufgabe 4: mehr 'Versucht es erstmal selbst'. Aufgabe 6+: nur noch 'Ihr wisst wie es geht — los.'"]

### Wie viel Eigenarbeit erwarte ich?
[z.B. "Hint 1 sollen ALLE lesen. Hint 2 brauchen ~60% der TN. Hint 3 brauchen ~30%. Hint 4 (Lösung) nur die die wirklich nicht weiterkommen — das Ziel ist, dass die meisten bei Hint 2–3 stoppen."]

---

## Aufteilungs-Empfehlung

**Empfehlung:** [1 Datei / 2 Dateien]

**Begründung:** [z.B. "8 Aufgaben sind für eine Session machbar, ABER nach Aufgabe 4 gibt es einen
natürlichen Pausenpunkt (erster lauffähiger Test). Für jüngere Gruppen (13–14): aufteilen.
Für ältere (15–16): 1 Datei mit Halfway-Motivation reicht."]

**Falls 2 Dateien:**
- Teil 1: Aufgabe 0–[N] → Dateiname: `[thema]_teil1.html`
- Teil 2: Aufgabe [N+1]–[M] → Dateiname: `[thema]_teil2.html`
- Bruchstelle: [z.B. "Nach Aufgabe 4 — da funktioniert der Grundmechanismus"]

---

## Notizen für Step 2

_[Platz für Anmerkungen die beim HTML-Generieren beachtet werden sollen.
z.B. "Die TN haben kein loadSprite für 'feind' — Hint 4 von Aufgabe 1 muss darauf hinweisen."
oder "Server-Adresse ist mc15.codary.org — als Platzhalter im Code lassen."]_

---

<!-- 
NACH DER ÜBERARBEITUNG:
Gib diese Datei an Claude zurück mit der Nachricht:
"Hier ist die überarbeitete Blaupause. Generiere die HTML-Seite(n)."
Claude verwendet dann den workshop-page Skill Step 2 um die finale(n) Datei(en) zu erstellen.
-->

======================================================================
# references/creation_steps.md
======================================================================
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

======================================================================
# references/nbpaper_style.md
======================================================================
# TBJS Paper — Neo-Brutalism Side-App Style (Reference for workshop-page skill)

> Copy of the full TBJS Paper design system tokens. The SKILL.md references this file.
> When building a workshop page, read this file FIRST to get all CSS variables, component patterns,
> shadow recipes, typography rules, and do's/don'ts.

The full style guide is at: /mnt/user-data/uploads/nbpaper_style.md
If that file is present, read it directly — it is the canonical source.
If not present, use the tokens and patterns defined in the SKILL.md inline reference section.
