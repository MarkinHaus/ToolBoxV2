# TBJS Terminal — Internal Tools Style

> **System:** TBJS Design System v3.0 — Dark Terminal Variant
> **Scope:** Internal-facing surfaces — admin panels, log viewers, icli web shells, ISAA monitors, registry admin, worker dashboards. Anything an operator sees, not a user.
> **Philosophy:** *Phosphor precision.* Monospace-only, scanline-aware, no chrome, no glass, no decoration. Text is the interface.
> **Relation to Glass:** Shares the same `--raw-primary` hue (blue `230`). Swap a user from `simplecore.app` into an internal tool and the accent color stays home. Everything else is sharper.
> **Format:** Stitch `DESIGN.md` compatible.

---

## 1. Visual Theme & Atmosphere

**Mood.** A real terminal. Not a skeuomorphic terminal theme — a terminal. Monospace across the entire UI including headings and buttons. Black background. Phosphor-bright accent color. Borders are 1px sharp lines, not rounded glass edges.

**Density.** Extreme. Body at **12px** mono. Tables are walls of data. Line height 1.4. Information per vertical inch is maximized.

**Design principles**

- **Mono everywhere.** Not just code. Headings, buttons, labels, nav — all `IBM Plex Mono`. The only sans-serif text is within user-generated content (markdown bodies inside a `.markdown-body` wrapper).
- **No glass, no blur, no gradient, no shadow.** Every surface is a flat `rgba()` or `#hex`. Elevation is communicated via a 1px border shift, never a shadow.
- **Text as chrome.** Buttons look like `[ label ]`. Active nav items get a `> ` prefix. Menus are `─` separated.
- **Cursor-aware.** Focus states show a solid 1-character-wide accent bar, not an outline halo. Inputs have a blinking caret indicator for live streams.
- **Block, not pill.** `--term-radius-sm: 2px` is the maximum corner radius. Most things are `border-radius: 0`.
- **Status is a character.** `●` for connected, `○` for idle, `✕` for error, `▲` for warning. Mono ligatures carry semantic weight.
- **Alignment over spacing.** Columns align to character boundaries when possible. Think tmux, not Tailwind.

---

## 2. Color Palette & Roles

### Shared with Glass — same hue family

The raw OKLCH tokens are identical to `tbjs-main.css`. A terminal view of a system and a glass view of the same system share accent. Only surfaces differ.

```css
--raw-primary: 55% 0.18 230;     /* same deep-tech blue as Glass */
--raw-success: 65% 0.2 145;
--raw-warning: 75% 0.18 85;
--raw-error:   55% 0.22 25;
--raw-info:    60% 0.15 230;

--primary: oklch(var(--raw-primary));
--success: oklch(var(--raw-success));
--warning: oklch(var(--raw-warning));
--error:   oklch(var(--raw-error));
--info:    oklch(var(--raw-info));
```

### Terminal-specific surfaces

| Token | Value | Role |
|---|---|---|
| `--term-bg` | `#000000` | True black canvas |
| `--term-bg-raised` | `#0a0a0f` | Panels, cards (1 step up) |
| `--term-bg-sunken` | `#030306` | Input fields, code blocks |
| `--term-border` | `rgba(255, 255, 255, 0.12)` | Default 1px rule |
| `--term-border-active` | `var(--primary)` | Focused / hovered input, active tab |
| `--term-selection` | `color-mix(in oklch, var(--primary) 30%, transparent)` | Text selection, row highlight |

### Terminal text

Brighter and harder than Glass text — phosphor, not paper.

| Token | Value | Role |
|---|---|---|
| `--term-fg` | `rgba(255, 255, 255, 0.92)` | Body — brighter than Glass (0.85) |
| `--term-fg-dim` | `rgba(255, 255, 255, 0.5)` | Labels, timestamps |
| `--term-fg-muted` | `rgba(255, 255, 255, 0.3)` | Divider labels, hints |
| `--term-fg-accent` | `var(--primary)` | Commands, active items, prompts |

### Status characters — glyphs are part of the palette

| Glyph | Color | Meaning |
|---|---|---|
| `●` | `--success` | Connected, online, ready |
| `○` | `--term-fg-muted` | Idle, disabled |
| `◐` | `--warning` | Pending, starting |
| `✕` | `--error` | Failed, disconnected |
| `▲` | `--warning` | Attention, caution |
| `→` | `--primary` | Active selection, focus |

### No light mode

Terminal is dark-only by design. Operators work in low-light environments; a light terminal is a category error. If a light-mode page is needed, use Glass or Paper.

---

## 3. Typography Rules

### Font — mono only

```css
--font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, 'Cascadia Code', monospace;
```

There is no `--font-sans` in this style. All headings, buttons, labels, body text, and UI chrome use mono. The only exception: content rendered inside `.markdown-body` may use sans, but the chrome around it stays mono.

### Type scale — tighter, no fluid scaling

Terminals don't scale. Readability comes from consistency, not responsive sizing.

| Token | Value | Role |
|---|---|---|
| `--term-text-h1` | `16px` | Page title |
| `--term-text-h2` | `14px` | Section heading |
| `--term-text-h3` | `13px` | Subsection |
| `--term-text-base` | `12px` | Body default |
| `--term-text-sm` | `11px` | Secondary |
| `--term-text-xs` | `10px` | Timestamps, micro-labels |

Line-height is **always 1.4** — tight enough for data density, loose enough to scan.

### Heading treatment

No weight variation — all headings are weight 500. Mono doesn't render weight contrast reliably at small sizes. Differentiation comes from size and a leading `#` prefix.

```css
h1::before { content: "# ";   color: var(--term-fg-muted); }
h2::before { content: "## ";  color: var(--term-fg-muted); }
h3::before { content: "### "; color: var(--term-fg-muted); }
```

The `#` is part of the design, echoing markdown source. Operators read markdown daily — this makes the page look like what they'd type.

### Prompt prefix

Shell-like prompts for command-relevant headings:

```css
.prompt-line { color: var(--term-fg-accent); }
.prompt-line::before {
  content: "$ ";
  color: var(--success);
}
```

For root/admin contexts, `#` replaces `$`.

### Body paragraph

`max-inline-size: 80ch` — the terminal-traditional line length. Not 65ch like Glass.

```css
p {
  font-size: var(--term-text-base);
  line-height: 1.4;
  max-inline-size: 80ch;
  color: var(--term-fg-dim);
}
```

---

## 4. Component Stylings

### Button — bracketed label

Buttons look like `[ action ]`. The brackets are part of the button, not CSS borders.

```css
.term-btn {
  font-family: var(--font-mono);
  font-size: var(--term-text-base);
  font-weight: 500;
  padding: 4px 12px;
  background: transparent;
  color: var(--term-fg);
  border: 1px solid var(--term-border);
  border-radius: 0;
  cursor: pointer;
  transition: all 100ms linear;  /* linear, not ease — instant feel */
}
.term-btn::before { content: "[ "; color: var(--term-fg-muted); }
.term-btn::after  { content: " ]"; color: var(--term-fg-muted); }

.term-btn:hover {
  border-color: var(--primary);
  color: var(--primary);
}
.term-btn:hover::before,
.term-btn:hover::after { color: var(--primary); }

.term-btn:focus-visible {
  outline: none;
  background: var(--term-selection);
}
.term-btn:active {
  background: var(--primary);
  color: var(--term-bg);
}
```

**Primary variant** — solid primary fill:

```css
.term-btn-primary {
  background: var(--primary);
  color: var(--term-bg);
  border-color: var(--primary);
}
.term-btn-primary::before,
.term-btn-primary::after { color: var(--term-bg); }
```

**Danger variant** — swap `--primary` for `--error`.

**No transform on hover.** Terminals don't lift — they invert. Hover flips foreground/border color; that's the signal.

### Card — framed panel

No rounded corners, no shadow. A 1px border and a 1-line header.

```css
.term-card {
  background: var(--term-bg-raised);
  border: 1px solid var(--term-border);
  border-radius: 0;
  padding: 12px 16px;
  font-family: var(--font-mono);
}
.term-card-title {
  font-size: var(--term-text-sm);
  color: var(--term-fg-dim);
  text-transform: uppercase;
  letter-spacing: 1px;
  padding-bottom: 8px;
  margin-bottom: 8px;
  border-bottom: 1px solid var(--term-border);
}
```

Optional: ASCII-art framing. `.term-card.boxed` renders a full `┌─┐/└─┘` border via `::before`/`::after` content. Use sparingly for emphasis.

### Input — caret-aware

```css
.term-input {
  width: 100%;
  font-family: var(--font-mono);
  font-size: var(--term-text-base);
  color: var(--term-fg);
  background: var(--term-bg-sunken);
  border: 1px solid var(--term-border);
  border-radius: 0;
  padding: 6px 10px;
  caret-color: var(--primary);
  transition: border-color 100ms linear;
}
.term-input:focus {
  outline: none;
  border-color: var(--primary);
  background: #000;  /* deeper black on focus, not lighter */
}
.term-input::placeholder {
  color: var(--term-fg-muted);
}
```

**Blinking cursor** for live/readonly streams:

```css
.term-input.live::after {
  content: "▌";
  color: var(--primary);
  animation: caret-blink 1s steps(2) infinite;
}
@keyframes caret-blink { 50% { opacity: 0; } }
```

### Navigation — breadcrumb path

Terminal nav is a shell path, not a menu bar.

```css
.term-nav {
  font-family: var(--font-mono);
  font-size: var(--term-text-sm);
  padding: 8px 16px;
  background: var(--term-bg);
  border-bottom: 1px solid var(--term-border);
  color: var(--term-fg-dim);
  display: flex;
  align-items: center;
}
.term-nav a {
  color: var(--primary);
  text-decoration: none;
  padding: 0 4px;
}
.term-nav .separator::before {
  content: "/";
  color: var(--term-fg-muted);
  padding: 0 4px;
}
.term-nav .current { color: var(--term-fg); }
```

Rendered: `/ simplecore / workers / ryzen-01`

### Tab navigation — no pills

```css
.term-tabs {
  display: flex;
  border-bottom: 1px solid var(--term-border);
  font-family: var(--font-mono);
  font-size: var(--term-text-sm);
}
.term-tab {
  padding: 8px 16px;
  color: var(--term-fg-dim);
  background: transparent;
  border: none;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  cursor: pointer;
}
.term-tab:hover { color: var(--term-fg); }
.term-tab.is-active {
  color: var(--primary);
  border-bottom-color: var(--primary);
}
.term-tab.is-active::before {
  content: "> ";
  margin-left: -12px;
}
```

### Log viewer — the signature component

The most important internal component. A monospace scroll with timestamp, level, source, message columns. Fixed grid, hover highlights row.

```css
.term-log {
  font-family: var(--font-mono);
  font-size: var(--term-text-xs);
  line-height: 1.5;
  background: var(--term-bg);
  padding: 8px 0;
  overflow-y: auto;
}
.term-log-row {
  display: grid;
  grid-template-columns: 180px 60px 140px 1fr;
  gap: 12px;
  padding: 2px 16px;
  border-left: 2px solid transparent;
  color: var(--term-fg-dim);
  transition: background 80ms linear;
}
.term-log-row:hover {
  background: var(--term-selection);
  border-left-color: var(--primary);
}
.term-log-row .ts     { color: var(--term-fg-muted); }
.term-log-row .level  { font-weight: 500; }
.term-log-row .source { color: var(--term-fg-dim); }
.term-log-row .msg    { color: var(--term-fg); white-space: pre-wrap; }

.term-log-row.level-error   .level { color: var(--error); }
.term-log-row.level-warn    .level { color: var(--warning); }
.term-log-row.level-info    .level { color: var(--info); }
.term-log-row.level-success .level { color: var(--success); }
```

### Status bar — tmux style

Bottom-fixed, single line, mono, phosphor. Multiple segments separated by `│`.

```css
.term-status-bar {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  height: 22px;
  background: var(--term-bg-raised);
  border-top: 1px solid var(--term-border);
  font-family: var(--font-mono);
  font-size: var(--term-text-xs);
  color: var(--term-fg-dim);
  display: flex;
  align-items: center;
  padding: 0 12px;
  gap: 12px;
}
.term-status-bar .seg + .seg::before {
  content: "│";
  color: var(--term-border);
  margin-right: 12px;
}
.term-status-bar .dot-connected::before {
  content: "●";
  color: var(--success);
  margin-right: 4px;
}
```

Rendered: `● ryzen-01  │  CPU 42%  │  MEM 18.2G/48G  │  12:04:33`

### Table — aligned grid

```css
.term-table {
  display: grid;
  font-family: var(--font-mono);
  font-size: var(--term-text-sm);
  border: 1px solid var(--term-border);
  background: var(--term-bg-raised);
}
.term-table-header { display: contents; }
.term-table-header .cell {
  padding: 6px 12px;
  background: var(--term-bg);
  border-bottom: 1px solid var(--term-border);
  color: var(--term-fg-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-size: var(--term-text-xs);
}
.term-table-row { display: contents; }
.term-table-row .cell {
  padding: 4px 12px;
  border-bottom: 1px solid var(--term-border);
  color: var(--term-fg);
}
.term-table-row:hover .cell {
  background: var(--term-selection);
}
.term-table-row.selected .cell {
  background: color-mix(in oklch, var(--primary) 15%, transparent);
  border-left: 2px solid var(--primary);
}
```

### Modal — boxed dialog

Centered, not full-screen. No overlay blur. Hard border, no shadow.

```css
.term-modal-overlay {
  position: fixed; inset: 0;
  background: rgba(0, 0, 0, 0.85);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
}
.term-modal {
  background: var(--term-bg);
  border: 1px solid var(--primary);  /* accent border for modals */
  border-radius: 0;
  min-width: 400px;
  max-width: 90vw;
  font-family: var(--font-mono);
}
.term-modal-header {
  padding: 8px 12px;
  background: var(--primary);
  color: var(--term-bg);
  font-size: var(--term-text-sm);
  font-weight: 500;
  display: flex;
  justify-content: space-between;
}
.term-modal-body {
  padding: 16px;
  color: var(--term-fg);
  font-size: var(--term-text-base);
}
.term-modal-footer {
  padding: 8px 12px;
  border-top: 1px solid var(--term-border);
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}
```

The accent-colored header bar is the terminal modal's identity — like a window manager title bar.

### Toast — corner line

No blur, no glass. A single colored border-left on a black panel.

```css
.term-toast {
  font-family: var(--font-mono);
  font-size: var(--term-text-sm);
  background: var(--term-bg);
  border: 1px solid var(--term-border);
  border-left: 3px solid var(--toast-color, var(--primary));
  border-radius: 0;
  padding: 8px 12px;
  color: var(--term-fg);
  min-width: 240px;
}
.term-toast-title {
  color: var(--toast-color, var(--primary));
  font-size: var(--term-text-xs);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 4px;
}
```

### Command palette

Signature internal component — `Ctrl+K` invokes, shows fuzzy-matched commands.

```css
.term-palette {
  position: fixed;
  top: 20%;
  left: 50%;
  transform: translateX(-50%);
  width: 600px;
  max-width: 90vw;
  background: var(--term-bg);
  border: 1px solid var(--primary);
  font-family: var(--font-mono);
}
.term-palette-input {
  width: 100%;
  padding: 12px 16px;
  background: transparent;
  border: none;
  border-bottom: 1px solid var(--term-border);
  color: var(--term-fg);
  font-family: inherit;
  font-size: var(--term-text-base);
}
.term-palette-item {
  display: flex;
  justify-content: space-between;
  padding: 6px 16px;
  color: var(--term-fg-dim);
  font-size: var(--term-text-sm);
  cursor: pointer;
}
.term-palette-item.is-active {
  background: var(--term-selection);
  color: var(--term-fg);
  border-left: 2px solid var(--primary);
  padding-left: 14px;
}
.term-palette-item .shortcut {
  color: var(--term-fg-muted);
  font-size: var(--term-text-xs);
}
```

---

## 5. Layout Principles

### Spacing scale — 4px rhythm

Half the Glass scale. Terminals live in 4px increments, not 8.

```
--term-space-1:  2px
--term-space-2:  4px
--term-space-3:  8px
--term-space-4: 12px
--term-space-5: 16px
--term-space-6: 24px
--term-space-8: 32px
```

### Radius — almost nothing

```
--term-radius-sm:   2px   /* inputs, buttons */
--term-radius-none: 0     /* cards, modals, panels, tables */
```

Most surfaces are `border-radius: 0`.

### Border width — one value

```
--term-border-width: 1px
```

No 2px, 3px borders anywhere. The only exception is the accent `border-left: 2px` on active rows — the width shift itself is the selection signal.

### Grid

Layouts anchor to character grids when possible. For a page with a sidebar + main area, prefer `grid-template-columns: 40ch 1fr` over pixel widths.

### Z-index — flatter

```
--term-z-raised:   10
--term-z-dropdown: 100
--term-z-modal:   1000
--term-z-toast:   1100
```

No z-index inflation. Terminal UIs don't stack much.

---

## 6. Depth & Elevation

**There is no elevation.** This is the hardest rule and also the easiest.

No shadows. No `box-shadow` anywhere in this style. No gradients to suggest depth. No inset highlights. Surfaces differ only by:

1. **Background brightness** — `var(--term-bg)` → `var(--term-bg-raised)` → `var(--term-bg-sunken)`.
2. **Border presence** — 1px `var(--term-border)` around elevated containers.
3. **Border color shift** — hover/active flips to `var(--primary)`.

Why: shadows imply atmospheric perspective, which implies a physical world. Terminals are information — they exist in the same plane.

If a component *needs* to feel elevated (rare), darken the outer context instead of raising the surface. The `.term-modal-overlay` at 85% black is the only such technique.

---

## 7. Do's and Don'ts

### Do

- **Use mono for everything UI-related.** Including buttons, nav, headings, labels. Sans appears only inside user content.
- **Use semantic glyphs.** `●`, `○`, `→`, `✕`, `▲` — these are part of the palette.
- **Flip, don't lift.** Hover inverts foreground/background or swaps border color. It does not translate.
- **Prefix active items with `> `.** This is the terminal convention for selection.
- **Align to character columns.** Width in `ch`, not `px`, when possible.
- **Use `linear` transitions.** Not ease. Terminals feel instant.
- **Keep line-height at 1.4.** Every component. Consistency over perfection.
- **Use the same `--raw-primary` as Glass.** Internal tools are the same system, different view.

### Don't

- **Don't use `--font-sans`.** There is no sans in this style.
- **Don't use `box-shadow`.** Zero shadows, zero glows, zero blur.
- **Don't round corners.** Max `--term-radius-sm: 2px`, and only for inputs/buttons.
- **Don't use Glass tokens.** `--glass-bg`, `--glass-border`, `--glass-blur` do not exist here. Use `--term-bg-raised`, `--term-border`.
- **Don't use `color-mix` for chrome.** Surfaces are concrete values. `color-mix` is reserved for `--term-selection` and active row fills.
- **Don't implement a light theme.** Terminal is dark-only. If a light version is required, the caller is using the wrong style — route to Glass or Paper.
- **Don't use emojis as status.** Use the glyph palette (`●○◐✕▲→`). Emojis render inconsistently in mono fonts.
- **Don't lift on hover.** No `translateY(-1px)`. Terminals don't move.
- **Don't use fluid typography.** Fixed sizes only.

---

## 8. Responsive Behavior

### Breakpoints

```css
@media (max-width: 960px) { /* tablet */ }
@media (max-width: 640px) { /* mobile */ }
```

### Mobile adjustments

Terminal UI is operator-focused — mobile is a secondary experience, but not hostile.

- `.term-log-row` grid collapses: timestamp + level on row 1, source + message on row 2.
- `.term-table` horizontal-scrolls inside its border rather than reflowing. Maintains alignment.
- `.term-nav` truncates with ellipsis on narrow screens, shows only current + parent.
- `.term-status-bar` shows only the leftmost 2 segments; rest collapses to a `[...]` glyph that expands on tap.
- Font sizes do not scale.

### Touch targets

Inputs bump from 6px to 10px vertical padding. Buttons bump to 8px 16px. Log rows bump to 6px vertical.

Below 640px, tighten grid gutters but keep character alignment.

---

## 9. Agent Prompt Guide

### Quick color reference

```
Accent / commands   → var(--primary)
Success / online    → var(--success)
Warning             → var(--warning)
Error               → var(--error)

Body text           → var(--term-fg)
Dim (labels, ts)    → var(--term-fg-dim)
Muted (hints)       → var(--term-fg-muted)

Canvas              → var(--term-bg)
Raised panel        → var(--term-bg-raised) + border
Input / code block  → var(--term-bg-sunken) + border
Selection tint      → var(--term-selection)
```

### Ready-to-use prompts

**"Build an internal admin panel"**
> Use TBJS Terminal. Mono font throughout. Root background `--term-bg`. Panels are `--term-bg-raised` with 1px `--term-border`. Headings get `#` prefix. Buttons look like `[ label ]`. Active nav items prefixed with `> `. No shadows, no gradients, no rounded corners beyond 2px on inputs.

**"Show a log stream"**
> Use `.term-log` with `.term-log-row` grid rows. Columns: 180px timestamp, 60px level, 140px source, 1fr message. Level classes `.level-error`, `.level-warn`, `.level-info`, `.level-success` color the level text. Row hover shows `--term-selection` background and a 2px `--primary` left border.

**"Add a status indicator"**
> Use a glyph, not a colored dot div. `●` in `--success` for connected, `○` in `--term-fg-muted` for idle, `◐` in `--warning` for pending, `✕` in `--error` for failed. Wrap in a `<span>` with appropriate color.

**"Match Glass in the accent color"**
> Both styles read the same `--raw-primary: 55% 0.18 230`. Re-use the token, not a copied hex. A user switching from `simplecore.app` (Glass) to `registry.simplecore.app/admin` (Terminal) sees the same blue.

**"Build a command palette"**
> Use `.term-palette` — fixed `top: 20%`, centered, `width: 600px`, `--primary` border. Input has `> ` prefix. Items show label left, shortcut right (`--term-fg-muted`). Active item has `--term-selection` background and 2px `--primary` left border with `padding-left: 14px` to compensate.

### The three questions before adding CSS

1. **Am I using mono?** If no, stop. Switch to mono or check if you're in the wrong style (should be Glass).
2. **Am I adding a shadow, gradient, or blur?** If yes, stop. Find another way — usually a border-color or background-brightness shift.
3. **Am I lifting on hover?** If yes, replace with an invert (flip color / background / border).

### Cross-style relationship

- **Glass ↔ Terminal:** same hue, different surfaces. Data displayed in one translates cleanly to the other. `--primary`, `--success`, `--warning`, `--error` are literally the same tokens.
- **Paper ↔ Terminal:** opposite extremes. Paper is daylight, Terminal is night-shift. They don't share a page.
- **Glass sits in between** — suitable for users; Terminal for operators; Paper for focus work (docs, forms, side-apps).

---

*This document is the contract for TBJS Terminal. It applies to internal admin surfaces, log viewers, worker dashboards, and any context where the viewer is an operator, not a user.*
