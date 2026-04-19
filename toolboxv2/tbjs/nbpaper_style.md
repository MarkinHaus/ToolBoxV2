# TBJS Paper — Neo-Brutalism Side-App Style

> **System:** TBJS Design System v3.0 — Paper Variant
> **Scope:** Side-applications, landing pages, docs, static content, marketing microsites. Anything that is *not* the main SimpleCore instrument panel and *not* an internal terminal tool.
> **Philosophy:** *Honest materials.* Hard edges, thick borders, offset drop shadows, visible ink. The interface is a printed page, not a glass HUD.
> **Format:** Stitch `DESIGN.md` compatible. Lives alongside the Glass and Terminal variants; shares the OKLCH color primitives and spacing scale.

---

## 1. Visual Theme & Atmosphere

**Mood.** A printed technical zine on warm off-white paper. Monospace headlines, thick black rules, hand-placed offset shadows. Rough around the edges on purpose — the *opposite* of Glass's smooth chrome.

**Density.** Medium. Paper breathes more than Glass. Body at 16px (not 13px), headings up to 40px. The aesthetic rewards whitespace; use it.

**Design principles**

- **Hard edges, visible borders.** Every surface has a `2px solid var(--ink)` border. No glass, no `backdrop-filter`, no translucency. What you see is the full surface.
- **Offset drop shadows, not ambient.** Shadows are a hard `4px 4px 0 var(--ink)` — a displaced copy of the shape, not a soft blur. Depth is *parallax*, not *glow*.
- **Shared color primitives.** Pulls from the same `--raw-primary`, `--raw-success` etc. as Glass. Color identity across all three styles is unified — only the *treatment* differs.
- **Monospace headlines.** IBM Plex Mono is the *display* face here, not the data face. Sans is the body. Inversion vs Glass: in Paper, mono is decorative, sans is functional.
- **Paper base, not void.** Default background is a warm off-white (`#f4f1ea`). Dark mode exists but is deep charcoal (`#1a1a1a`), not void black. Paper must always read as *paper*.
- **No motion drift.** Hover is a 2px translate diagonally (`translate(-2px, -2px)`) combined with shadow offset increasing — the card "lifts off the page". No scale, no fade.

---

## 2. Color Palette & Roles

### Shared primitive tokens

Same `--raw-*` OKLCH tuples as Glass. Re-stated so the file stands alone:

```css
--raw-primary: 55% 0.18 230;  /* Deep-tech blue */
--raw-success: 65% 0.2 145;
--raw-warning: 75% 0.18 85;
--raw-error:   55% 0.22 25;
--raw-info:    60% 0.15 230;
```

### Paper-specific surfaces (default — "Paper Light")

| Token | Value | Role |
|---|---|---|
| `--paper-bg` | `#f4f1ea` | Warm off-white — the page itself |
| `--paper-surface` | `#ffffff` | Card, callout, form field |
| `--paper-sunken` | `#ebe7dc` | Input field, inset areas |
| `--ink` | `#111111` | Primary text *and* border color — unified |
| `--ink-muted` | `#555555` | Secondary text |
| `--ink-faint` | `#888888` | Captions, metadata |
| `--rule` | `#111111` | Dividers, horizontal rules — always full-strength |

### Paper Dark

Not a light-to-dark inversion. Paper Dark is *carbon paper* — dark page, light ink.

| Token | Value |
|---|---|
| `--paper-bg` | `#1a1a1a` |
| `--paper-surface` | `#2a2a2a` |
| `--paper-sunken` | `#0f0f0f` |
| `--ink` | `#f4f1ea` |
| `--ink-muted` | `#b8b3a8` |
| `--ink-faint` | `#7a7770` |
| `--rule` | `#f4f1ea` |

### Accent usage

- `--primary` is used **sparingly** in Paper — primarily for hyperlinks (underlined, always) and the primary button fill.
- Semantic tokens (`--success`, `--error`, `--warning`) appear as solid color blocks and badge fills, never as gradients or glows.
- No `color-mix()` dynamic surfaces. Paper's surfaces are static and defined.

---

## 3. Typography Rules

### Font families — Mono for headlines, Sans for body

**Inversion of Glass.** In Paper, mono is the display face.

```css
--font-display: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
--font-body:    'IBM Plex Sans', system-ui, -apple-system, sans-serif;
```

Serif is explicitly not used. Paper is a *technical* zine, not a literary one.

### Type scale — generous

| Token | Value | Role |
|---|---|---|
| `--text-display` | `clamp(32px, 5vw, 48px)` | Hero headline — display mono, weight 600 |
| `--text-h1` | `clamp(28px, 3.5vw, 36px)` | Page title |
| `--text-h2` | `clamp(22px, 2.5vw, 28px)` | Section |
| `--text-h3` | `clamp(18px, 2vw, 22px)` | Subsection |
| `--text-base` | `16px` | Body — **larger than Glass's 13px** |
| `--text-sm` | `14px` | Captions, meta |
| `--text-xs` | `12px` | Micro-labels |

### Heading treatment

All headings use `--font-display` (mono).

| Property | Value |
|---|---|
| `font-weight` | 600 (display, h1), 500 (h2, h3) |
| `line-height` | 1.15 |
| `letter-spacing` | `-0.02em` for display/h1, `0` for h2+ |
| `text-transform` | none (mono already reads as decorative) |
| `color` | `var(--ink)` |

### Body

```css
p {
  font-family: var(--font-body);
  font-size: var(--text-base);
  line-height: 1.6;
  max-inline-size: 68ch;
  color: var(--ink);
  margin-block-end: 1.2em;
}
```

### Link — always visible

No hover-to-reveal underlines. Links are underlined *always*, with thickness 2px and offset 3px.

```css
a {
  color: var(--primary);
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-underline-offset: 3px;
  text-decoration-color: var(--primary);
}
a:hover {
  background: var(--primary);
  color: var(--paper-surface);
  text-decoration-color: transparent;  /* hidden by background */
}
```

Hover is a **full color-swap highlight**, not a color shift. Think of a marker stroke across the word.

### Code

Inline and block code are boxed with ink borders.

```css
code {
  font-family: var(--font-display);
  font-size: 0.9em;
  background: var(--paper-sunken);
  padding: 0.15em 0.4em;
  border: 1px solid var(--ink);
  border-radius: 0;           /* no rounding in Paper */
}

pre {
  font-family: var(--font-display);
  background: var(--paper-sunken);
  padding: 1rem 1.25rem;
  border: 2px solid var(--ink);
  box-shadow: 4px 4px 0 var(--ink);
  overflow-x: auto;
}
```

---

## 4. Component Stylings

### Button

Three variants. All buttons have the signature offset shadow.

**Base:**

```css
.btn {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1.25rem;

  font-family: var(--font-display);
  font-size: var(--text-base);
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  text-decoration: none;

  background: var(--paper-surface);
  color: var(--ink);
  border: 2px solid var(--ink);
  border-radius: 0;  /* square corners */
  box-shadow: 4px 4px 0 var(--ink);
  cursor: pointer;

  transition:
    transform 80ms linear,
    box-shadow 80ms linear;
}

.btn:hover {
  transform: translate(-2px, -2px);
  box-shadow: 6px 6px 0 var(--ink);
}

.btn:active {
  transform: translate(2px, 2px);
  box-shadow: 0 0 0 var(--ink);
}
```

The press state explicitly removes the shadow — the button visually "slams into the page".

**Primary:** Same shape, `background: var(--primary)`, `color: white`. Shadow stays `var(--ink)`.

**Danger:** `background: var(--error)`, `color: white`.

**Ghost:** `background: transparent`, everything else identical.

### Card

```css
.card {
  padding: 1.5rem;
  background: var(--paper-surface);
  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 6px 6px 0 var(--ink);

  transition:
    transform 100ms linear,
    box-shadow 100ms linear;
}

.card:hover {
  transform: translate(-2px, -2px);
  box-shadow: 8px 8px 0 var(--ink);
}

.card-title {
  font-family: var(--font-display);
  font-size: var(--text-h3);
  font-weight: 600;
  margin: 0 0 0.5rem;
}

.card-eyebrow {
  font-family: var(--font-display);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2px;
  color: var(--ink-muted);
  margin: 0 0 0.75rem;
}
```

### Input / Textarea

```css
input, textarea, select {
  width: 100%;
  padding: 0.75rem 1rem;

  font-family: var(--font-body);
  font-size: var(--text-base);
  color: var(--ink);
  background: var(--paper-surface);

  border: 2px solid var(--ink);
  border-radius: 0;
  box-shadow: 4px 4px 0 var(--ink);

  transition: box-shadow 80ms linear, transform 80ms linear;
}

input:focus {
  outline: none;
  transform: translate(-1px, -1px);
  box-shadow: 5px 5px 0 var(--primary);
  border-color: var(--primary);
}
```

Focus is the only place `--primary` enters the shadow stack.

### Badge / Tag

```css
.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.2rem 0.5rem;

  font-family: var(--font-display);
  font-size: var(--text-xs);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;

  background: var(--ink);
  color: var(--paper-bg);
  border: 2px solid var(--ink);
  border-radius: 0;
}

.badge--success { background: var(--success); border-color: var(--success); }
.badge--warning { background: var(--warning); border-color: var(--warning); color: var(--ink); }
.badge--error   { background: var(--error); border-color: var(--error); }
```

### Callout / Blockquote

Content set aside from body prose.

```css
.callout {
  padding: 1rem 1.25rem;
  margin: 1.5rem 0;
  background: var(--paper-sunken);
  border-left: 6px solid var(--ink);
  /* No drop shadow — callouts embed into the page */
}

.callout--warn { border-left-color: var(--warning); }
.callout--error { border-left-color: var(--error); }
```

### Navigation

Top bar, full-width, thick bottom rule.

```css
.nav {
  position: sticky;
  top: 0;
  padding: 1rem 1.5rem;

  background: var(--paper-bg);
  border-block-end: 3px solid var(--ink);
}

.nav-link {
  font-family: var(--font-display);
  font-size: var(--text-base);
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--ink);
  padding: 0.25rem 0;
  border-bottom: 2px solid transparent;
}

.nav-link:hover,
.nav-link[aria-current="page"] {
  border-bottom-color: var(--ink);
}
```

### Horizontal rule

```css
hr {
  border: none;
  border-block-start: 2px solid var(--ink);
  margin-block: 2rem;
}

hr.heavy { border-block-start-width: 4px; }
hr.double { border-block-start: 2px double var(--ink); }
```

### List — numbered squares

Ordered lists get filled square counters in the display font.

```css
ol { list-style: none; counter-reset: item; padding-left: 2.5rem; }
ol li { counter-increment: item; position: relative; margin-block-end: 0.75rem; }
ol li::before {
  content: counter(item, decimal-leading-zero);
  position: absolute;
  left: -2.5rem;
  top: 0;
  font-family: var(--font-display);
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--paper-bg);
  background: var(--ink);
  padding: 0.1rem 0.35rem;
  min-width: 1.75rem;
  text-align: center;
}
```

---

## 5. Layout Principles

### Spacing — shared 8pt grid

Same tokens as Glass (`--space-1` through `--space-12`). No additions, no overrides.

### Radius — zero

**There is no radius scale in Paper.** All corners are square. If a designer reaches for `border-radius`, it is a signal they are thinking in Glass terms and should stop.

The only exception is `--radius-full` for avatars and status dots, used rarely.

### Grid

- Content max-width: `72ch` for prose, `1100px` for landing layouts.
- Outer margin: minimum `var(--space-5)` on mobile, `var(--space-8)` on desktop.
- Column gutter: `var(--space-6)` — wider than Glass because shadows need room to extend without clipping.

### Shadow clearance rule

Any element with a `Npx Npx 0 var(--ink)` shadow needs `N px + 2px` of padding from its container edge. Otherwise the shadow clips. When in doubt: `margin-right: 8px; margin-bottom: 8px` on shadowed elements near edges.

---

## 6. Depth & Elevation

**The entire elevation vocabulary is offset distance.** No blur, no spread, no opacity.

| Level | Recipe | Use |
|---|---|---|
| Flat | None | Body text, inline elements, callouts |
| Paper | `4px 4px 0 var(--ink)` | Button, input, small card |
| Raised | `6px 6px 0 var(--ink)` | Card, panel |
| Floating | `8px 8px 0 var(--ink)` | Card hover, floating dialog |
| Hero | `12px 12px 0 var(--ink)` | Hero block, feature callout |

### Shadow color override

For emphasis, shadows may use `--primary` or `--error` instead of `--ink` — but only on *focus* or *destructive-confirm* states. Default shadow is always `--ink`.

### Why zero-blur

Blur implies ambient light and atmosphere. Paper rejects atmosphere. The shadow is a *physical displacement* of the same shape — as if the card floats above the page and a hard directional light casts behind it.

---

## 7. Do's and Don'ts

### Do

- **Use zero border-radius everywhere** except `--radius-full` for circular elements (avatars, dots).
- **Always give interactive elements an offset shadow.** A button without a `4px 4px 0` shadow is not a Paper button.
- **Use mono for headlines, sans for body.** Do not mix — no serif, no italic body.
- **Underline links always.** Never rely on color alone.
- **Use `border: 2px solid var(--ink)`** as the default border recipe. 1px borders read as weak.
- **Use uppercase with tracked letter-spacing** for buttons, nav links, badges. Tracking is 0.5px to 2px depending on size.
- **Let whitespace breathe.** Paper's density tolerance is ~60% of Glass.

### Don't

- **Don't use `backdrop-filter`.** Paper is opaque.
- **Don't use radial or linear gradients.** Solid fills only.
- **Don't soften shadows.** `0 4px 12px rgba(0,0,0,0.1)` is a Glass shadow in a Paper file — remove it.
- **Don't use the Glass `--glass-*` tokens.** They don't exist in Paper.
- **Don't animate beyond `transform` and `box-shadow`.** Hover = translate + shadow. That's it.
- **Don't use OKLCH dynamic surfaces (`color-mix()`) for UI chrome.** Paper surfaces are hand-picked paper tones, not derived.
- **Don't use `--text-main` / `--text-muted` / `--glass-bg` tokens here.** They are Glass-only. Use `--ink`, `--ink-muted`, `--paper-surface`.
- **Don't use `data-theme="dark"` to mean "deep void".** In Paper, dark mode is *carbon paper*, not *dashboard void*. Use the dedicated Paper Dark palette.

---

## 8. Responsive Behavior

### Breakpoints — same as Glass

```css
@media (max-width: 1024px) { }
@media (max-width: 767px)  { }
@media (max-width: 640px)  { }
```

### Mobile adjustments

- Display headline clamps down to a minimum of 28px (vs desktop max 48px).
- Body stays at 16px — never shrink body below 16px in Paper.
- Shadow offset reduces: `4px 4px` → `3px 3px` on cards to prevent edge clipping on narrow viewports.
- Content padding collapses from `var(--space-8)` to `var(--space-4)`.

### Touch targets

- Minimum 44×44px touch area on all interactive elements.
- Buttons already meet this at `0.75rem 1.25rem` padding with 16px body.
- Nav links get `padding: var(--space-3) var(--space-4)` on mobile to hit the minimum.

### Collapsing strategy

- Multi-column layouts stack vertically at 767px.
- Sticky nav stays sticky on mobile — the thick bottom rule provides enough separation without a shadow.

---

## 9. Agent Prompt Guide

### Quick color reference

```
Ink / text / border    → var(--ink)
Muted text             → var(--ink-muted)
Page                   → var(--paper-bg)
Card / surface         → var(--paper-surface)
Inset / input          → var(--paper-sunken)
Divider                → var(--rule)

Accent (link, primary) → var(--primary)
Success block          → var(--success)
Warning block          → var(--warning)
Error block            → var(--error)
```

### Ready-to-use prompts

**"Build a new card component in Paper style"**
> `background: var(--paper-surface)`, `border: 2px solid var(--ink)`, `border-radius: 0`, `box-shadow: 6px 6px 0 var(--ink)`, `padding: 1.5rem`. Hover: `transform: translate(-2px, -2px)`, `box-shadow: 8px 8px 0 var(--ink)`. Title uses `--font-display` weight 600. Never add a focus glow.

**"Build a landing page hero in Paper style"**
> Background `--paper-bg`. Display headline at `--text-display` in `--font-display` weight 600, color `--ink`. Below: one paragraph at `--text-base` max 65ch, then one primary `.btn` and one ghost `.btn` inline with a 12px gap. No images unless they have a 2px `--ink` border with a 4px offset shadow.

**"Convert a Glass component to Paper"**
> Strip all `var(--glass-*)` tokens and `backdrop-filter` rules. Replace `border-radius: var(--radius-lg)` with `border-radius: 0`. Replace `box-shadow: var(--highlight-inset), var(--shadow-micro)` with `box-shadow: 4px 4px 0 var(--ink)`. Replace text tokens: `--text-main` → `--ink`, `--text-muted` → `--ink-muted`. Change background to `--paper-surface`.

**"Style a form field in Paper"**
> `background: var(--paper-surface)`, `border: 2px solid var(--ink)`, `border-radius: 0`, `box-shadow: 4px 4px 0 var(--ink)`, `padding: 0.75rem 1rem`, font `--font-body` at `--text-base`. Focus shifts shadow color to `--primary` and adds `translate(-1px, -1px)`.

### The three questions before adding Paper CSS

1. **Is every border at least 2px solid?** If 1px, reconsider.
2. **Does every interactive surface have a `Npx Npx 0 var(--ink)` shadow?** If not, it's not Paper.
3. **Is every radius zero (except avatars)?** If rounded, you're drifting to Glass.

---

*Paper shares color primitives with Glass and Terminal, but the treatment is distinct. When ambiguous: Paper has borders and offset shadows; Glass has blur and micro-elevation; Terminal has neither.*
