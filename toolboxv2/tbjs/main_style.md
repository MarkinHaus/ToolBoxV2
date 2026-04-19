# TBJS Glass — SimpleCore Main Style

> **System:** TBJS Design System v3.0 — Deep Tech Unified Style
> **Scope:** Primary visual language for `simplecore.app` and all public-facing ToolBoxV2 surfaces.
> **Philosophy:** *Quiet UI* — precise, discrete, minimally invasive. Colors live through `color-mix()`, chrome stays silent.
> **Format:** Stitch `DESIGN.md` compatible — drop into project root, reference from coding agents.

---

## 1. Visual Theme & Atmosphere

**Mood.** Deep-tech dashboard meets scientific instrument. A dark void base with a 3D-rastered background bleeding colored light through a micro-glass layer. Content floats over the scene; it does not sit on a page.

**Density.** High information density without visual noise. Body text at **13px**, `h1` capped at **22px** — this is an *instrument panel*, not a marketing page. Fluid typography only scales within a narrow range; nothing grows into hero-sized display text.

**Design principles**

- **Dynamic over static.** Colors are authored as **raw OKLCH tuples** (`L C H` without the `oklch()` wrapper) and consumed via `color-mix(in oklch, ...)`. One hue change propagates across the whole system. Never hardcode hex values in components.
- **Subtract, don't add.** No loud shadows, no focus glows, no decorative gradients. Elevation is a 1px inset highlight plus a single soft drop shadow. That is the entire elevation vocabulary.
- **Data-first typography.** Uppercase tracked micro-labels in mono, body in sans. Mono appears when content *is* data (metrics, IDs, paths, code). Sans for prose and chrome.
- **Transparent chrome.** Navigation, cards, modals, widgets all sit on `rgba(255,255,255,0.02)` glass with `blur(12px)`. The 3D scene behind shows through — the UI is a HUD, not a page.
- **Two moods, not two themes.** Dark (default) is deep void blue-black. Light is "Water Machine" — cyan-blue-green, consciously inverted. There is no neutral white mode.
- **Grid over table.** Tabular data uses `display: grid` with `grid-template-columns`, never `<table>`. Rows are `display: contents` so cells flow directly into the grid.

---

## 2. Color Palette & Roles

All colors are authored as **raw OKLCH tuples** so they can be consumed by `color-mix()` for dynamic surfaces. The `oklch()` wrapper is applied once to produce the base color; all surface variants are derived at use-site.

### Raw primitive tokens

| Token | Raw OKLCH | Role |
|---|---|---|
| `--raw-primary` | `55% 0.18 230` | Deep-tech blue — interactive accent, links, focus, active states |
| `--raw-success` | `65% 0.2 145` | Positive status, confirmation |
| `--raw-warning` | `75% 0.18 85` | Caution, pending |
| `--raw-error` | `55% 0.22 25` | Destructive, failure |
| `--raw-info` | `60% 0.15 230` | Neutral notification |

### Resolved semantic colors

| Token | Derivation | Use |
|---|---|---|
| `--primary` | `oklch(var(--raw-primary))` | All interactive accent |
| `--success` | `oklch(var(--raw-success))` | Status dots, success toasts |
| `--warning` | `oklch(var(--raw-warning))` | Pending indicators |
| `--error` | `oklch(var(--raw-error))` | Error toasts, destructive actions |
| `--info` | `oklch(var(--raw-info))` | Informational |

### Dark theme surfaces (default)

| Token | Value | Use |
|---|---|---|
| `--bg-base` | `#08080d` | Body void — canvas behind everything |
| `--bg-surface` | `rgba(10, 10, 18, 0.8)` | Raised container with slight opacity |
| `--bg-elevated` | `rgba(15, 15, 25, 0.9)` | Modals, dropdowns, popovers |
| `--bg-sunken` | `rgba(0, 0, 0, 0.3)` | Input fields, insets |
| `--glass-bg` | `rgba(255, 255, 255, 0.02)` | Primary chrome fill — nav, card, widget, modal |
| `--glass-border` | `rgba(255, 255, 255, 0.05)` | Chrome border — always paired with glass-bg |
| `--glass-blur` | `12px` | Applied via `backdrop-filter: blur(var(--glass-blur))` |
| `--border-subtle` | `rgba(255, 255, 255, 0.08)` | Dividers, separators, subtle rules |

### Dark theme text

| Token | Value | Use |
|---|---|---|
| `--text-main` | `rgba(255, 255, 255, 0.85)` | Body text, headings |
| `--text-label` | `rgba(255, 255, 255, 0.4)` | Form labels, micro-labels |
| `--text-muted` | `rgba(255, 255, 255, 0.25)` | De-emphasized, captions, hints |

Paragraph body specifically drops to `rgba(255, 255, 255, 0.7)` — slightly dimmer than `--text-main` for long-form readability.

### Light theme — "Water Machine"

Not a neutral light mode. Cyan-blue-green surfaces, consciously inverted.

| Token | Value |
|---|---|
| `--bg-base` | `#d8e8ec` (blue-green wash) |
| `--bg-surface` | `rgba(220, 235, 242, 0.85)` |
| `--bg-elevated` | `rgba(230, 242, 248, 0.92)` |
| `--glass-bg` | `rgba(215, 235, 245, 0.7)` |
| `--glass-border` | `rgba(100, 150, 170, 0.15)` |
| `--text-main` | `#0a1a1f` |
| `--text-label` | `#4a5a60` |
| `--text-muted` | `#7a8a90` |
| `--border-subtle` | `rgba(100, 150, 170, 0.2)` |

### Dynamic surfaces (via `color-mix`)

Never define a new hardcoded surface color. Derive from `--primary`:

```css
--surface-badge:  color-mix(in oklch, var(--primary) 15%, transparent);
--surface-hover:  color-mix(in oklch, var(--primary)  5%, transparent);
--surface-active: color-mix(in oklch, var(--primary) 10%, transparent);
--border-active:  color-mix(in oklch, var(--primary) 30%, transparent);
```

Light mode uses `in srgb` instead of `in oklch` with a tinted base color instead of `transparent`:

```css
--surface-badge: color-mix(in srgb, var(--primary) 12%, #d0e5f0);
```

### 3D scene tokens

The rastered WebGL background reads two variables to stay theme-consistent:

| Token | Dark | Light | Role |
|---|---|---|---|
| `--theme-bg-sun` | `#404060` | `#d0f0ff` | Point light color |
| `--theme-bg-light` | `#181823` | `#40a0c0` | Ambient light color |

---

## 3. Typography Rules

### Font families — IBM Plex dual system

```css
--font-sans: 'IBM Plex Sans', system-ui, -apple-system, sans-serif;
--font-mono: 'IBM Plex Mono', ui-monospace, 'SF Mono', Consolas, monospace;
```

**Rule:** Mono is reserved for content that *is* data — metrics, IDs, file paths, code, timestamps, micro-labels. Everything else uses sans. Never use mono for decorative effect.

### Type scale — capped fluid

Unusually capped at 22px max. This is intentional — the system is a dashboard, not marketing.

| Token | Value | Role |
|---|---|---|
| `--text-h1` | `clamp(18px, 2vw, 22px)` | Page title — bold (700) |
| `--text-h2` | `clamp(16px, 1.8vw, 19px)` | Section heading — bold (700) |
| `--text-h3` | `clamp(14px, 1.5vw, 16px)` | Subsection — semibold (600) |
| `--text-base` | `13px` | Body default |
| `--text-sm` | `11px` | Secondary, captions, labels |
| `--text-xs` | `9px` | Micro-labels — always mono, uppercase, tracked |

### Heading treatment

| Property | Value |
|---|---|
| `font-weight` | 700 (h1, h2), 600 (h3–h6) |
| `line-height` | 1.2 |
| `letter-spacing` | `-0.02em` |
| `text-wrap` | `balance` |
| `margin-block-end` | `var(--space-3)` |
| `color` | `var(--text-main)` |

### Body paragraph

| Property | Value |
|---|---|
| `line-height` | 1.6 |
| `max-inline-size` | `65ch` |
| `color` | `rgba(255, 255, 255, 0.7)` in dark |
| `margin-block-end` | `var(--space-4)` |

### Micro-label — the signature mark

The `.label` utility and `<h6>` share this treatment. Use anywhere a field needs a category marker.

```css
.label {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  text-transform: uppercase;
  letter-spacing: 2.5px;
  color: var(--text-label);
  user-select: none;
}
```

Form `<label>` is similar but less extreme: `text-sm`, weight 500, uppercase, `letter-spacing: 1px`.

### Inline code and pre blocks

```css
code {
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  background: var(--input-bg);
  padding: 0.2em 0.4em;
  border-radius: var(--radius-sm);
  color: var(--primary);  /* accent color, not main text */
}

pre {
  background: var(--input-bg);
  padding: var(--space-4);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-subtle);
  overflow-x: auto;
}
```

Inline code uses `--primary` for color — ties code fragments to the interactive accent. Pre blocks do not; they inherit body text.

---

## 4. Component Stylings

### Button

Three variants. No loud gradients, no colored glow — Quiet UI means hover is `translateY(-1px)` and nothing else animates.

**Base** (shared by all):

```css
button, .btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-2);
  padding: var(--space-3) var(--space-5);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  font-weight: 500;
  line-height: 1.2;
  border: none;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition:
    transform var(--duration-fast) var(--ease-default),
    background-color var(--duration-fast) var(--ease-default),
    border-color var(--duration-fast) var(--ease-default);
}
```

**Primary** — solid primary on black text. Also applies to `button[type="submit"]`.

```css
.btn-primary {
  color: #000;
  background: var(--primary);
  box-shadow: var(--highlight-inset);
}
.btn-primary:hover {
  background: color-mix(in oklch, var(--primary) 90%, white);
  transform: translateY(-1px);
}
.btn-primary:active {
  transform: translateY(0);
  box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.2);
}
```

**Secondary** — glass fill, subtle border.

```css
.btn-secondary {
  color: var(--text-main);
  background: var(--glass-bg);
  border: 1px solid var(--border-subtle);
  box-shadow: var(--highlight-inset);
}
.btn-secondary:hover {
  background: var(--surface-hover);
  border-color: var(--border-active);
  transform: translateY(-1px);
}
```

**Ghost** — transparent until hover.

```css
.btn-ghost {
  color: var(--text-main);
  background: transparent;
}
.btn-ghost:hover {
  background: var(--surface-hover);
}
```

**Focus** — sharp outline, no glow:

```css
button:focus-visible { outline: 1px solid var(--primary); outline-offset: 2px; }
```

**Disabled** — `opacity: 0.4`, no transform.

### Card

```css
.card {
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  padding: var(--space-5);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
  transition:
    transform var(--duration-normal) var(--ease-default),
    box-shadow var(--duration-normal) var(--ease-default);
}
.card:hover {
  transform: translateY(-1px);
  box-shadow: var(--highlight-inset), 0 4px 8px rgba(0, 0, 0, 0.6);
}
```

`.card-title` — `text-base`, weight 600. `.card-content` — `text-sm`, `rgba(255,255,255,0.6)`, line-height 1.5.

### Input / Textarea / Select

```css
input, textarea, select {
  width: 100%;
  padding: var(--space-3) var(--space-4);
  font-family: var(--font-sans);
  font-size: var(--text-base);
  color: var(--text-main);
  background: var(--input-bg);
  border: 1px solid var(--input-border);
  border-radius: var(--radius-md);
  margin-block-end: var(--space-4);
}
input:focus {
  outline: none;
  border-color: var(--input-focus);  /* = --primary */
  background: color-mix(in oklch, var(--input-focus) 5%, var(--input-bg));
}
```

**No focus glow.** The only focus signal is the border color flip plus a 5% tint on the background.

Select gets a custom chevron via data-URI SVG with stroke `%23ffffff40` (matches `--text-muted`).

### Radio / Checkbox

- `18×18px`, `appearance: none`, 2px `--input-border`.
- Radio: `border-radius: var(--radius-full)`.
- Checkbox: `border-radius: var(--radius-sm)`.
- Checked state uses `--primary` fill with an inset ring (3px glow via `--surface-badge`). Light mode uses a thicker white inset. Checkbox gets an SVG checkmark baked in.

### Navigation

Floating glass pill, top-left, fixed position. Width fits children.

```css
.nav-controls {
  position: fixed;
  top: var(--space-4);
  left: var(--space-4);
  z-index: var(--z-sticky);
  display: flex;
  gap: var(--space-3);
  padding: var(--space-2);
  background: var(--glass-bg);
  backdrop-filter: blur(calc(var(--glass-blur) * 0.8));
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--highlight-inset), var(--shadow-micro);
}

.nav-toggle {
  width: 36px;
  height: 36px;
  background: transparent;
  border-radius: var(--radius-full);
  color: var(--text-muted);
}
.nav-toggle:hover {
  background: var(--surface-hover);
  color: var(--text-main);
}
```

Dropdown appears via `.is-open` class — opacity + visibility + `translateY(0)` transition. Never display:none toggling, always animate.

### Tab navigation

```css
.tab-nav {
  display: flex;
  gap: var(--space-1);
  border-bottom: 1px solid var(--border-subtle);
}
.tab-item {
  padding: var(--space-3) var(--space-4);
  font-size: var(--text-sm);
  color: var(--text-muted);
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;  /* overlap the nav border */
}
.tab-item.is-active {
  color: var(--primary);
  border-bottom-color: var(--primary);
  background: var(--surface-active);
}
```

### Modal

```css
.modal {
  position: fixed;
  top: 50%; left: 50%;
  transform: translate(-50%, -50%) scale(0.96);
  width: 90%;
  max-width: 500px;
  max-height: 90vh;
  padding: var(--space-6);
  background: var(--glass-bg);
  backdrop-filter: blur(var(--glass-blur));
  border: 1px solid var(--glass-border);
  border-radius: var(--radius-lg);
  box-shadow: var(--highlight-inset), 0 10px 30px rgba(0, 0, 0, 0.7);
  opacity: 0;
  visibility: hidden;
}
.modal.is-active {
  opacity: 1;
  visibility: visible;
  transform: translate(-50%, -50%) scale(1);
}
```

Modal drop shadow is **harder** than card drop shadow (`rgba(0,0,0,0.7)` vs micro) — modals sit above the overlay so they need more separation.

Overlay: `position: fixed; inset: 0; background: rgba(0,0,0,0.7); backdrop-filter: blur(4px);` — half the glass blur to keep content behind legible as context.

### Widget

Resizable floating glass panel. `padding-block-start: var(--space-8)` leaves room for the absolute-positioned header.

- Header: absolutely positioned top `--space-2`, contains mono uppercase title (`--text-xs`, tracked 1.5px, `--text-label`) and action buttons (20×20px circle).
- Widget enter animation: `translateY(8px) scale(0.98)` → `0,1` over `--duration-normal`.

### Grid table

No HTML `<table>`. `.grid-table` with `display: grid`. `.grid-row` uses `display: contents` so `.grid-cell` children flow into the parent grid.

```css
.grid-header .grid-cell {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--text-label);
  text-transform: uppercase;
  letter-spacing: 1px;
  border-bottom: 1px solid var(--border-subtle);
}
.grid-row:hover .grid-cell {
  background: var(--surface-hover);
  border-bottom-color: var(--border-active);
  transform: translateY(-1px);
}
```

### LED score bars

Signature component — visualizes quantized values as lit segments.

- `.score-block`: `12×5px`, `1px` radius, default `--border-subtle`.
- `.is-active`: `--primary` with `box-shadow: 0 0 4px var(--surface-badge)` — the *only* glow in the system, because it reads as LED emission.
- Semantic variants: `.is-success`, `.is-warning`, `.is-error`.

### Chat (glass panel)

Glass widget with header / messages / input areas. Max width 600px, max height 70vh.

- Container: `--glass-bg` + `--glass-border` + `--radius-lg` + `--shadow-micro`.
- User message: `background: var(--primary)`, black text, `align-self: flex-end`.
- Other message: `background: var(--input-bg)`, `--border-subtle`, `align-self: flex-start`.
- System message: italic, centered, `--text-muted`, no background.
- Status dot: 10×10px circle. `--warning` default, `--success` when `.connected`, `--error` when `.error`.

### Toast

Slim floating panels — top-right default. Container `max-width: 400px`.

- Fill: `--glass-bg` + `blur(var(--glass-blur))` + `--glass-border` + `--radius-lg`.
- Shadow: `--highlight-inset, var(--shadow-micro)` plus a softer outer `0 8px 24px rgba(0,0,0,0.25)` for elevation above content.
- Header row: icon + uppercase title (weight 600, 13px, letter-spacing 0.5px, colored via `--toast-color` CSS var set inline).
- Message: `--text-main` at 0.9 opacity, 14px, line-height 1.5.
- Action button: `background: var(--toast-color)`, white text, 6px radius.
- Progress bar: 2px, `--toast-color`, left-origin `scaleX` transition.
- Dot indicator (collapsed toast): 8×8px circle colored via `--toast-color`.
- Enter: `translateY(-20px) scale(0.95)` → `0,1` over 250ms cubic-bezier.

`--toast-color` is set per-instance (JS) to `--success`, `--warning`, `--error`, or `--info`.

### Cookie banner

Bottom-fixed glass bar, not a modal.

- `background: var(--bg-base)` with top border `1px solid var(--border-subtle)`.
- Inner container: `max-width: 1024px`, centered, flex wrap with `gap: var(--space-2) var(--space-4)`.
- Close button: absolutely positioned top-right, `--text-muted`, hover → `--text-main`.
- Dismiss button: `.btn-secondary` base.
- Enter: `translateY(100%)` + `opacity: 0` → `0, 1` over 300ms.

### Status bar (desktop platform)

24px height, bottom-fixed, mono font. Reads like a tmux status line.

- `background: var(--bg-surface)`, top border `--border-subtle`.
- `font-size: 11px`, `--text-muted`.
- Status dot: 8×8px, semantic colors (`--success` = online, `--error` = offline, `--warning`, `--text-muted` for disabled).

### Bottom nav (mobile platform)

60px height, bottom-fixed with safe-area inset.

- Same glass treatment as status bar.
- Active item: `color: var(--primary)`. Inactive: `--text-muted`.
- Icon 20px, label 10px weight 500.

### Quick capture popup

Centered modal-lite, 400px wide. Same glass treatment as `.modal` but simpler enter (scale 0.9 → 1, no full overlay).

---

## 5. Layout Principles

### Spacing scale — 8pt grid

```
--space-1: 0.25rem   (4px)
--space-2: 0.5rem    (8px)
--space-3: 0.75rem   (12px)
--space-4: 1rem      (16px)
--space-5: 1.5rem    (24px)
--space-6: 2rem      (32px)
--space-8: 3rem      (48px)
--space-10: 4rem     (64px)
--space-12: 6rem     (96px)
```

No `--space-7`, `--space-9`, `--space-11` — missing steps are intentional. If a gap seems to need one, reconsider whether two adjacent tokens are used inconsistently.

### Radius scale — sharper than typical

Tech-look demands tighter corners.

```
--radius-sm:   2px
--radius-md:   6px
--radius-lg:  12px
--radius-xl:  18px
--radius-full: 9999px
```

Buttons and inputs: `--radius-md`. Cards, glass, nav, modal, widget, toast: `--radius-lg`. Chips, checkboxes, code tags: `--radius-sm`. Circles (nav toggles, dots): `--radius-full`.

### Grid and whitespace

- Main content max-width: `95vw`, centered with glass treatment.
- Content wrapper padding: `var(--space-8) var(--space-4)`, top `var(--space-12)` to clear fixed nav.
- Paragraph max inline size: `65ch`.

### Z-index stack

```
--z-behind:    -1
--z-base:       0
--z-raised:    10
--z-dropdown: 100
--z-sticky:   500
--z-overlay: 1000
--z-modal:  10000
--z-toast:  10100
```

Toasts sit *above* modals — intentional, because a toast may announce modal state change.

---

## 6. Depth & Elevation

**This system has only two elevation primitives.** Compose them.

```css
--highlight-inset: inset 0 1px 0 rgba(255, 255, 255, 0.05);
--shadow-micro:    0 2px 4px rgba(0, 0, 0, 0.5);
```

In light mode, both shift:

```css
--highlight-inset: inset 0 1px 0 rgba(255, 255, 255, 0.6);
--shadow-micro:    0 2px 8px rgba(50, 100, 120, 0.15);  /* blue-tinted */
```

### Elevation vocabulary

| Level | Recipe | Use |
|---|---|---|
| Flat | None | Body, inline elements |
| Resting | `var(--highlight-inset)` only | Primary button (subtle top-lip) |
| Raised | `var(--highlight-inset), var(--shadow-micro)` | Card, glass, nav, widget, toast, tab |
| Floating | `var(--highlight-inset), 0 4px 8px rgba(0,0,0,0.6)` | Card hover |
| Modal | `var(--highlight-inset), 0 10px 30px rgba(0,0,0,0.7)` | Modal only |

**Never invent a new shadow.** If a new elevation is needed, use one of these five.

### Why inset highlight matters

The 1px inset top highlight simulates a light source from above hitting the edge of the surface. On a dark body with glass surfaces, without it the chrome looks pasted on. With it, the chrome reads as a physical layer catching light.

---

## 7. Do's and Don'ts

### Do

- **Derive surfaces from `--primary` via `color-mix`.** When a new surface color is needed, extend the `--surface-*` pattern. Never introduce a hardcoded hex.
- **Use `[data-theme="dark"]` / `[data-theme="light"]`** on `:root` to switch themes. Not `.dark` or media-query-only.
- **Prefix component classes consistently.** Global tokens are `--token-name`. Component classes follow the pattern in `tbjs-main.css` — `.card`, `.btn-*`, `.nav-*`, `.widget-*`.
- **Use mono for data, sans for chrome.** Timestamps, IDs, file paths, numbers → mono. Buttons, nav, headings → sans.
- **Animate `opacity` + `visibility` + `transform` together** for entry/exit — never `display: none` toggles.
- **Use `translateY(-1px)` for hover**, nothing more dramatic.
- **Cap modal `max-height: 90vh`** and make the inner area scroll. Never let a modal exceed the viewport.

### Don't

- **Don't use Tailwind-era tokens.** Anything starting with `--tb-*` (e.g. `--tb-bg-secondary`, `--tb-primary`, `--tb-border`), `--theme-*` (e.g. `--theme-border`, `--theme-primary`), or `--color-primary-300`, `--text-color`, `--glass-shadow`, `--transition-fast`, `--font-size-*`, `--font-weight-*`, `--input-focus-border` is legacy. Map it to the v3.0 token.
- **Don't introduce focus glow.** The focus signal is `outline: 1px solid var(--primary); outline-offset: 2px` — no `box-shadow` glow, no halo.
- **Don't use `<table>` for data.** Use `.grid-table` with `display: grid` + `display: contents` rows.
- **Don't hardcode status colors** (`#22c55e`, `#ef4444`, `#f59e0b`). Use `--success`, `--error`, `--warning`.
- **Don't use `.dark` as selector.** Always `[data-theme="dark"]`.
- **Don't grow typography past 22px.** Even hero titles stay at `--text-h1`. Density is the brand.
- **Don't stack gradients.** Solid colors only, except the LED `box-shadow: 0 0 4px var(--surface-badge)` on `.score-block.is-active`, which is the single permitted glow.
- **Don't wrap `rgba()` in another `rgba()`.** Expressions like `rgba(var(--text-main), 0.1)` are broken — `--text-main` is already `rgba(...)`. Use `color-mix()` instead.
- **Don't nest selectors like `:root[data-theme="dark"] { .tb-toast { ... } }`.** CSS nesting without a fallback build step is unsafe in this project.
- **Don't target DOM structure** like `> div:first-child`. Components get explicit classes (`.modal-header`, `.modal-body`, `.modal-footer`).

---

## 8. Responsive Behavior

### Breakpoints

```css
@media (max-width: 1024px) { /* tablet / narrow desktop */ }
@media (max-width: 767px)  { /* mobile */  }
@media (max-width: 640px)  { /* small mobile — toast adjustment only */ }
```

### Mobile adjustments (≤767px)

- `.content-wrapper` padding collapses: `var(--space-5) var(--space-3)`.
- `.main-content` loses border-radius to `--radius-md` and fills width.
- Headings shrink by ~1-2px absolute (h1: 19px, h2: 17px, h3: 15px).
- Buttons stretch to `width: 100%` with `max-width: 280px`.
- `.nav-controls` moves to `top/left: var(--space-2)`, reduced padding.
- Safe-area insets: `body.platform-mobile` adds `padding-bottom: calc(64px + env(safe-area-inset-bottom))`, header gets `padding-top: env(safe-area-inset-top)`.

### Touch targets

- Nav toggle, widget button: 36px and 20px. Widget button is internal — acceptable. Public nav is 36px.
- Toast close: 4px padding on a 16px icon = 24px target. **Too small for touch** — lifted to 32px on mobile.
- Bottom nav items: 60px height, full tap area.

### Collapsing strategy

- Glass stays glass at all widths. It does not flatten to opaque on mobile.
- Grid tables: columns become single-column stacked via JS grid-template-columns override, not CSS alone.
- Modal widths use `width: 90%` with `max-width: 500px` — always usable on mobile.

---

## 9. Agent Prompt Guide

### Quick color reference

When asked for a color role, use these tokens. Do not guess hex values.

```
Accent / interactive  → var(--primary)
Success               → var(--success)
Warning               → var(--warning)
Error                 → var(--error)
Info                  → var(--info)

Body text             → var(--text-main)
Label text            → var(--text-label)
Dim text              → var(--text-muted)

Canvas                → var(--bg-base)
Raised container      → var(--glass-bg) + var(--glass-border) + blur(var(--glass-blur))
Input field           → var(--input-bg)
Hover tint            → var(--surface-hover)
Active tint           → var(--surface-active)
Divider               → var(--border-subtle)
```

### Ready-to-use prompts

**"Build a new card component"**
> Use `.card` as the base. Glass background (`--glass-bg` + `--glass-border`), `--radius-lg`, `--space-5` padding, elevation is `var(--highlight-inset), var(--shadow-micro)`. Hover lifts 1px and deepens shadow. Title uses `--text-base` weight 600. Content uses `--text-sm` at `rgba(255,255,255,0.6)`.

**"Add a new status color"**
> Extend the `--raw-*` tokens. Add `--raw-newstatus: L% C H;` at `:root`, then `--newstatus: oklch(var(--raw-newstatus));`. Never add a hardcoded hex status color.

**"Style a new modal"**
> Fixed centered. Start at `translate(-50%, -50%) scale(0.96)` + `opacity: 0`. `.is-active` class swaps to `scale(1)` + `opacity: 1`. Glass background, `--radius-lg`, hard shadow `0 10px 30px rgba(0,0,0,0.7)`. Backdrop uses `.overlay` with `blur(4px)`.

**"Make it respond to theme switches"**
> Read from tokens only. If the component needs a value not yet tokenized, add it to both `[data-theme="dark"]` and `[data-theme="light"]` in `tbjs-main.css` — don't hardcode a mode-specific value in the component file.

**"Match the existing dashboard density"**
> Body stays at 13px. Headings cap at 22px. Spacing uses the 8pt scale only (`--space-1` through `--space-12`, skipping 7/9/11). Never use arbitrary pixel spacing.

### Legacy → v3.0 migration map

Agents encountering old code should translate:

```
--tb-bg-secondary, --tb-bg          → --bg-elevated, --bg-base
--tb-border, --theme-border         → --border-subtle
--tb-primary, --theme-primary       → --primary
--tb-text, --theme-text             → --text-main
--tb-text-secondary, --theme-text-muted → --text-muted
--text-color                        → --text-main
--glass-shadow                      → --shadow-micro
--transition-fast                   → --duration-fast
--font-size-sm / -base              → --text-sm / --text-base
--font-weight-semibold              → 600 (literal)
--input-focus-border                → --input-focus
--color-primary-300                 → color-mix(in oklch, var(--primary) 30%, transparent)
Hardcoded #22c55e / #ef4444 / #f59e0b → --success / --error / --warning
.dark selector                      → [data-theme="dark"]
```

### The three questions before adding CSS

1. **Does this token already exist?** Check `tbjs-main.css` sections 1 and 1.5 first.
2. **Can this surface be derived via `color-mix()`?** If yes, derive. Don't invent.
3. **Am I adding a new shadow?** If yes, stop. Use one of the five elevation recipes in §6.

---

*This document is the contract between design and code for the TBJS Glass style. It supersedes all `--tb-*`, `--theme-*`, and Tailwind-era conventions previously present in component CSS files.*
