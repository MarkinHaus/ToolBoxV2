"""
Agent tools for the icli_web interactive panel system.

Exposes four tools to the agent when running under icli_web:

    show_interactive_panel(template, content, state?, panel_id?)
        Render a panel in the orb. Returns {ok, panel_id} or {ok: False, error}.

    get_interactive_panel_state()
        Read current user-edited values from the orb. Returns {panels: {...}}.

    list_panel_templates(category?)
        List built-in templates the agent can use out-of-the-box.

    validate_panel_spec(template, content)
        Check a (template, content) pair without rendering. Returns
        {ok, errors, warnings}. Use this BEFORE show_interactive_panel
        to avoid rendering broken panels.

The core templates: "form", "choice", "info", "html", "markdown", "file".
See orb.html:renderInteractivePanel for the orb-side renderer. If you add
a new built-in template here, add matching rendering in orb.html too.

Registration (once per agent, usually in ICli.__init__):

    from toolboxv2.mods.icli_web.client import IcliWebClient
    from toolboxv2.mods.icli_web._icli_web_tool import register_icli_web_tools

    IcliWebClient.get().attach(self)

    async def _reg():
        for name in ("self", "isaa", "coder"):
            try:
                agent = await self.isaa_tools.get_agent(name)
                register_icli_web_tools(agent)
            except Exception:
                pass
    asyncio.get_event_loop().create_task(_reg())
"""
from __future__ import annotations

from typing import Any, Optional

import json
import logging
import os
from pathlib import Path

from toolboxv2.mods.icli_web.client import IcliWebClient

log = logging.getLogger("icli_web.tools")


# ─── Built-in templates library ─────────────────────────────────────────────
#
# Each entry is a (template, content) preset. Call-site can merge its own
# state / panel_id in on top. Keys are short stable identifiers the agent
# learns and can refer to directly: list_panel_templates() returns them.

_TEMPLATES: dict[str, dict] = {
    # ── Forms ────────────────────────────────────────────────────────────
    "feedback": {
        "template": "form",
        "category": "forms",
        "description": "Collect user feedback with rating + comment.",
        "content": {
            "title": "Feedback",
            "fields": [
                {"name": "rating", "label": "Rating", "type": "select",
                 "options": ["1", "2", "3", "4", "5"], "value": "5"},
                {"name": "comment", "label": "Comment", "type": "textarea"},
                {"name": "contact_ok", "label": "Okay to contact me",
                 "type": "checkbox", "value": False},
            ],
            "submit_label": "Submit feedback",
            "submit_query": "User submitted feedback panel.",
        },
    },
    "bug_report": {
        "template": "form",
        "category": "forms",
        "description": "Collect enough info to file a bug.",
        "content": {
            "title": "Bug report",
            "fields": [
                {"name": "summary", "label": "Summary", "type": "text"},
                {"name": "steps", "label": "Steps to reproduce",
                 "type": "textarea"},
                {"name": "expected", "label": "Expected", "type": "text"},
                {"name": "actual", "label": "Actual", "type": "text"},
                {"name": "severity", "label": "Severity", "type": "select",
                 "options": ["low", "medium", "high", "critical"],
                 "value": "medium"},
            ],
            "submit_label": "File bug",
            "submit_query": "User filed a bug report.",
        },
    },
    "settings_quick": {
        "template": "form",
        "category": "forms",
        "description": "A handful of common settings at once.",
        "content": {
            "title": "Quick settings",
            "fields": [
                {"name": "language", "label": "Language", "type": "select",
                 "options": ["de", "en", "fr", "es"]},
                {"name": "notifications", "label": "Notifications",
                 "type": "checkbox", "value": True},
                {"name": "theme", "label": "Theme", "type": "select",
                 "options": ["auto", "light", "dark"]},
            ],
            "submit_label": "Apply",
            "submit_query": "User updated quick settings.",
        },
    },

    # ── Choices ──────────────────────────────────────────────────────────
    "confirm": {
        "template": "choice",
        "category": "choices",
        "description": "Binary yes/no confirmation.",
        "content": {
            "title": "Confirm",
            "prompt": "Are you sure?",
            "options": [
                {"value": "yes", "label": "Yes"},
                {"value": "no", "label": "Cancel"},
            ],
            "on_choice_query": "User confirmed: {choice}",
        },
    },
    "pick_one": {
        "template": "choice",
        "category": "choices",
        "description": "Multiple-choice picker. Caller should fill options.",
        "content": {
            "title": "Pick one",
            "prompt": "Choose an option:",
            "options": [
                {"value": "a", "label": "Option A"},
                {"value": "b", "label": "Option B"},
                {"value": "c", "label": "Option C"},
            ],
            "on_choice_query": "User picked: {choice}",
        },
    },

    # ── Info ─────────────────────────────────────────────────────────────
    "info_card": {
        "template": "info",
        "category": "info",
        "description": "A titled information card. Caller fills body.",
        "content": {
            "title": "Info",
            "body": "Replace this text with the content you want shown.",
        },
    },

    # ── HTML (trusted) ───────────────────────────────────────────────────
    "html_panel": {
        "template": "html",
        "category": "info",
        "description": (
            "Arbitrary HTML. SECURITY: set trusted=True only if the HTML "
            "came from agent-generated code, never from untrusted sources."
        ),
        "content": {
            "title": "HTML",
            "trusted": False,
            "html": "<p>Replace this with your HTML content.</p>",
        },
    },

    # ── Markdown (safe, escaped-first renderer) ──────────────────────────
    "markdown_doc": {
        "template": "markdown",
        "category": "info",
        "description": (
            "Render a markdown string as formatted HTML. The orb escapes "
            "all HTML first, then applies markdown patterns — safe for "
            "untrusted markdown. Supports headers, bold/italic, inline + "
            "fenced code, links, lists, blockquotes, paragraphs."
        ),
        "content": {
            "title": "Document",
            "md": (
                "# Hello\n\n"
                "Replace this with **your** markdown content.\n\n"
                "- Lists work\n"
                "- `inline code` too\n\n"
                "```python\nprint('fenced code blocks too')\n```\n"
            ),
        },
    },

    # ── File upload (metadata-only; bytes go to agent VFS) ───────────────
    "file_upload": {
        "template": "file",
        "category": "input",
        "description": (
            "Drag-and-drop / click-to-pick file upload. The orb streams "
            "file bytes into the agent's session VFS under `vfs_target`; "
            "the panel state exposes only metadata per file "
            "({name, size, type, vfs_path}). Caps at max_size_mb per file "
            "(default 10, hard ceiling 100). Use `accept` to restrict "
            "MIME types and `multiple=True` for batch uploads."
        ),
        "content": {
            "title": "Upload file",
            "prompt": "Drop a file here or click to pick one.",
            "max_size_mb": 10,
            "multiple": False,
            "vfs_target": "uploads/",
            # "accept": "image/*",   # uncomment to filter by MIME
            "submit_label": "Use this file",
            "submit_query": "User uploaded a file.",
        },
    },
}



# ─── ENV-loaded user templates ──────────────────────────────────────────────
#
# A user can extend the library by dropping JSON files into a directory
# pointed at by ICLI_WEB_TEMPLATES_DIR (falls back to ~/.toolbox/icli_web/
# templates). Each file looks like:
#
#   { "my_key": {
#       "template": "form",
#       "category": "my-cat",        (optional)
#       "description": "...",        (optional)
#       "content": {...}             (required)
#     },
#     "another_key": { ... }
#   }
#
# Files are loaded ONCE, lazily — the first call to list_panel_templates
# or show_interactive_panel triggers the load. A file with a bad template
# is skipped with a warning, not a hard error. User keys override built-ins
# if they collide (so you can locally customize "confirm", "feedback", etc).

_user_templates_loaded = False


def _templates_dir() -> Path:
    """Resolve the directory for user-contributed templates."""
    env = os.environ.get("ICLI_WEB_TEMPLATES_DIR", "").strip()
    if env: return Path(env).expanduser()
    return Path.home() / ".toolbox" / "icli_web" / "templates"


def _load_user_templates_once() -> None:
    """Read every *.json in the templates dir, merge into _TEMPLATES.

    Idempotent: subsequent calls are no-ops. Never raises — bad files get
    a warn log and are skipped so one broken template can't nuke the tool.

    Logs the resolved directory on first call so debugging "why aren't my
    templates showing up" is a single log inspection away.
    """
    global _user_templates_loaded
    if _user_templates_loaded: return
    _user_templates_loaded = True

    directory = _templates_dir()
    if not directory.exists():
        log.info("icli_web templates dir does not exist: %s "
                 "(set ICLI_WEB_TEMPLATES_DIR or create the dir to load "
                 "user templates)", directory)
        return

    files = sorted(directory.glob("*.json"))
    if not files:
        log.info("icli_web templates dir %s has no *.json files", directory)
        return
    log.info("icli_web loading templates from %s (%d file%s)",
             directory, len(files), "" if len(files) == 1 else "s")

    count = 0
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("skipping template file %s: %s", fp.name, e)
            continue
        if not isinstance(data, dict):
            log.warning("template file %s: top-level must be an object",
                        fp.name)
            continue
        for key, spec in data.items():
            if not isinstance(spec, dict):
                log.warning("template %s in %s: must be object", key, fp.name)
                continue
            if "template" not in spec or "content" not in spec:
                log.warning("template %s in %s: needs template+content",
                            key, fp.name)
                continue
            # Validate before accepting
            v = _validate(spec["template"], spec["content"])
            if not v["ok"]:
                log.warning("template %s in %s invalid: %s",
                            key, fp.name, "; ".join(v["errors"]))
                continue
            _TEMPLATES[key] = {
                "template": spec["template"],
                "category": spec.get("category", "user"),
                "description": spec.get(
                    "description", f"User template from {fp.name}"),
                "content": spec["content"],
            }
            count += 1
    log.info("icli_web loaded %d user template(s) from %s", count, directory)


def reload_user_templates() -> dict:
    """Force a re-scan of the templates directory.

    Clears the one-shot flag so _load_user_templates_once will re-read
    the directory. Useful after dropping new JSON files into
    ICLI_WEB_TEMPLATES_DIR without restarting icli. Returns a summary
    with the directory path, number of files found, and the current
    template keys — so the caller (or the user) can see at a glance
    whether the load worked.
    """
    global _user_templates_loaded
    _user_templates_loaded = False
    _load_user_templates_once()
    directory = _templates_dir()
    files = (sorted(p.name for p in directory.glob("*.json"))
             if directory.exists() else [])
    return {
        "directory": str(directory),
        "exists": directory.exists(),
        "json_files": files,
        "template_keys": sorted(_TEMPLATES.keys()),
    }


# ─── Validation ─────────────────────────────────────────────────────────────

_VALID_TEMPLATES = {"form", "choice", "info", "html", "markdown", "file"}
_VALID_FIELD_TYPES = {"text", "number", "textarea", "select", "checkbox"}

# File-upload defaults. Hard cap at 100MB to prevent runaway base64 blowups
# on the wire from the orb before the agent has a chance to stream into VFS.
_FILE_DEFAULT_MAX_MB = 10
_FILE_HARD_MAX_MB = 100


def _validate_form(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    fields = content.get("fields")
    if not isinstance(fields, list):
        errors.append("form: content.fields must be a list")
        return errors, warnings
    if not fields:
        warnings.append("form: content.fields is empty — nothing to render")
    seen_names = set()
    for i, f in enumerate(fields):
        if not isinstance(f, dict):
            errors.append(f"form: fields[{i}] must be a dict")
            continue
        if not f.get("name"):
            errors.append(f"form: fields[{i}] missing 'name'")
        elif f["name"] in seen_names:
            errors.append(f"form: duplicate field name {f['name']!r}")
        else:
            seen_names.add(f["name"])
        ftype = f.get("type", "text")
        if ftype not in _VALID_FIELD_TYPES:
            errors.append(
                f"form: fields[{i}].type {ftype!r} invalid; "
                f"one of {sorted(_VALID_FIELD_TYPES)}"
            )
        if ftype == "select":
            opts = f.get("options")
            if not isinstance(opts, list) or not opts:
                errors.append(
                    f"form: fields[{i}] (select) needs non-empty 'options'")
    if content.get("submit_label") and not content.get("submit_query"):
        warnings.append(
            "form: submit_label set without submit_query — "
            "the submit button will send a raw JSON blob as the user message")
    return errors, warnings


def _validate_choice(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    opts = content.get("options")
    if not isinstance(opts, list) or not opts:
        errors.append("choice: content.options must be a non-empty list")
    else:
        for i, o in enumerate(opts):
            if isinstance(o, str):
                continue
            if not isinstance(o, dict):
                errors.append(f"choice: options[{i}] must be string or dict")
                continue
            if not o.get("value"):
                errors.append(f"choice: options[{i}] missing 'value'")
    if not content.get("prompt"):
        warnings.append("choice: no prompt set — user sees only the buttons")
    return errors, warnings


def _validate_info(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not content.get("body") and not content.get("title"):
        warnings.append("info: neither body nor title set — panel will be empty")
    return errors, warnings


def _validate_html(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not content.get("html"):
        errors.append("html: content.html is required")
    if content.get("trusted") is not True:
        warnings.append(
            "html: trusted=False means HTML will render as plain text. "
            "Set trusted=True ONLY if the HTML is agent-generated and safe.")
    return errors, warnings


def _validate_markdown(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    # Accept either `md` or `markdown` — both spellings are common in the wild
    # and neither is wrong. Renderer on orb side will prefer `md` then fall
    # back to `markdown`.
    md = content.get("md")
    if md is None:
        md = content.get("markdown")
    if not isinstance(md, str):
        errors.append(
            "markdown: content.md (or content.markdown) must be a string")
        return errors, warnings
    if not md.strip():
        warnings.append(
            "markdown: content.md is empty — panel will render blank")
    return errors, warnings


def _validate_file(content: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    # max_size_mb: per-file cap, validated against a hard ceiling.
    max_mb = content.get("max_size_mb", _FILE_DEFAULT_MAX_MB)
    if not isinstance(max_mb, (int, float)) or max_mb <= 0:
        errors.append(
            f"file: max_size_mb must be a positive number "
            f"(got {max_mb!r})")
    elif max_mb > _FILE_HARD_MAX_MB:
        warnings.append(
            f"file: max_size_mb={max_mb} exceeds the hard ceiling of "
            f"{_FILE_HARD_MAX_MB}MB — the orb will cap uploads at "
            f"{_FILE_HARD_MAX_MB}MB regardless")

    # accept: MIME filter, either a string ("image/*") or a list of strings.
    accept = content.get("accept")
    if accept is not None:
        if isinstance(accept, str):
            pass  # fine
        elif isinstance(accept, list):
            for i, a in enumerate(accept):
                if not isinstance(a, str):
                    errors.append(
                        f"file: accept[{i}] must be a string MIME pattern")
        else:
            errors.append(
                "file: content.accept must be a string or list of strings "
                "(e.g. 'image/*' or ['image/png', 'application/pdf'])")

    # multiple: optional bool, default False
    if "multiple" in content and not isinstance(content["multiple"], bool):
        errors.append("file: content.multiple must be a boolean")

    # vfs_target: where uploaded bytes should land inside the agent's VFS.
    # Metadata-only flow: the orb streams file bytes to the worker which
    # writes them into the active session's VFS under this prefix; the
    # panel state carries only {name, size, type, vfs_path} per file —
    # the agent never sees raw base64 payloads on the message wire.
    target = content.get("vfs_target")
    if target is not None and not isinstance(target, str):
        errors.append("file: content.vfs_target must be a string path")
    elif isinstance(target, str) and not target.strip():
        errors.append("file: content.vfs_target must not be empty")

    if not content.get("title") and not content.get("prompt"):
        warnings.append(
            "file: neither title nor prompt set — drop-zone will be unlabelled")

    return errors, warnings


_VALIDATORS = {
    "form":     _validate_form,
    "choice":   _validate_choice,
    "info":     _validate_info,
    "html":     _validate_html,
    "markdown": _validate_markdown,
    "file":     _validate_file,
}


def _validate(template: str, content: Any) -> dict:
    """Return {ok, errors:[...], warnings:[...]}."""
    errors: list[str] = []
    warnings: list[str] = []
    if template not in _VALID_TEMPLATES:
        errors.append(
            f"template {template!r} not recognised; "
            f"must be one of {sorted(_VALID_TEMPLATES)}"
        )
        return {"ok": False, "errors": errors, "warnings": warnings}
    if not isinstance(content, dict):
        errors.append("content must be a dict")
        return {"ok": False, "errors": errors, "warnings": warnings}
    e, w = _VALIDATORS[template](content)
    errors.extend(e); warnings.extend(w)
    return {"ok": not errors, "errors": errors, "warnings": warnings}


# ─── Public tool functions ──────────────────────────────────────────────────

def show_interactive_panel(template: str,
                           content: dict,
                           state: Optional[dict] = None,
                           panel_id: Optional[str] = None) -> dict:
    """Render an interactive panel in the orb's right-hand pane.

    The spec is validated first; if invalid, returns the validation report
    without rendering. Warnings are returned alongside ok=True so the
    agent can still see them and choose to iterate.

    Args:
        template: "form" | "choice" | "info" | "html"
        template: "form" | "choice" | "info" | "html" | "markdown" | "file"
        content:  Template-specific payload (see list_panel_templates).
        state:    Optional initial field values.
        panel_id: Stable id to update an existing panel. Auto-generated if omitted.

    Returns:
        {"ok": True,  "panel_id": "...", "warnings": [...]}
        {"ok": False, "error": "...",    "errors": [...], "warnings": [...]}
    """
    _load_user_templates_once()
    v = _validate(template, content)
    if not v["ok"]:
        return {
            "ok": False,
            "error": v["errors"][0] if v["errors"] else "invalid panel spec",
            "errors": v["errors"],
            "warnings": v["warnings"],
        }
    result = IcliWebClient.get().send_interactive_panel(
        template=template, state=state, content=content, panel_id=panel_id,
    )
    if v["warnings"]:
        result = {**result, "warnings": v["warnings"]}
    return result


def get_interactive_panel_state() -> dict:
    """Read the current state of all rendered interactive panels.

    Returns a dict shaped like:
        {"panels": {panel_id: {"template": "...", "state": {...}}}}
    Empty dict if called outside an icli_web agent turn.
    """
    return IcliWebClient.get().get_panel_state()


def list_panel_templates(category: Optional[str] = None) -> dict:
    """List built-in panel presets you can use with show_interactive_panel.

    Each entry includes the template type, a description, and the content
    payload you'd pass (as a reference — you can clone and modify it).

    User-contributed templates from ICLI_WEB_TEMPLATES_DIR (or
    ~/.toolbox/icli_web/templates) are included too, with category "user"
    unless overridden in the JSON file.

    Args:
        category: Optional filter: "forms", "choices", "info", "user", …
                  If omitted, all templates are returned.

    Returns:
        {"templates": {template_key: {template, category, description, content}}}
    """
    _load_user_templates_once()
    out = {}
    for key, spec in _TEMPLATES.items():
        if category and spec.get("category") != category:
            continue
        out[key] = {
            "template": spec["template"],
            "category": spec.get("category"),
            "description": spec.get("description"),
            "content": spec["content"],
        }
    return {"templates": out}


def show_template(key: str,
                  overrides: Optional[dict] = None,
                  state: Optional[dict] = None,
                  panel_id: Optional[str] = None) -> dict:
    """Render a built-in (or user-loaded) template by key.

    Shorthand for:
        tpl = list_panel_templates()["templates"][key]
        show_interactive_panel(tpl["template"], tpl["content"], state, panel_id)

    Use this to quickly show a named preset like "globe_3d", "feedback",
    "confirm", or any key returned by list_panel_templates(). If you need
    to change a few fields (title, options, etc.) pass them in `overrides`
    — it's shallow-merged on top of the template's content.

    Args:
        key:       Template key from list_panel_templates() / user JSON files.
        overrides: Optional dict shallow-merged into content. E.g.
                   {"title": "Custom title", "prompt": "Pick X"}.
        state:     Optional initial field values.
        panel_id:  Stable id to update an existing panel.

    Returns:
        {"ok": True, "panel_id": "..."} or {"ok": False, "error": "..."}

    Example:
        show_template("globe_3d")            # render the earth demo
        show_template("confirm",
                      overrides={"prompt": "Deploy to production?"})
    """
    _load_user_templates_once()
    spec = _TEMPLATES.get(key)
    if spec is None:
        # Build a helpful error: list what we DO have, where we looked,
        # and suggest reload_user_templates if the user just added files.
        directory = _templates_dir()
        return {
            "ok": False,
            "error": f"no template named {key!r}",
            "available": sorted(_TEMPLATES.keys()),
            "templates_dir": str(directory),
            "templates_dir_exists": directory.exists(),
            "json_files_found": (
                sorted(p.name for p in directory.glob("*.json"))
                if directory.exists() else []
            ),
            "hint": (
                "If you just added a JSON file to the templates directory, "
                "call reload_user_templates() to pick it up without an "
                "icli restart."
            ),
        }
    content = dict(spec["content"])  # shallow copy so we don't mutate the template
    if overrides:
        content.update(overrides)
    return show_interactive_panel(
        template=spec["template"], content=content,
        state=state, panel_id=panel_id,
    )


def validate_panel_spec(template: str, content: dict) -> dict:
    """Validate a (template, content) pair without rendering anything.

    Useful when the agent is constructing a custom template and wants to
    check it before calling show_interactive_panel. Cheaper + faster than
    trial-and-error.

    Returns:
        {"ok": True|False, "errors": [...], "warnings": [...]}
    """
    return _validate(template, content)


# ─── Registration ───────────────────────────────────────────────────────────

_SHOW_PANEL_DESCRIPTION = (
    "Render an interactive panel in the web orb's right-hand pane. "
    "Templates: 'form' (fields with submit), 'choice' (click-to-pick), "
    "'info' (titled text), 'html' (trusted html only), "
    "'markdown' (escaped-first safe markdown render), "
    "'file' (drag-and-drop upload; bytes land in session VFS, state "
    "exposes metadata only). "
    "Use list_panel_templates to discover built-in presets, and "
    "validate_panel_spec to check custom specs before sending. The orb "
    "echoes user-edited state back in every subsequent query under "
    "context.panels[panel_id].state — you can always read current values "
    "via get_interactive_panel_state()."
)

_GET_STATE_DESCRIPTION = (
    "Read current user-edited state of all rendered interactive panels. "
    "Returns {panels: {panel_id: {template, state}}}."
)

_LIST_TEMPLATES_DESCRIPTION = (
    "List built-in interactive-panel templates ready for use. Optionally "
    "filter by category (forms / choices / info). Returns {templates: {...}}; "
    "each entry has the template type, description, and a content payload "
    "you can clone and pass to show_interactive_panel directly or adapt."
)

_VALIDATE_DESCRIPTION = (
    "Validate a (template, content) spec WITHOUT rendering. Returns "
    "{ok, errors, warnings}. Always call this before show_interactive_panel "
    "when building a custom spec — catches missing fields, bad types, "
    "empty selects, unrecognised template names."
)

_SHOW_TEMPLATE_DESCRIPTION = (
    "Render a preset panel by key (e.g. 'globe_3d', 'feedback', 'confirm'). "
    "Fast shorthand for looking up a template via list_panel_templates and "
    "calling show_interactive_panel. Pass `overrides` to change content "
    "fields (title, prompt, etc.) without having to rebuild the whole spec. "
    "Use this whenever a built-in or user-contributed template already "
    "covers your need — cheaper than constructing content from scratch."
)


_RELOAD_DESCRIPTION = (
    "Force a re-scan of the user-templates directory "
    "(ICLI_WEB_TEMPLATES_DIR, default ~/.toolbox/icli_web/templates). "
    "Use when the user has added or edited JSON template files and you "
    "need the new templates visible without restarting icli. Returns "
    "{directory, exists, json_files, template_keys} — inspect this to "
    "debug why a template isn't showing up."
)


def register_icli_web_tools(agent: Any) -> None:
    """Register all icli_web tools on an agent.

    Idempotent: re-running just re-binds the same functions. Works with
    the tool-spec shape: agent.add_tools([{tool_func, name, description,
    category, flags}]).

    Eagerly loads user templates (from ICLI_WEB_TEMPLATES_DIR) so that
    by the time the agent executes its first query, list_panel_templates
    already returns everything — no cold-start hole where custom templates
    aren't visible yet.
    """
    _load_user_templates_once()
    agent.add_tools([
        {
            "tool_func": show_interactive_panel,
            "name": "show_interactive_panel",
            "description": _SHOW_PANEL_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
        {
            "tool_func": show_template,
            "name": "show_template",
            "description": _SHOW_TEMPLATE_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
        {
            "tool_func": get_interactive_panel_state,
            "name": "get_interactive_panel_state",
            "description": _GET_STATE_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
        {
            "tool_func": list_panel_templates,
            "name": "list_panel_templates",
            "description": _LIST_TEMPLATES_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
        {
            "tool_func": validate_panel_spec,
            "name": "validate_panel_spec",
            "description": _VALIDATE_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
        {
            "tool_func": reload_user_templates,
            "name": "reload_user_templates",
            "description": _RELOAD_DESCRIPTION,
            "category": ["icli_web", "ui"],
            "flags": {"icli_web": True},
        },
    ])
