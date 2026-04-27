import { useState, useCallback, useEffect } from "react";

const BUILTIN_VALIDATORS = [
  { name: "contains", desc: "Response contains value", params: ["value"] },
  { name: "not_contains", desc: "Response does NOT contain value", params: ["value"] },
  { name: "equals", desc: "Response equals value (trimmed, case-insensitive)", params: ["value"] },
  { name: "regex", desc: "Response matches regex pattern", params: ["pattern"] },
  { name: "char_count_gte", desc: "Response ≥ N characters", params: ["value"] },
  { name: "char_count_lte", desc: "Response ≤ N characters", params: ["value"] },
  { name: "max_tokens", desc: "Response ≤ N tokens (whitespace split)", params: ["value"] },
  { name: "json_valid", desc: "Response is valid JSON", params: [] },
  { name: "json_has_key", desc: "Response JSON has key", params: ["key"] },
  { name: "tool_calls_lte", desc: "Tool calls ≤ N", params: ["value"] },
  { name: "tool_calls_gte", desc: "Tool calls ≥ N", params: ["value"] },
  { name: "file_exists", desc: "File was created", params: ["path"] },
  { name: "latency_lte", desc: "Execution time ≤ N seconds", params: ["value"] },
  { name: "any_of", desc: "Contains at least one value", params: ["values"] },
  { name: "all_of", desc: "Contains all values", params: ["values"] },
  { name: "none_of", desc: "Contains none of the values", params: ["values"] },
  { name: "judge", desc: "LLM judge answers yes/no question", params: ["question"] },
  { name: "judge_compare", desc: "Compare against ground truth via judge", params: ["ground_truth", "question"] },
];

const PRESET_MODELS = [
  { id: "gemini-2.5-flash", sig: "openrouter/google/gemini-2.5-flash", caps: ["text"] },
  { id: "gemini-3-flash", sig: "openrouter/google/gemini-3-flash-preview", caps: ["text", "image"] },
  { id: "claude-haiku-4.5", sig: "openrouter/anthropic/claude-haiku-4.5", caps: ["text", "image"] },
  { id: "claude-opus-4.5", sig: "openrouter/anthropic/claude-opus-4.5", caps: ["text", "image"] },
  { id: "grok-fast-4", sig: "openrouter/x-ai/grok-4-fast", caps: ["text"] },
  { id: "grok-code-fast-1", sig: "openrouter/x-ai/grok-code-fast-1", caps: ["text"] },
  { id: "qwen3-8b", sig: "openrouter/qwen/qwen3-8b", caps: ["text"] },
  { id: "qwen3-plus", sig: "openrouter/qwen/qwen3-coder-plus", caps: ["text"] },
  { id: "llama-3.3-70b", sig: "openrouter/meta-llama/llama-3.3-70b-instruct", caps: ["text"] },
  { id: "mistral-8b", sig: "openrouter/mistralai/ministral-8b-2512", caps: ["text"] },
  { id: "gemma-3-27b", sig: "openrouter/google/gemma-3-27b-it", caps: ["text"] },
  { id: "olmo-7b", sig: "openrouter/allenai/olmo-3-7b-instruct", caps: ["text"] },
];

const Tab = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    style={{
      padding: "8px 20px",
      background: active ? "rgba(0, 230, 210, 0.1)" : "transparent",
      border: "1px solid",
      borderColor: active ? "rgba(0, 230, 210, 0.4)" : "rgba(255,255,255,0.06)",
      borderRadius: "4px",
      color: active ? "#00e6d2" : "rgba(255,255,255,0.4)",
      cursor: "pointer",
      fontSize: "11px",
      fontFamily: "'IBM Plex Mono', monospace",
      letterSpacing: "1.5px",
      textTransform: "uppercase",
      transition: "all 0.15s",
    }}
  >
    {children}
  </button>
);

const Badge = ({ children, color = "cyan" }) => {
  const colors = {
    cyan: { bg: "rgba(0,230,210,0.08)", border: "rgba(0,230,210,0.25)", text: "#00e6d2" },
    amber: { bg: "rgba(245,158,11,0.08)", border: "rgba(245,158,11,0.25)", text: "#f59e0b" },
    red: { bg: "rgba(239,68,68,0.08)", border: "rgba(239,68,68,0.25)", text: "#ef4444" },
    gray: { bg: "rgba(255,255,255,0.04)", border: "rgba(255,255,255,0.1)", text: "rgba(255,255,255,0.5)" },
  };
  const c = colors[color] || colors.cyan;
  return (
    <span style={{
      padding: "2px 8px", borderRadius: "3px", fontSize: "9px",
      background: c.bg, border: `1px solid ${c.border}`, color: c.text,
      fontFamily: "'IBM Plex Mono', monospace", letterSpacing: "0.5px",
    }}>
      {children}
    </span>
  );
};

const Card = ({ title, children, actions }) => (
  <div style={{
    background: "rgba(255,255,255,0.015)",
    border: "1px solid rgba(255,255,255,0.04)",
    borderRadius: "6px",
    padding: "20px",
    marginBottom: "16px",
  }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
      <h3 style={{
        fontSize: "9px", fontWeight: 600, letterSpacing: "2px",
        textTransform: "uppercase", color: "rgba(255,255,255,0.25)", margin: 0,
      }}>
        {title}
      </h3>
      {actions && <div style={{ display: "flex", gap: "8px" }}>{actions}</div>}
    </div>
    {children}
  </div>
);

const Input = ({ label, value, onChange, placeholder, mono, small, ...props }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
    {label && (
      <label style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)", letterSpacing: "1px", textTransform: "uppercase" }}>
        {label}
      </label>
    )}
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      style={{
        background: "rgba(0,0,0,0.3)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: "4px",
        padding: small ? "4px 8px" : "8px 12px",
        color: "#e2e2e8",
        fontSize: small ? "11px" : "12px",
        fontFamily: mono ? "'IBM Plex Mono', monospace" : "'IBM Plex Sans', sans-serif",
        outline: "none",
        width: "100%",
        transition: "border-color 0.15s",
      }}
      onFocus={(e) => e.target.style.borderColor = "rgba(0,230,210,0.3)"}
      onBlur={(e) => e.target.style.borderColor = "rgba(255,255,255,0.06)"}
      {...props}
    />
  </div>
);

const Select = ({ label, value, onChange, options, small }) => (
  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
    {label && (
      <label style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)", letterSpacing: "1px", textTransform: "uppercase" }}>
        {label}
      </label>
    )}
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        background: "rgba(0,0,0,0.3)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: "4px",
        padding: small ? "4px 8px" : "8px 12px",
        color: "#e2e2e8",
        fontSize: small ? "11px" : "12px",
        fontFamily: "'IBM Plex Mono', monospace",
        outline: "none",
      }}
    >
      {options.map((o) => (
        <option key={typeof o === "string" ? o : o.value} value={typeof o === "string" ? o : o.value}>
          {typeof o === "string" ? o : o.label}
        </option>
      ))}
    </select>
  </div>
);

const Btn = ({ children, onClick, variant = "default", small, disabled }) => {
  const styles = {
    default: { bg: "rgba(255,255,255,0.03)", border: "rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.6)" },
    primary: { bg: "rgba(0,230,210,0.08)", border: "rgba(0,230,210,0.3)", color: "#00e6d2" },
    danger: { bg: "rgba(239,68,68,0.06)", border: "rgba(239,68,68,0.2)", color: "#ef4444" },
  };
  const s = styles[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: s.bg, border: `1px solid ${s.border}`, color: disabled ? "rgba(255,255,255,0.2)" : s.color,
        borderRadius: "4px", padding: small ? "3px 10px" : "6px 14px", cursor: disabled ? "default" : "pointer",
        fontSize: small ? "9px" : "10px", fontFamily: "'IBM Plex Mono', monospace",
        letterSpacing: "1px", textTransform: "uppercase", transition: "all 0.15s",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      {children}
    </button>
  );
};

// ─── Model Config Panel ───
function ModelPanel({ models, setModels }) {
  const [customId, setCustomId] = useState("");
  const [customSig, setCustomSig] = useState("");

  const addPreset = (p) => {
    if (!models.find((m) => m.id === p.id)) {
      setModels([...models, { ...p, enabled: true }]);
    }
  };

  const addCustom = () => {
    if (customId && customSig) {
      setModels([...models, { id: customId, sig: customSig, caps: ["text"], enabled: true }]);
      setCustomId("");
      setCustomSig("");
    }
  };

  const toggle = (id) => setModels(models.map((m) => m.id === id ? { ...m, enabled: !m.enabled } : m));
  const remove = (id) => setModels(models.filter((m) => m.id !== id));

  return (
    <Card title="Models" actions={<Badge>{models.filter(m => m.enabled).length} active</Badge>}>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "16px" }}>
        {PRESET_MODELS.map((p) => (
          <Btn key={p.id} onClick={() => addPreset(p)} small
            disabled={models.some((m) => m.id === p.id)}>
            + {p.id}
          </Btn>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 2fr auto", gap: "8px", marginBottom: "16px", alignItems: "end" }}>
        <Input label="Model ID" value={customId} onChange={setCustomId} placeholder="my-model" mono small />
        <Input label="LiteLLM String" value={customSig} onChange={setCustomSig} placeholder="openrouter/provider/model" mono small />
        <Btn onClick={addCustom} variant="primary" small>Add</Btn>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
        {models.map((m) => (
          <div key={m.id} style={{
            display: "grid", gridTemplateColumns: "auto 1fr auto auto auto",
            gap: "12px", alignItems: "center", padding: "8px 12px",
            background: m.enabled ? "rgba(0,230,210,0.02)" : "rgba(255,255,255,0.01)",
            border: `1px solid ${m.enabled ? "rgba(0,230,210,0.1)" : "rgba(255,255,255,0.03)"}`,
            borderRadius: "4px",
          }}>
            <input type="checkbox" checked={m.enabled} onChange={() => toggle(m.id)}
              style={{ accentColor: "#00e6d2" }} />
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "12px",
              color: m.enabled ? "#e2e2e8" : "rgba(255,255,255,0.3)" }}>{m.id}</span>
            <span style={{ fontSize: "10px", color: "rgba(255,255,255,0.2)", fontFamily: "'IBM Plex Mono', monospace" }}>
              {m.sig}
            </span>
            <div style={{ display: "flex", gap: "4px" }}>
              {m.caps.map((c) => <Badge key={c} color={c === "text" ? "gray" : "cyan"}>{c}</Badge>)}
            </div>
            <Btn onClick={() => remove(m.id)} variant="danger" small>×</Btn>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ─── Task Builder ───
function TaskBuilder({ tasks, setTasks }) {
  const emptyTask = { id: "", complexity: "tutorial", modality: ["text"], prompt: "", tags: "",
    checks: [], ground_truth: "" };
  const [draft, setDraft] = useState({ ...emptyTask });
  const [checkType, setCheckType] = useState("contains");
  const [checkParam, setCheckParam] = useState("");

  const addCheck = () => {
    const vdef = BUILTIN_VALIDATORS.find((v) => v.name === checkType);
    if (!vdef) return;
    const check = { type: checkType };
    if (vdef.params.includes("value")) check.value = checkParam;
    if (vdef.params.includes("pattern")) check.pattern = checkParam;
    if (vdef.params.includes("key")) check.key = checkParam;
    if (vdef.params.includes("path")) check.path = checkParam;
    if (vdef.params.includes("question")) check.question = checkParam;
    if (vdef.params.includes("ground_truth")) check.ground_truth = checkParam;
    if (vdef.params.includes("values")) {
      try { check.values = JSON.parse(checkParam); }
      catch { check.values = checkParam.split(",").map((s) => s.trim()); }
    }
    setDraft({ ...draft, checks: [...draft.checks, check] });
    setCheckParam("");
  };

  const removeCheck = (i) => setDraft({ ...draft, checks: draft.checks.filter((_, j) => j !== i) });

  const saveTask = () => {
    if (!draft.id || !draft.prompt) return;
    const task = {
      ...draft,
      tags: draft.tags.split(",").map((t) => t.trim()).filter(Boolean),
      modality: draft.modality,
    };
    setTasks([...tasks.filter((t) => t.id !== task.id), task]);
    setDraft({ ...emptyTask });
  };

  const editTask = (t) => setDraft({ ...t, tags: t.tags.join(", ") });
  const deleteTask = (id) => setTasks(tasks.filter((t) => t.id !== id));

  const vdef = BUILTIN_VALIDATORS.find((v) => v.name === checkType);

  return (
    <Card title="Task Builder" actions={<Badge>{tasks.length} tasks</Badge>}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "12px", marginBottom: "12px" }}>
        <Input label="Task ID" value={draft.id} onChange={(v) => setDraft({ ...draft, id: v })} placeholder="logic-001" mono />
        <Select label="Complexity" value={draft.complexity} onChange={(v) => setDraft({ ...draft, complexity: v })}
          options={["tutorial", "extended", "phd"]} />
        <Input label="Tags" value={draft.tags} onChange={(v) => setDraft({ ...draft, tags: v })} placeholder="logic, math" />
      </div>

      <div style={{ marginBottom: "12px" }}>
        <label style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)", letterSpacing: "1px", textTransform: "uppercase", display: "block", marginBottom: "4px" }}>
          Modality
        </label>
        <div style={{ display: "flex", gap: "8px" }}>
          {["text", "image", "document", "video", "audio"].map((m) => (
            <label key={m} style={{ display: "flex", alignItems: "center", gap: "4px", fontSize: "11px", color: "rgba(255,255,255,0.5)", cursor: "pointer" }}>
              <input type="checkbox" checked={draft.modality.includes(m)}
                onChange={() => {
                  const mods = draft.modality.includes(m)
                    ? draft.modality.filter((x) => x !== m)
                    : [...draft.modality, m];
                  setDraft({ ...draft, modality: mods.length ? mods : ["text"] });
                }}
                style={{ accentColor: "#00e6d2" }} />
              {m}
            </label>
          ))}
        </div>
      </div>

      <div style={{ marginBottom: "12px" }}>
        <label style={{ fontSize: "9px", color: "rgba(255,255,255,0.3)", letterSpacing: "1px", textTransform: "uppercase", display: "block", marginBottom: "4px" }}>
          Prompt
        </label>
        <textarea
          value={draft.prompt}
          onChange={(e) => setDraft({ ...draft, prompt: e.target.value })}
          placeholder="Enter the prompt..."
          rows={3}
          style={{
            width: "100%", background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.06)",
            borderRadius: "4px", padding: "8px 12px", color: "#e2e2e8", fontSize: "12px",
            fontFamily: "'IBM Plex Mono', monospace", outline: "none", resize: "vertical",
          }}
        />
      </div>

      <Input label="Ground Truth (optional)" value={draft.ground_truth}
        onChange={(v) => setDraft({ ...draft, ground_truth: v })} placeholder="Expected correct answer" />

      <div style={{ marginTop: "16px", padding: "12px", background: "rgba(0,0,0,0.2)", borderRadius: "4px", border: "1px solid rgba(255,255,255,0.03)" }}>
        <h4 style={{ fontSize: "9px", color: "rgba(255,255,255,0.25)", letterSpacing: "2px", textTransform: "uppercase", marginBottom: "8px" }}>
          Checks (binary validators)
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "180px 1fr auto", gap: "8px", alignItems: "end", marginBottom: "8px" }}>
          <Select value={checkType} onChange={setCheckType}
            options={BUILTIN_VALIDATORS.map((v) => ({ value: v.name, label: v.name }))} small />
          {vdef && vdef.params.length > 0 ? (
            <Input value={checkParam} onChange={setCheckParam}
              placeholder={vdef.params.join(", ")} mono small />
          ) : (
            <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.2)", padding: "6px" }}>no params</div>
          )}
          <Btn onClick={addCheck} variant="primary" small>+ Check</Btn>
        </div>
        {vdef && (
          <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.2)", marginBottom: "8px" }}>{vdef.desc}</div>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          {draft.checks.map((c, i) => (
            <div key={i} style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              padding: "4px 8px", background: "rgba(0,230,210,0.02)", borderRadius: "3px",
              border: "1px solid rgba(0,230,210,0.06)",
            }}>
              <span style={{ fontSize: "11px", fontFamily: "'IBM Plex Mono', monospace", color: "#00e6d2" }}>
                {c.type}
                <span style={{ color: "rgba(255,255,255,0.3)", marginLeft: "8px" }}>
                  {c.value || c.pattern || c.key || c.question || c.path || (c.values && JSON.stringify(c.values)) || ""}
                </span>
              </span>
              <Btn onClick={() => removeCheck(i)} variant="danger" small>×</Btn>
            </div>
          ))}
        </div>
      </div>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px", marginTop: "12px" }}>
        <Btn onClick={() => setDraft({ ...emptyTask })} small>Clear</Btn>
        <Btn onClick={saveTask} variant="primary" disabled={!draft.id || !draft.prompt}>Save Task</Btn>
      </div>

      {tasks.length > 0 && (
        <div style={{ marginTop: "16px" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {["ID", "Complexity", "Modality", "Tags", "Checks", "GT", ""].map((h) => (
                  <th key={h} style={{
                    textAlign: "left", padding: "6px 10px", borderBottom: "1px solid rgba(255,255,255,0.04)",
                    fontSize: "9px", color: "rgba(255,255,255,0.25)", letterSpacing: "1.5px", textTransform: "uppercase",
                  }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tasks.map((t) => (
                <tr key={t.id} style={{ borderBottom: "1px solid rgba(255,255,255,0.02)" }}>
                  <td style={{ padding: "6px 10px", fontSize: "11px", fontFamily: "'IBM Plex Mono', monospace", color: "#00e6d2" }}>{t.id}</td>
                  <td style={{ padding: "6px 10px" }}><Badge color={t.complexity === "phd" ? "red" : t.complexity === "extended" ? "amber" : "gray"}>{t.complexity}</Badge></td>
                  <td style={{ padding: "6px 10px", fontSize: "10px", color: "rgba(255,255,255,0.4)" }}>{t.modality.join(", ")}</td>
                  <td style={{ padding: "6px 10px" }}>
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                      {t.tags.map((tag) => <Badge key={tag}>{tag}</Badge>)}
                    </div>
                  </td>
                  <td style={{ padding: "6px 10px", fontSize: "11px", fontFamily: "'IBM Plex Mono', monospace", color: "rgba(255,255,255,0.5)" }}>{t.checks.length}</td>
                  <td style={{ padding: "6px 10px", fontSize: "11px" }}>{t.ground_truth ? "✓" : ""}</td>
                  <td style={{ padding: "6px 10px", display: "flex", gap: "4px" }}>
                    <Btn onClick={() => editTask(t)} small>Edit</Btn>
                    <Btn onClick={() => deleteTask(t.id)} variant="danger" small>×</Btn>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// ─── Run Config ───
function RunConfig({ config, setConfig }) {
  return (
    <Card title="Run Configuration">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "12px" }}>
        <Select label="Adapter" value={config.adapter} onChange={(v) => setConfig({ ...config, adapter: v })}
          options={[
            { value: "row", label: "Row (direct LiteLLM)" },
            { value: "agent", label: "Agent (ISAA FlowAgent)" },
          ]} />
        <Input label="Timeout (s)" value={config.timeout} onChange={(v) => setConfig({ ...config, timeout: v })} mono />
        <Input label="Seed" value={config.seed} onChange={(v) => setConfig({ ...config, seed: v })} placeholder="random" mono />
        <Input label="Max Workers" value={config.workers} onChange={(v) => setConfig({ ...config, workers: v })} placeholder="auto" mono />
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "12px" }}>
        <Input label="Output Directory" value={config.outputDir} onChange={(v) => setConfig({ ...config, outputDir: v })} mono />
        <Input label="Task Directory" value={config.taskDir} onChange={(v) => setConfig({ ...config, taskDir: v })} mono />
      </div>
    </Card>
  );
}

// ─── Export Panel ───
function ExportPanel({ models, tasks, config }) {
  const [copied, setCopied] = useState(null);

  const activeModels = models.filter((m) => m.enabled);

  const generateYAML = (task) => {
    let y = `id: ${task.id}\ncomplexity: ${task.complexity}\nmodality: [${task.modality.join(", ")}]\ntags: [${task.tags.join(", ")}]\nprompt: |\n  ${task.prompt.replace(/\n/g, "\n  ")}`;
    if (task.ground_truth) y += `\nground_truth: "${task.ground_truth}"`;
    y += "\nchecks:";
    for (const c of task.checks) {
      y += `\n  - type: ${c.type}`;
      for (const [k, v] of Object.entries(c)) {
        if (k === "type") continue;
        if (Array.isArray(v)) y += `\n    ${k}: ${JSON.stringify(v)}`;
        else y += `\n    ${k}: ${JSON.stringify(v)}`;
      }
    }
    return y;
  };

  const generatePython = () => {
    const modelDict = activeModels.map((m) => `    "${m.id}": "${m.sig}",`).join("\n");
    return `from bench.multirun import multi_benchmark, aggregate_dashboard

models = {
${modelDict}
}

report_paths = multi_benchmark(
    models=models,
    task_dir="${config.taskDir || "tasks/"}",
    output_dir="${config.outputDir || "reports/"}",
    adapter_type="${config.adapter}",
    timeout=${config.timeout || 90},${config.seed ? `\n    seed=${config.seed},` : ""}${config.workers ? `\n    max_workers=${config.workers},` : ""}
    skip_existing=True,
)

aggregate_dashboard(report_paths, "comparison.html")
`;
  };

  const generateCLI = () => {
    const modelArgs = activeModels.map((m) => `  ${m.id}=${m.sig}`).join(" \\\n");
    return `python -m bench.multirun \\
  --task-dir ${config.taskDir || "tasks/"} \\
  --output-dir ${config.outputDir || "reports/"} \\
  --adapter ${config.adapter} \\
  --timeout ${config.timeout || 90} \\${config.seed ? `\n  --seed ${config.seed} \\` : ""}
  --dashboard comparison.html \\
${modelArgs}`;
  };

  const copy = (text, label) => {
    navigator.clipboard.writeText(text);
    setCopied(label);
    setTimeout(() => setCopied(null), 2000);
  };

  return (
    <Card title="Export">
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h4 style={{ fontSize: "9px", color: "rgba(255,255,255,0.25)", letterSpacing: "2px", textTransform: "uppercase", margin: 0 }}>Python Script</h4>
            <Btn onClick={() => copy(generatePython(), "python")} variant={copied === "python" ? "primary" : "default"} small>
              {copied === "python" ? "Copied ✓" : "Copy"}
            </Btn>
          </div>
          <pre style={{
            background: "rgba(0,0,0,0.3)", padding: "12px", borderRadius: "4px",
            fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", color: "rgba(255,255,255,0.6)",
            overflow: "auto", maxHeight: "200px", border: "1px solid rgba(255,255,255,0.04)",
            whiteSpace: "pre-wrap",
          }}>
            {generatePython()}
          </pre>
        </div>

        <div>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h4 style={{ fontSize: "9px", color: "rgba(255,255,255,0.25)", letterSpacing: "2px", textTransform: "uppercase", margin: 0 }}>CLI Command</h4>
            <Btn onClick={() => copy(generateCLI(), "cli")} variant={copied === "cli" ? "primary" : "default"} small>
              {copied === "cli" ? "Copied ✓" : "Copy"}
            </Btn>
          </div>
          <pre style={{
            background: "rgba(0,0,0,0.3)", padding: "12px", borderRadius: "4px",
            fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", color: "rgba(255,255,255,0.6)",
            overflow: "auto", maxHeight: "200px", border: "1px solid rgba(255,255,255,0.04)",
            whiteSpace: "pre-wrap",
          }}>
            {generateCLI()}
          </pre>
        </div>
      </div>

      {tasks.length > 0 && (
        <div style={{ marginTop: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h4 style={{ fontSize: "9px", color: "rgba(255,255,255,0.25)", letterSpacing: "2px", textTransform: "uppercase", margin: 0 }}>
              Task YAML Files ({tasks.length})
            </h4>
            <Btn onClick={() => copy(tasks.map(generateYAML).join("\n---\n"), "yaml")}
              variant={copied === "yaml" ? "primary" : "default"} small>
              {copied === "yaml" ? "Copied ✓" : "Copy All"}
            </Btn>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
            {tasks.map((t) => (
              <div key={t.id}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
                  <span style={{ fontSize: "11px", fontFamily: "'IBM Plex Mono', monospace", color: "#00e6d2" }}>{t.id}.yaml</span>
                  <Btn onClick={() => copy(generateYAML(t), t.id)} variant={copied === t.id ? "primary" : "default"} small>
                    {copied === t.id ? "✓" : "Copy"}
                  </Btn>
                </div>
                <pre style={{
                  background: "rgba(0,0,0,0.3)", padding: "8px", borderRadius: "4px",
                  fontSize: "10px", fontFamily: "'IBM Plex Mono', monospace", color: "rgba(255,255,255,0.5)",
                  overflow: "auto", maxHeight: "120px", border: "1px solid rgba(255,255,255,0.04)",
                  whiteSpace: "pre-wrap",
                }}>
                  {generateYAML(t)}
                </pre>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}

// ─── Validator Reference ───
function ValidatorRef() {
  return (
    <Card title="Validator Reference">
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))", gap: "8px" }}>
        {BUILTIN_VALIDATORS.map((v) => (
          <div key={v.name} style={{
            padding: "8px 12px", background: "rgba(0,0,0,0.2)", borderRadius: "4px",
            border: "1px solid rgba(255,255,255,0.03)",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
              <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "11px", color: "#00e6d2" }}>{v.name}</span>
              {v.params.length > 0 && (
                <span style={{ fontSize: "9px", color: "rgba(255,255,255,0.2)", fontFamily: "'IBM Plex Mono', monospace" }}>
                  {v.params.join(", ")}
                </span>
              )}
            </div>
            <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.35)" }}>{v.desc}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

// ─── Main App ───
export default function BenchConfigUI() {
  const [tab, setTab] = useState("models");
  const [models, setModels] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [config, setConfig] = useState({
    adapter: "row", timeout: "90", seed: "", workers: "",
    outputDir: "reports/", taskDir: "tasks/",
  });

  return (
    <div style={{
      fontFamily: "'IBM Plex Sans', -apple-system, sans-serif",
      background: "#08080d",
      color: "#e2e2e8",
      minHeight: "100vh",
      padding: "24px",
    }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "24px",
          paddingBottom: "16px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <div>
            <div style={{ fontSize: "9px", letterSpacing: "3px", textTransform: "uppercase", color: "rgba(255,255,255,0.15)", marginBottom: "4px" }}>
              bench framework
            </div>
            <h1 style={{ fontSize: "22px", fontWeight: 300, margin: 0 }}>Benchmark Configurator</h1>
          </div>
          <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
            <Badge>{models.filter(m => m.enabled).length} models</Badge>
            <Badge>{tasks.length} tasks</Badge>
          </div>
        </div>

        <div style={{ display: "flex", gap: "8px", marginBottom: "20px" }}>
          <Tab active={tab === "models"} onClick={() => setTab("models")}>Models</Tab>
          <Tab active={tab === "tasks"} onClick={() => setTab("tasks")}>Tasks</Tab>
          <Tab active={tab === "config"} onClick={() => setTab("config")}>Config</Tab>
          <Tab active={tab === "export"} onClick={() => setTab("export")}>Export</Tab>
          <Tab active={tab === "reference"} onClick={() => setTab("reference")}>Validators</Tab>
        </div>

        {tab === "models" && <ModelPanel models={models} setModels={setModels} />}
        {tab === "tasks" && <TaskBuilder tasks={tasks} setTasks={setTasks} />}
        {tab === "config" && <RunConfig config={config} setConfig={setConfig} />}
        {tab === "export" && <ExportPanel models={models} tasks={tasks} config={config} />}
        {tab === "reference" && <ValidatorRef />}
      </div>
    </div>
  );
}
