"""
Dreamer V3 — Tool Handler

Implements all dream_* tool logic.
Operates on in-memory copies of Skills, Rules, Personas.
Persist via handle_persist_checkpoint().

Author: FlowAgent V3
"""

import json
import logging
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

_log = logging.getLogger("isaa.dreamer_v3.tool_handler")


# ═══════════════════════════════════════════════════════════════════
# BLOAT CONSTANTS
# ═══════════════════════════════════════════════════════════════════

_MAX_TRIGGERS = 12
_MAX_INSTRUCTION_LEN = 1200
_MAX_TOOLS = 12
_COMPRESS_MAX_TRIGGERS = 6
_COMPRESS_MAX_TOOLS = 8
_COMPRESS_MAX_INSTRUCTION = 800
_EVOLVE_MAX_TRIGGERS = 8
_EVOLVE_MAX_TOOLS = 10


class DreamerToolHandler:
    """
    Implements dream_* tools for the DreamerAgent.

    All methods operate on in-memory data structures.
    Call handle_persist_checkpoint() to write to VFS.
    """

    def __init__(
        self,
        skills: Dict[str, Any],
        rules: Dict[str, Any],
        patterns: List[Any],
        personas: Dict[str, Any],
        records: List[Any] = None,
        dream_cycle_count: int = 1,
        vfs_provider=None,
    ):
        self._skills = skills          # {id: Skill}
        self._rules = rules            # {id: SituationRule}
        self._patterns = patterns      # [LearnedPattern]
        self._personas = personas      # {key: persona_entry}
        self._records = records or []
        self._dream_cycle_count = dream_cycle_count
        # callable -> VFS (taskmap lives in /global, any session vfs works)
        self._vfs_provider = vfs_provider

        self._report = {
            "skills_evolved": [], "skills_created": [], "skills_merged": [],
            "skills_split": [], "skills_compressed": [], "skills_deleted": [],
            "skills_deactivated": [], "rules_created": [], "rules_deleted": [],
            "patterns_added": [], "patterns_pruned": [], "personas_evolved": [],
            "personas_pruned": [], "memories_added": [],
        }


    # ═══════════════════════════════════════════════════════════════
    # TASK MAP ACCESS (multi-run intel from background learning)
    # ═══════════════════════════════════════════════════════════════

    _TASKMAP_ROOT = "/global/.memory/taskmap"

    def _taskmap_vfs(self):
        if self._vfs_provider is None:
            return None
        try:
            return self._vfs_provider()
        except Exception:
            return None

    @staticmethod
    def _vfs_read(vfs, path: str) -> str:
        try:
            r = vfs.read(path)
            return r.get("content", "") if r.get("success") else ""
        except Exception:
            return ""

    def handle_get_taskmap(self, task_type: str = "", subtype: str = "", limit: int = 20) -> str:
        """
        Multi-run intel from the task map (written by background learning
        after every run). Without args: overview of all classes (_index).
        With task_type+subtype: the last `limit` formatted rows + sub index
        + happypath/guid status — THE data basis for cross-run evaluation.
        """
        vfs = self._taskmap_vfs()
        if vfs is None:
            return json.dumps({"error": "no VFS available for task map access"})

        root = self._TASKMAP_ROOT
        top_raw = self._vfs_read(vfs, f"{root}/_index.json")
        if not top_raw:
            return json.dumps({"error": "task map empty — no background-learning data yet"})

        if not task_type:
            try:
                top = json.loads(top_raw)
            except Exception:
                return json.dumps({"error": "corrupt taskmap _index.json"})
            # enrich overview with per-subtype indexes (new flags = Dreamer priority)
            for tt, info in (top.get("task_types") or {}).items():
                subs = {}
                for st in info.get("subtypes", []):
                    raw = self._vfs_read(vfs, f"{root}/{tt}/{st}/_index.json")
                    if raw:
                        try:
                            subs[st] = json.loads(raw)
                        except Exception:
                            pass
                info["subtype_indexes"] = subs
            return json.dumps(top, indent=2, default=str)

        sub = subtype or "general"
        base = f"{root}/{task_type}/{sub}"
        rows_raw = self._vfs_read(vfs, f"{base}/formatted_row.jsonl")
        rows = []
        for line in rows_raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        idx_raw = self._vfs_read(vfs, f"{base}/_index.json")
        out = {
            "task_type": task_type,
            "subtype": sub,
            "index": json.loads(idx_raw) if idx_raw else {},
            "rows": rows[-max(1, int(limit)):],
            "row_count_total": len(rows),
            "happypath": self._vfs_read(vfs, f"{base}/happypath.md")[:2000],
            "guid_exists": bool(self._vfs_read(vfs, f"{base}/guid.md")),
        }
        return json.dumps(out, indent=2, default=str)

    def handle_write_taskmap_guide(self, task_type: str, subtype: str, content: str) -> str:
        """Write/replace guid.md for a class — the Dreamer's per-task guide."""
        vfs = self._taskmap_vfs()
        if vfs is None:
            return json.dumps({"error": "no VFS available"})
        if not task_type or not content:
            return json.dumps({"error": "task_type and content required"})
        sub = subtype or "general"
        path = f"{self._TASKMAP_ROOT}/{task_type}/{sub}/guid.md"
        try:
            vfs.mkdir(f"{self._TASKMAP_ROOT}/{task_type}/{sub}", parents=True)
        except Exception:
            pass
        r = vfs.write(path, content)
        ok = bool(r.get("success", True)) if isinstance(r, dict) else True
        if ok:
            self._report.setdefault("taskmap_guides_written", []).append(f"{task_type}/{sub}")
        return json.dumps({"success": ok, "path": path})

    # ═══════════════════════════════════════════════════════════════
    # BLOAT CALCULATION
    # ═══════════════════════════════════════════════════════════════

    def calculate_bloat(self, skill) -> float:
        """0.0-1.0 bloat score. Matches Diamond display metric."""
        trigger_bloat = min(1.0, len(skill.triggers) / _MAX_TRIGGERS)
        instruction_bloat = min(1.0, len(skill.instruction) / _MAX_INSTRUCTION_LEN)
        tools_bloat = min(1.0, len(skill.tools_used) / _MAX_TOOLS)
        return trigger_bloat * 0.25 + instruction_bloat * 0.50 + tools_bloat * 0.25

    # ═══════════════════════════════════════════════════════════════
    # DATA ACCESS (read-only views for the DreamerAgent)
    # ═══════════════════════════════════════════════════════════════

    def handle_get_records(
        self,
        query_filter: str = "",
        success_only: bool = False,
        failure_only: bool = False,
        limit: int = 50,
    ) -> str:
        """Return filtered RunRecords as JSON."""
        from toolboxv2.mods.isaa.base.dreamer.harvest import filter_records
        filtered = filter_records(
            self._records,
            query_filter=query_filter,
            success_only=success_only,
            failure_only=failure_only,
            limit=limit,
        )
        records_out = []
        for r in filtered:
            if hasattr(r, 'to_dict'):
                records_out.append(r.to_dict())
            else:
                records_out.append({
                    "run_id": r.run_id, "query": r.query,
                    "tools_used": r.tools_used, "success": r.success,
                    "error_traces": r.error_traces, "summary": r.summary,
                })
        return json.dumps(records_out, indent=2, default=str)

    def handle_get_skills(self) -> str:
        """Return all skills with stats + bloat as JSON."""
        skills_info = []
        for sid, skill in self._skills.items():
            skills_info.append({
                "id": sid,
                "name": skill.name,
                "source": skill.source,
                "confidence": round(skill.confidence, 3),
                "effectiveness": round(getattr(skill, 'effectiveness', 0.0), 3),
                "total_uses": getattr(skill, 'total_uses', 0),
                "avg_iterations": round(getattr(skill, 'avg_iterations', 0.0), 1),
                "triggers": skill.triggers[:5],
                "trigger_count": len(skill.triggers),
                "tool_count": len(skill.tools_used),
                "instruction_len": len(skill.instruction),
                "bloat_score": round(self.calculate_bloat(skill), 3),
                "active": skill.is_active(),
            })
        return json.dumps(skills_info, indent=2)

    def handle_get_rules(self) -> str:
        """Return all rules + patterns as JSON."""
        rules_out = []
        for rid, rule in self._rules.items():
            rules_out.append({
                "id": rid,
                "situation": rule.situation,
                "intent": rule.intent,
                "confidence": rule.confidence,
                "instructions": rule.instructions,
                "success_count": getattr(rule, 'success_count', 0),
                "failure_count": getattr(rule, 'failure_count', 0),
            })
        patterns_out = []
        for p in self._patterns:
            patterns_out.append({
                "pattern": p.pattern,
                "source": p.source_situation,
                "confidence": p.confidence,
                "usage_count": p.usage_count,
                "category": getattr(p, 'category', 'general'),
            })
        return json.dumps({"rules": rules_out, "patterns": patterns_out}, indent=2)

    def handle_get_personas(self) -> str:
        """Return all personas as JSON."""
        return json.dumps(self._personas, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════
    # CLUSTERING
    # ═══════════════════════════════════════════════════════════════

    _STOP_WORDS = {"ich", "ein", "eine", "der", "die", "das", "wie", "was", "ist",
                   "und", "oder", "für", "mit", "von", "zu", "the", "a", "an",
                   "is", "to", "for", "and", "of", "in", "on", "how", "what", "can"}

    def handle_cluster_records(
        self,
        record_ids: List[str] = None,
        threshold: float = 0.65,
    ) -> str:
        """Cluster RunRecords by keyword similarity. Returns JSON map."""
        records = self._records
        if record_ids:
            records = [r for r in records if r.run_id in set(record_ids)]

        if not records:
            return json.dumps({})

        clusters: Dict[str, list] = {}
        assigned = set()

        for i, r in enumerate(records):
            if i in assigned:
                continue
            words_i = set(r.query.lower().split()) - self._STOP_WORDS
            cluster = [r]
            assigned.add(i)

            for j in range(i + 1, len(records)):
                if j in assigned:
                    continue
                words_j = set(records[j].query.lower().split()) - self._STOP_WORDS
                union = len(words_i | words_j)
                overlap = len(words_i & words_j)
                if union > 0 and overlap / union >= (1.0 - threshold):
                    cluster.append(records[j])
                    assigned.add(j)

            cid = f"c_{len(clusters)}"
            success_n = sum(1 for rec in cluster if rec.success)
            clusters[cid] = cluster
            # Store metadata alongside

        # Build output
        result = {}
        for cid, recs in clusters.items():
            success_n = sum(1 for rec in recs if rec.success)
            result[cid] = {
                "intent": recs[0].query[:80] if recs else "",
                "record_count": len(recs),
                "success_count": success_n,
                "queries": [rec.query[:100] for rec in recs[:10]],
                "tools": list(set(t for rec in recs for t in rec.tools_used[:5])),
            }

        # Add unclustered as singles
        for i, r in enumerate(records):
            if i not in assigned:
                cid = f"single_{i}"
                result[cid] = {
                    "intent": r.query[:80],
                    "record_count": 1,
                    "success_count": 1 if r.success else 0,
                    "queries": [r.query[:100]],
                    "tools": r.tools_used[:5],
                }

        self._report["clusters_found"] = len(result)
        return json.dumps(result, indent=2)

    # ═══════════════════════════════════════════════════════════════
    # PERSONA EVOLUTION
    # ═══════════════════════════════════════════════════════════════

    def handle_evolve_persona(
        self,
        name: str = "",
        prompt_modifier: str = "",
        model_preference: str = "fast",
        temperature: float = 0.3,
        max_iterations_factor: float = 1.0,
        verification_level: str = "basic",
        dominant_intent: str = "",
        evidence_count: int = 1,
        **kwargs,
    ) -> str:
        """Create or update a learned persona."""
        if not name:
            return "ERROR: name is required"

        import re
        key = "learned_" + re.sub(r"\W+", "_", name[:25]).lower().strip("_")

        if key in self._personas:
            entry = self._personas[key]
            old_conf = entry.get("confidence", 0.5)
            total_ev = entry.get("evidence_count", 0) + evidence_count
            weight = min(0.5, evidence_count / max(total_ev, 1))
            new_conf = old_conf * (1 - weight) + 0.5 * weight
            entry["confidence"] = round(min(1.0, new_conf), 4)
            entry["evidence_count"] = total_ev
            entry["dream_cycles"] = entry.get("dream_cycles", 0) + 1
            entry["profile"] = {
                "name": name, "prompt_modifier": prompt_modifier,
                "model_preference": model_preference, "temperature": temperature,
                "max_iterations_factor": max_iterations_factor,
                "verification_level": verification_level,
            }
        else:
            self._personas[key] = {
                "profile": {
                    "name": name, "prompt_modifier": prompt_modifier,
                    "model_preference": model_preference, "temperature": temperature,
                    "max_iterations_factor": max_iterations_factor,
                    "verification_level": verification_level,
                },
                "confidence": 0.3,
                "evidence_count": evidence_count,
                "dream_cycles": 1,
                "usage_count": 0,
                "created": datetime.now().isoformat(),
            }

        self._report["personas_evolved"].append(key)
        return f"OK: Persona '{key}' evolved (conf={self._personas[key]['confidence']:.2f})"

    # ═══════════════════════════════════════════════════════════════
    # MEMORY EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def handle_extract_memories(self, memories: List[Dict] = None) -> str:
        """Store extracted facts. In production writes to agent memory."""
        memories = memories or []
        added = []
        for m in memories:
            text = m.get("text", "")
            concepts = m.get("concepts", [])
            if text:
                added.append(text[:80])
        self._report["memories_added"].extend(added)
        return f"OK: {len(added)} memories extracted" if added else "OK: No memories to extract"

    # ═══════════════════════════════════════════════════════════════
    # SKILL EVOLUTION
    # ═══════════════════════════════════════════════════════════════

    def handle_evolve_skill(
        self,
        skill_id: str,
        cluster_size: int,
        success_ratio: float,
        instruction_update: str = "",
        failure_patterns: List[str] = None,
        new_triggers: List[str] = None,
        success_tools: List[str] = None,
    ) -> str:
        skill = self._skills.get(skill_id)
        if not skill:
            return f"ERROR: Skill '{skill_id}' not found"

        version = getattr(skill, '_version', 1)

        # Rollback storage
        if not hasattr(skill, '_instruction_history'):
            skill._instruction_history = []
        skill._instruction_history.append({
            "version": version,
            "instruction": skill.instruction,
            "date": datetime.now().isoformat(),
        })
        skill._instruction_history = skill._instruction_history[-5:]
        skill._version = version + 1

        # Evidence gate
        is_mature = skill.confidence >= 0.7 and skill.source == "predefined"
        too_few = cluster_size < 3

        if instruction_update:
            if is_mature and too_few:
                pass  # Don't touch instruction
            elif is_mature:
                # Merge: old base + new insights (ALWAYS keep old for mature)
                base = skill.instruction
                if "\n\n── EVOLVED UPDATE" in base:
                    base = base[:base.index("\n\n── EVOLVED UPDATE")]
                merged = f"{base}\n\n── EVOLVED UPDATE ──\n{instruction_update}\n"
                if len(merged) > 1500:
                    merged = merged[:1500] + "\n[...truncated]"
                skill.instruction = merged
            else:
                # Learned/low-conf → replace
                skill.instruction = instruction_update

        # Failure patterns
        if failure_patterns:
            negatives = "\n".join(f"⚠️ {p}" for p in failure_patterns[:3])
            if "\nBEKANNTE FALLSTRICKE" in skill.instruction:
                idx = skill.instruction.index("\nBEKANNTE FALLSTRICKE")
                skill.instruction = skill.instruction[:idx]
            skill.instruction += f"\n\nBEKANNTE FALLSTRICKE (v{skill._version}):\n{negatives}\n"

        # Triggers (cap at 8)
        if new_triggers:
            existing = set(t.lower() for t in skill.triggers)
            for t in new_triggers:
                if t.lower() not in existing:
                    skill.triggers.append(t)
                    existing.add(t.lower())
            if len(skill.triggers) > _EVOLVE_MAX_TRIGGERS:
                skill.triggers = skill.triggers[:_EVOLVE_MAX_TRIGGERS]

        # Tools (cap at 10)
        if success_tools:
            existing_tools = set(skill.tools_used)
            for t in success_tools:
                if t not in existing_tools and t not in ("think", "final_answer"):
                    if len(skill.tools_used) < _EVOLVE_MAX_TOOLS:
                        skill.tools_used.append(t)

        # Confidence update (evidence-weighted)
        total_evidence = skill.success_count + skill.failure_count + cluster_size
        weight_new = min(0.5, cluster_size / max(total_evidence, 1))
        weight_old = 1.0 - weight_new
        skill.confidence = min(1.0, skill.confidence * weight_old + success_ratio * weight_new)
        skill.last_used = datetime.now()

        self._report["skills_evolved"].append(skill_id)
        return f"OK: Skill '{skill.name}' evolved to v{skill._version} (conf={skill.confidence:.2f})"

    def _merge_instructions(self, old: str, new: str) -> str:
        """Merge old + new instruction. Old stays as base, new appended."""
        if len(old) < 30:
            return new

        base = old
        if "\n\n── EVOLVED UPDATE" in base:
            base = base[:base.index("\n\n── EVOLVED UPDATE")]

        merged = f"{base}\n\n── EVOLVED UPDATE ──\n{new}\n"

        if len(merged) > 1500:
            merged = merged[:1500] + "\n[...truncated]"

        return merged

    # ═══════════════════════════════════════════════════════════════
    # SKILL CREATION
    # ═══════════════════════════════════════════════════════════════

    def handle_create_skill(
        self,
        name: str,
        triggers: List[str],
        instruction: str,
        tools_used: List[str] = None,
        failure_patterns: List[str] = None,
    ) -> str:
        skill_id = self._generate_skill_id(name)

        skill_cls = type(list(self._skills.values())[0]) if self._skills else None
        if skill_cls is None:
            # Fallback: build a duck-typed object
            from toolboxv2.mods.isaa.base.dreamer.harvest import RunRecord  # just for import path test
            from dataclasses import make_dataclass
            # Use a simple namespace
            class _Skill:
                pass
            s = _Skill()
            for k, v in {
                "id": skill_id, "name": name[:50], "triggers": triggers,
                "instruction": instruction, "tools_used": tools_used or [],
                "tool_groups": [], "source": "learned", "confidence": 0.3,
                "activation_threshold": 0.6, "success_count": 0, "failure_count": 0,
                "total_uses": 0, "created_at": datetime.now(), "last_used": None,
                "recent_queries": [],
            }.items():
                setattr(s, k, v)
            s.is_active = lambda: s.confidence >= s.activation_threshold
            s.matches_keywords = lambda q: any(t.lower() in q.lower() for t in s.triggers)
            s.to_dict = lambda: {k: getattr(s, k) for k in [
                "id", "name", "triggers", "instruction", "tools_used", "source",
                "confidence", "total_uses"]}
            self._skills[skill_id] = s
        else:
            s = skill_cls(
                id=skill_id, name=name[:50], triggers=triggers,
                instruction=instruction, tools_used=tools_used or [],
                source="learned", confidence=0.3,
            )
            self._skills[skill_id] = s

        s._version = 1

        if failure_patterns:
            negatives = "\n".join(f"⚠️ {p}" for p in failure_patterns[:3])
            s.instruction += f"\n\nBEKANNTE FALLSTRICKE:\n{negatives}\n"

        self._report["skills_created"].append(skill_id)
        return f"OK: Skill '{name}' created (id={skill_id}, conf=0.3)"

    def _generate_skill_id(self, name: str) -> str:
        normalized = name.lower().replace(' ', '_').replace('-', '_')
        normalized = ''.join(c for c in normalized if c.isalnum() or c == '_')
        base_id = f"learned_{normalized[:20]}"
        if base_id not in self._skills:
            return base_id
        counter = 1
        while f"{base_id}_{counter}" in self._skills:
            counter += 1
        return f"{base_id}_{counter}"

    # ═══════════════════════════════════════════════════════════════
    # SKILL MERGING
    # ═══════════════════════════════════════════════════════════════

    def handle_merge_skills(
        self,
        primary_skill_id: str,
        secondary_skill_id: str,
        merged_instruction: str = "",
    ) -> str:
        primary = self._skills.get(primary_skill_id)
        secondary = self._skills.get(secondary_skill_id)

        if not primary or not secondary:
            missing = primary_skill_id if not primary else secondary_skill_id
            return f"ERROR: Skill '{missing}' not found"

        # Swap if secondary has higher confidence
        if secondary.confidence > primary.confidence:
            primary, secondary = secondary, primary
            primary_skill_id, secondary_skill_id = secondary_skill_id, primary_skill_id

        # Merge triggers (dedup)
        existing = set(t.lower() for t in primary.triggers)
        for t in secondary.triggers:
            if t.lower() not in existing:
                primary.triggers.append(t)
                existing.add(t.lower())

        # Merge tools (dedup)
        existing_tools = set(primary.tools_used)
        for t in secondary.tools_used:
            if t not in existing_tools:
                primary.tools_used.append(t)
                existing_tools.add(t)

        # Merge instruction
        if merged_instruction:
            primary.instruction = merged_instruction
        else:
            if len(secondary.instruction) > 50:
                primary.instruction = self._merge_instructions(
                    primary.instruction, secondary.instruction
                )

        # Delete secondary
        del self._skills[secondary_skill_id]

        self._report["skills_merged"].append(f"{secondary_skill_id}→{primary_skill_id}")
        return f"OK: Merged '{secondary.name}' into '{primary.name}'"

    # ═══════════════════════════════════════════════════════════════
    # SKILL SPLITTING
    # ═══════════════════════════════════════════════════════════════

    def handle_split_skill(
        self,
        skill_id: str,
        sub_intents: List[str],
    ) -> str:
        parent = self._skills.get(skill_id)
        if not parent:
            return f"ERROR: Skill '{skill_id}' not found"

        new_ids = []
        for intent in sub_intents[:3]:
            sub_id = self._generate_skill_id(intent)

            # Build sub-skill using same class as parent
            skill_cls = type(parent)
            sub = skill_cls(
                id=sub_id,
                name=intent[:50],
                triggers=[intent.lower()] + [w for w in intent.lower().split() if len(w) > 3],
                instruction=f"Spezialisierung von '{parent.name}':\n{intent}\n\n"
                            f"Basis:\n{parent.instruction[:300]}",
                tools_used=parent.tools_used[:5],
                source="learned",
                confidence=parent.confidence * 0.8,
            )
            sub._version = 1
            sub._parent_skill = skill_id
            self._skills[sub_id] = sub
            new_ids.append(sub_id)

        # Deactivate parent
        if new_ids:
            parent.activation_threshold = 1.1
            parent._split_into = new_ids

        self._report["skills_split"].extend(new_ids)
        return f"OK: Split '{parent.name}' → {new_ids}"

    # ═══════════════════════════════════════════════════════════════
    # SKILL COMPRESSION
    # ═══════════════════════════════════════════════════════════════

    def handle_compress_skill(self, skill_id: str) -> str:
        skill = self._skills.get(skill_id)
        if not skill:
            return f"ERROR: Skill '{skill_id}' not found"

        # Instruction: remove old EVOLVED UPDATE sections
        instruction = skill.instruction
        parts = instruction.split("── EVOLVED UPDATE")
        if len(parts) > 2:
            instruction = parts[0] + "── EVOLVED UPDATE" + parts[-1]

        if len(instruction) > _COMPRESS_MAX_INSTRUCTION:
            cut = instruction[:_COMPRESS_MAX_INSTRUCTION].rfind("\n")
            if cut < _COMPRESS_MAX_INSTRUCTION * 0.5:
                cut = _COMPRESS_MAX_INSTRUCTION
            instruction = instruction[:cut].rstrip()
        skill.instruction = instruction

        # Triggers: keep first 3 + shortest rest
        if len(skill.triggers) > _COMPRESS_MAX_TRIGGERS:
            originals = skill.triggers[:3]
            rest = sorted(skill.triggers[3:], key=len)[:_COMPRESS_MAX_TRIGGERS - 3]
            skill.triggers = originals + rest

        # Tools: cap
        if len(skill.tools_used) > _COMPRESS_MAX_TOOLS:
            skill.tools_used = skill.tools_used[:_COMPRESS_MAX_TOOLS]

        self._report["skills_compressed"].append(skill_id)
        return f"OK: Compressed '{skill.name}' (bloat: {self.calculate_bloat(skill):.0%})"

    # ═══════════════════════════════════════════════════════════════
    # CLEANUP SKILLS
    # ═══════════════════════════════════════════════════════════════

    def handle_cleanup_skills(self) -> str:
        deleted = []
        deactivated = []
        compressed = []
        merged = []

        to_delete = []
        for sid, skill in self._skills.items():
            # 1. Delete bad (conf<0.15, uses≥5, not predefined)
            if (skill.source != "predefined"
                    and skill.confidence < 0.15
                    and skill.total_uses >= 5):
                to_delete.append(sid)
                continue

            # 2. Deactivate stale (no match in 3+ cycles)
            cycles_stale = getattr(skill, '_dream_cycles_since_last_match',
                                   self._dream_cycle_count if skill.last_used is None else 0)
            if (skill.source != "predefined"
                    and cycles_stale >= 3
                    and skill.is_active()):
                skill.activation_threshold = 1.1
                deactivated.append(sid)

            # 3. Compress bloated
            if self.calculate_bloat(skill) > 0.7:
                self.handle_compress_skill(sid)
                compressed.append(sid)

        for sid in to_delete:
            del self._skills[sid]
            deleted.append(sid)

        # 4. Merge duplicates (same normalized name)
        seen = {}
        merge_pairs = []
        for sid, skill in list(self._skills.items()):
            norm = skill.name.lower().strip()
            if norm in seen:
                merge_pairs.append((seen[norm], sid))
            else:
                seen[norm] = sid

        for primary_id, secondary_id in merge_pairs:
            if primary_id in self._skills and secondary_id in self._skills:
                self.handle_merge_skills(primary_id, secondary_id)
                merged.append(f"{secondary_id}→{primary_id}")

        self._report["skills_deleted"].extend(deleted)
        self._report["skills_deactivated"].extend(deactivated)

        parts = []
        if deleted: parts.append(f"{len(deleted)} deleted")
        if deactivated: parts.append(f"{len(deactivated)} deactivated")
        if compressed: parts.append(f"{len(compressed)} compressed")
        if merged: parts.append(f"{len(merged)} merged")

        return f"OK: Cleanup — {', '.join(parts) or 'no changes'}"

    # ═══════════════════════════════════════════════════════════════
    # CLEANUP RULES
    # ═══════════════════════════════════════════════════════════════

    def handle_cleanup_rules(self) -> str:
        deleted_rules = []
        pruned_patterns = []

        # 1. Delete low-confidence rules
        to_delete = [
            rid for rid, rule in self._rules.items()
            if rule.confidence < 0.2
        ]
        for rid in to_delete:
            del self._rules[rid]
            deleted_rules.append(rid)

        # 2. Prune unused patterns (usage_count=0 after 3+ cycles)
        kept = []
        for p in self._patterns:
            if p.usage_count == 0 and self._dream_cycle_count >= 3:
                pruned_patterns.append(p.pattern[:50])
            else:
                kept.append(p)
        self._patterns = kept

        # 3. Cap patterns at 50 (oldest first = front of list)
        if len(self._patterns) > 50:
            overflow = len(self._patterns) - 50
            pruned_patterns.extend([p.pattern[:50] for p in self._patterns[:overflow]])
            self._patterns = self._patterns[overflow:]

        self._report["rules_deleted"].extend(deleted_rules)
        self._report["patterns_pruned"].extend(pruned_patterns)

        parts = []
        if deleted_rules: parts.append(f"{len(deleted_rules)} rules deleted")
        if pruned_patterns: parts.append(f"{len(pruned_patterns)} patterns pruned")

        return f"OK: Cleanup — {', '.join(parts) or 'no changes'}"

    # ═══════════════════════════════════════════════════════════════
    # DELETE SKILL/RULE
    # ═══════════════════════════════════════════════════════════════

    def handle_delete_skill(self, skill_id: str, reason: str) -> str:
        skill = self._skills.get(skill_id)
        if not skill:
            return f"ERROR: Skill '{skill_id}' not found"

        if skill.source == "predefined":
            # Deactivate instead of delete
            skill.activation_threshold = 1.1
            self._report["skills_deactivated"].append(skill_id)
            return f"OK: Predefined skill '{skill.name}' deactivated (reason: {reason})"

        del self._skills[skill_id]
        self._report["skills_deleted"].append(skill_id)
        return f"OK: Skill '{skill.name}' deleted (reason: {reason})"

    def handle_delete_rule(self, rule_id: str, reason: str) -> str:
        if rule_id not in self._rules:
            return f"ERROR: Rule '{rule_id}' not found"

        del self._rules[rule_id]
        self._report["rules_deleted"].append(rule_id)
        return f"OK: Rule '{rule_id}' deleted (reason: {reason})"

    # ═══════════════════════════════════════════════════════════════
    # PERSONA PRUNING
    # ═══════════════════════════════════════════════════════════════

    def handle_prune_personas(self) -> str:
        pruned = []

        for key in list(self._personas.keys()):
            entry = self._personas[key]
            conf = entry.get("confidence", 1.0)
            ev = entry.get("evidence_count", 0)
            cyc = entry.get("dream_cycles", 0)
            usage = entry.get("usage_count", 0)

            # Bad confidence with enough evidence
            if conf < 0.25 and ev >= 5:
                del self._personas[key]
                pruned.append(key)
                continue

            # Zero usage after 3+ cycles
            if usage == 0 and cyc >= 3:
                del self._personas[key]
                pruned.append(key)

        self._report["personas_pruned"].extend(pruned)
        return f"OK: Pruned {len(pruned)} personas: {pruned}" if pruned else "OK: No personas to prune"

    # ═══════════════════════════════════════════════════════════════
    # RULE/PATTERN EXTRACTION
    # ═══════════════════════════════════════════════════════════════

    def handle_extract_rules(self, rules_data: List[Dict]) -> str:
        created = []
        for rd in rules_data:
            rule_id = f"rule_{uuid.uuid4().hex[:8]}"

            # Build rule object matching SituationRule interface
            rule_cls = type(list(self._rules.values())[0]) if self._rules else None
            if rule_cls:
                rule = rule_cls(
                    id=rule_id,
                    situation=rd["situation"],
                    intent=rd["intent"],
                    instructions=rd.get("instructions", []),
                    required_tool_groups=rd.get("required_tool_groups", []),
                    learned=True,
                    confidence=rd.get("confidence", 0.5),
                )
            else:
                # Fallback: simple namespace
                class _Rule:
                    pass
                rule = _Rule()
                rule.id = rule_id
                rule.situation = rd["situation"]
                rule.intent = rd["intent"]
                rule.instructions = rd.get("instructions", [])
                rule.required_tool_groups = rd.get("required_tool_groups", [])
                rule.learned = True
                rule.confidence = rd.get("confidence", 0.5)
                rule.success_count = 0
                rule.failure_count = 0
                rule.created_at = datetime.now()
                rule.last_used = None

            self._rules[rule_id] = rule
            created.append(rule_id)

        self._report["rules_created"].extend(created)
        return f"OK: Created {len(created)} rules"

    def handle_learn_pattern(
        self,
        pattern: str,
        source_situation: str,
        category: str = "general",
        tags: List[str] = None,
    ) -> str:
        # Check for duplicates
        for existing in self._patterns:
            if existing.pattern.lower() == pattern.lower():
                return f"OK: Pattern already exists (skipped)"

        # Build pattern object
        pat_cls = type(self._patterns[0]) if self._patterns else None
        if pat_cls:
            p = pat_cls(
                pattern=pattern,
                source_situation=source_situation,
                confidence=0.5,
                category=category,
                tags=tags or [],
            )
        else:
            class _Pattern:
                pass
            p = _Pattern()
            p.pattern = pattern
            p.source_situation = source_situation
            p.confidence = 0.5
            p.usage_count = 0
            p.category = category
            p.tags = tags or []
            p.created_at = datetime.now()
            p.last_used = None

        self._patterns.append(p)
        self._report["patterns_added"].append(pattern[:50])
        return f"OK: Pattern learned: '{pattern[:60]}'"

    # ═══════════════════════════════════════════════════════════════
    # PERSIST CHECKPOINT
    # ═══════════════════════════════════════════════════════════════

    def handle_persist_checkpoint(self, vfs) -> str:
        """Write all data to VFS."""
        try:
            # Skills
            skills_data = {}
            for sid, s in self._skills.items():
                if hasattr(s, 'to_dict'):
                    skills_data[sid] = s.to_dict()
                else:
                    skills_data[sid] = {
                        "id": s.id, "name": s.name,
                        "confidence": s.confidence, "source": s.source,
                    }

            vfs.write(
                "/global/.memory/skills_checkpoint.json",
                json.dumps({"skills": skills_data}, indent=2, default=str),
            )

            # Rules
            rules_data = {}
            for rid, r in self._rules.items():
                if hasattr(r, '__dict__'):
                    rules_data[rid] = {
                        k: v for k, v in r.__dict__.items()
                        if not k.startswith('_')
                    }

            vfs.write(
                "/global/.memory/rules_checkpoint.json",
                json.dumps({"rules": rules_data}, indent=2, default=str),
            )

            # Personas
            vfs.write(
                "/global/.memory/dreamer/personas.json",
                json.dumps(self._personas, indent=2, default=str),
            )

            # Report
            self._report["finished_at"] = datetime.now().isoformat()
            vfs.write(
                "/global/.memory/dreamer/last_report.json",
                json.dumps(self._report, indent=2, default=str),
            )

            return f"OK: Checkpoint persisted ({len(self._skills)} skills, {len(self._rules)} rules)"

        except Exception as e:
            return f"ERROR: Persist failed: {e}"
