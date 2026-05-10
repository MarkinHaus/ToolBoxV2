"""Tests for profiler.py — unit tests + end-to-end tests."""

import cProfile
import json
import os
import pstats
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure test_fixtures is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from toolboxv2.utils.extras.code_analyzer.profiler import (
    _build_function_graph,
    _build_html_graph,
    _get_package_name,
    _group_by_package,
    _parse_imports,
    _short_label,
    _shorten_path,
    _build_edges_ast,
    _find_chain,
    profile_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_fixtures():
    """Run cProfile on test_fixtures. Falls back to subprocess if another profiler is active."""
    # Disable any active profiler first
    sys.setprofile(None)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline, run_nested
        run_pipeline()
        run_nested()
    finally:
        profiler.disable()
    return pstats.Stats(profiler).stats


def _extract_graph_data(html: str) -> dict:
    """Extract the JSON data blob from graph HTML."""
    m = re.search(r'const D=(\{.*?\});', html, re.DOTALL)
    if not m:
        raise ValueError("No graph data found in HTML")
    return json.loads(m.group(1))


from toolboxv2 import tb_root_dir
PROJECT_ROOT = str(tb_root_dir / "utils" / "extras" / "code_analyzer")


# ═══════════════════════════════════════════════════════════════
#  Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestParseImports(unittest.TestCase):
    """Test AST-based import parsing."""

    def test_parse_real_file(self):
        """caller.py should have 'toolboxv2' as an import."""
        fixture = os.path.join(PROJECT_ROOT, "test_fixtures", "caller.py")
        imports = _parse_imports(fixture)
        self.assertIn("toolboxv2", imports)
        # Line numbers should be present
        self.assertIsInstance(imports["toolboxv2"], list)
        self.assertTrue(len(imports["toolboxv2"]) > 0)

    def test_parse_nonexistent_file(self):
        result = _parse_imports("/nonexistent/path.py")
        self.assertEqual(result, {})

    def test_parse_syntax_error(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(\n")
            f.flush()
            result = _parse_imports(f.name)
        os.unlink(f.name)
        self.assertEqual(result, {})


class TestGetPackageName(unittest.TestCase):
    """Test site-packages package name extraction."""

    def test_standard_package(self):
        self.assertEqual(
            _get_package_name("/usr/lib/python3.12/site-packages/requests/api.py"),
            "requests"
        )

    def test_versioned_package(self):
        self.assertEqual(
            _get_package_name("/venv/lib/site-packages/numpy-1.26.0/numpy/core.py"),
            "numpy"
        )

    def test_no_site_packages(self):
        self.assertIsNone(_get_package_name("/usr/lib/python3.12/os.py"))

    def test_windows_path(self):
        self.assertEqual(
            _get_package_name("C:\\Python\\site-packages\\flask\\app.py"),
            "flask"
        )


class TestShortLabel(unittest.TestCase):
    def test_project_file(self):
        root = "/home/user/project"
        label = _short_label("/home/user/project/src/mod/file.py", root)
        self.assertIn("file.py", label)
        self.assertNotIn("/home/user/project", label)

    def test_external_file(self):
        root = "/home/user/project"
        label = _short_label("/usr/lib/site-packages/requests/api.py", root)
        self.assertIn("requests", label)


class TestShortenPath(unittest.TestCase):
    def test_long_path(self):
        result = _shorten_path("/a/b/c/d/e/f.py", n=3)
        self.assertTrue(result.startswith(".../"))
        self.assertIn("f.py", result)

    def test_short_path(self):
        result = _shorten_path("a/b.py", n=3)
        self.assertEqual(result, "a/b.py")


class TestGroupByPackage(unittest.TestCase):
    def test_groups_exist(self):
        stats = _profile_fixtures()
        groups = _group_by_package(stats, PROJECT_ROOT, depth=2)
        self.assertIsInstance(groups, dict)
        self.assertTrue(len(groups) > 0)

        # Should have at least one group with nonzero cumtime
        has_nonzero = any(g["cumtime"] > 0 for g in groups.values())
        self.assertTrue(has_nonzero)

    def test_project_group_present(self):
        stats = _profile_fixtures()
        groups = _group_by_package(stats, PROJECT_ROOT, depth=2)
        # test_fixtures should appear as a group
        group_names = list(groups.keys())
        has_fixtures = any("test_fixtures" in name for name in group_names)
        self.assertTrue(has_fixtures, f"No test_fixtures group in {group_names}")


class TestFindChain(unittest.TestCase):
    def test_direct_dep(self):
        ext_deps = {"A": {"B"}, "B": {"C"}}
        chain = _find_chain("A", "B", ext_deps)
        self.assertEqual(chain, ["A"])

    def test_transitive_dep(self):
        ext_deps = {"A": {"B"}, "B": {"C"}}
        chain = _find_chain("A", "C", ext_deps)
        self.assertIn("A", chain)
        self.assertIn("B", chain)

    def test_no_path(self):
        ext_deps = {"A": {"B"}}
        chain = _find_chain("A", "Z", ext_deps)
        self.assertEqual(chain, ["A"])


class TestProfileCodeGuard(unittest.TestCase):
    """Test the PROFILING env var guard logic."""

    def test_override_true_enables(self):
        """override=True should enable profiling regardless of env."""
        decorator = profile_code(override=True, graph=False, function_graph=False)
        # Should NOT be the no-op lambda
        self.assertNotEqual(decorator.__name__ if hasattr(decorator, '__name__') else '', '<lambda>')

    def test_no_env_var_disables(self):
        """Without PROFILING=true and override=False, should return no-op."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove PROFILING if set
            os.environ.pop("PROFILING", None)
            decorator = profile_code(override=False, graph=False, function_graph=False)
            # Should be the identity lambda
            sentinel = object()
            self.assertIs(decorator(sentinel), sentinel)

    def test_env_var_true_enables(self):
        """PROFILING=true should enable profiling."""
        with patch.dict(os.environ, {"PROFILING": "true"}):
            decorator = profile_code(override=False, graph=False, function_graph=False)
            # Should NOT be identity lambda — it should be a real decorator
            self.assertNotEqual(decorator.__name__ if hasattr(decorator, '__name__') else '', '<lambda>')


# ═══════════════════════════════════════════════════════════════
#  Function Graph Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestBuildFunctionGraph(unittest.TestCase):
    """Unit tests for _build_function_graph data extraction."""

    @classmethod
    def setUpClass(cls):
        cls.stats = _profile_fixtures()
        cls.html = _build_function_graph(cls.stats, PROJECT_ROOT, min_time=0.0)
        cls.data = _extract_graph_data(cls.html)

    def test_returns_html(self):
        self.assertTrue(self.html.startswith("<!DOCTYPE html>"))

    def test_has_nodes(self):
        self.assertGreater(len(self.data["nodes"]), 0)

    def test_has_edges(self):
        self.assertGreater(len(self.data["edges"]), 0)

    def test_project_nodes_present(self):
        proj = [n for n in self.data["nodes"] if n["is_project"]]
        self.assertGreater(len(proj), 0, "No project nodes found")

    def test_project_functions_found(self):
        """compute, transform, aggregate, run_pipeline should be nodes."""
        funcs = {n["func"] for n in self.data["nodes"] if n["is_project"]}
        for expected in ["compute", "transform", "aggregate", "run_pipeline"]:
            self.assertIn(expected, funcs, f"{expected} not in project functions: {funcs}")

    def test_node_fields(self):
        """Every node must have all required fields."""
        required = {"id", "func", "label", "line", "ncalls", "tottime",
                    "cumtime", "avgtime", "is_project", "group"}
        for node in self.data["nodes"]:
            missing = required - set(node.keys())
            self.assertEqual(missing, set(), f"Node {node.get('func')} missing: {missing}")

    def test_edge_fields(self):
        required = {"source", "target", "calls", "caller_line"}
        for edge in self.data["edges"]:
            missing = required - set(edge.keys())
            self.assertEqual(missing, set(), f"Edge missing: {missing}")

    def test_edges_reference_valid_nodes(self):
        node_ids = {n["id"] for n in self.data["nodes"]}
        for edge in self.data["edges"]:
            self.assertIn(edge["source"], node_ids, f"Edge source {edge['source']} not in nodes")
            self.assertIn(edge["target"], node_ids, f"Edge target {edge['target']} not in nodes")

    def test_no_self_edges(self):
        for edge in self.data["edges"]:
            self.assertNotEqual(edge["source"], edge["target"])

    def test_call_counts_positive(self):
        for edge in self.data["edges"]:
            self.assertGreater(edge["calls"], 0)

    def test_time_values_non_negative(self):
        for node in self.data["nodes"]:
            self.assertGreaterEqual(node["tottime"], 0)
            self.assertGreaterEqual(node["cumtime"], 0)
            self.assertGreaterEqual(node["avgtime"], 0)

    def test_avgtime_le_tottime(self):
        """avg per call should be <= tottime (tottime = avg * ncalls)."""
        for node in self.data["nodes"]:
            if node["ncalls"] > 0:
                self.assertLessEqual(node["avgtime"], node["tottime"] + 1e-9)

    def test_tottime_le_cumtime(self):
        for node in self.data["nodes"]:
            self.assertLessEqual(node["tottime"], node["cumtime"] + 1e-9)

    def test_caller_edges_for_compute(self):
        """compute should be called by run_pipeline and run_nested."""
        compute_id = None
        for n in self.data["nodes"]:
            if n["func"] == "compute" and n["is_project"]:
                compute_id = n["id"]
                break
        self.assertIsNotNone(compute_id, "compute node not found")

        callers = [e for e in self.data["edges"] if e["target"] == compute_id]
        self.assertGreater(len(callers), 0, "compute has no incoming edges")
        total_calls = sum(e["calls"] for e in callers)
        self.assertEqual(total_calls, 30, f"compute should have 30 calls (20+10), got {total_calls}")

    def test_min_time_filter(self):
        """High min_time should reduce node count."""
        html_filtered = _build_function_graph(self.stats, PROJECT_ROOT, min_time=100.0)
        data_filtered = _extract_graph_data(html_filtered)
        self.assertLess(len(data_filtered["nodes"]), len(self.data["nodes"]))


# ═══════════════════════════════════════════════════════════════
#  Import Graph Unit Tests
# ═══════════════════════════════════════════════════════════════

class TestBuildImportGraph(unittest.TestCase):
    """Unit tests for _build_html_graph (import graph)."""

    @classmethod
    def setUpClass(cls):
        # Import graph needs <module> entries which only appear on first import.
        # Use a fresh cProfile session that imports something.
        profiler = cProfile.Profile()
        profiler.enable()
        import json as _fresh_json  # guaranteed to have <module> entry
        profiler.disable()
        cls.stats = pstats.Stats(profiler).stats
        cls.html = _build_html_graph(cls.stats, PROJECT_ROOT, min_time=0.0)
        cls.data = _extract_graph_data(cls.html)

    def test_returns_html(self):
        self.assertTrue(self.html.startswith("<!DOCTYPE html>"))

    def test_has_valid_structure(self):
        """Graph data should have nodes and edges lists."""
        self.assertIn("nodes", self.data)
        self.assertIn("edges", self.data)
        self.assertIsInstance(self.data["nodes"], list)
        self.assertIsInstance(self.data["edges"], list)

    def test_node_has_required_fields(self):
        required = {"id", "label", "cumtime", "is_project", "group"}
        for node in self.data["nodes"]:
            missing = required - set(node.keys())
            self.assertEqual(missing, set(), f"Import node missing: {missing}")


# ═══════════════════════════════════════════════════════════════
#  HTML Structure Tests
# ═══════════════════════════════════════════════════════════════

class TestFunctionGraphHTML(unittest.TestCase):
    """Test the HTML output structure of function graph."""

    @classmethod
    def setUpClass(cls):
        stats = _profile_fixtures()
        cls.html = _build_function_graph(stats, PROJECT_ROOT, min_time=0.0)

    def test_has_canvas(self):
        self.assertIn('<canvas id="cv"', self.html)

    def test_has_controls(self):
        self.assertIn('id="ms"', self.html)   # min-time slider
        self.assertIn('id="po"', self.html)   # project-only checkbox
        self.assertIn('id="hm"', self.html)   # hide-module checkbox

    def test_has_legend(self):
        self.assertIn("avg/call", self.html)
        self.assertIn("tottime", self.html)
        self.assertIn("cumtime", self.html)
        self.assertIn("Upstream", self.html)
        self.assertIn("Downstream", self.html)

    def test_has_detail_panel(self):
        self.assertIn('id="detail"', self.html)

    def test_has_tooltip(self):
        self.assertIn('id="tip"', self.html)

    def test_has_tbjs_glass_css(self):
        self.assertIn("--bg-base", self.html)
        self.assertIn("--glass-border", self.html)
        self.assertIn("IBM Plex", self.html)
        self.assertIn("oklch", self.html)

    def test_has_three_ring_rendering(self):
        """JS should render 3 concentric circles."""
        self.assertIn("rr.cum", self.html)
        self.assertIn("rr.tot", self.html)
        self.assertIn("rr.avg", self.html)

    def test_valid_json_data(self):
        """Embedded JSON should be parseable."""
        data = _extract_graph_data(self.html)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

    def test_node_count_in_stats_text(self):
        """Stats element should show node/edge count."""
        m = re.search(r'(\d+) nodes, (\d+) edges', self.html)
        self.assertIsNotNone(m, "Node/edge count not in HTML")


# ═══════════════════════════════════════════════════════════════
#  End-to-End Tests
# ═══════════════════════════════════════════════════════════════

class TestE2EDecoratorImportGraph(unittest.TestCase):
    """End-to-end: @profile_code writes import graph to disk."""

    def test_import_graph_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            graph_path = os.path.join(td, "import_graph.html")

            @profile_code(override=True, min_time=0.0,
                          graph=True, graph_file=graph_path,
                          function_graph=False)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline
                return run_pipeline()

            result = run()
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(graph_path),
                            f"Import graph not written to {graph_path}")
            html = Path(graph_path).read_text()
            data = _extract_graph_data(html)
            # Import graph may have 0 nodes if all modules already cached
            self.assertIn("nodes", data)
            self.assertIn("edges", data)


class TestE2EDecoratorFunctionGraph(unittest.TestCase):
    """End-to-end: @profile_code writes function graph to disk."""

    def test_function_graph_file_created(self):
        with tempfile.TemporaryDirectory() as td:
            fg_path = os.path.join(td, "function_graph.html")

            @profile_code(override=True, min_time=0.0,
                          graph=False,
                          function_graph=True, function_graph_file=fg_path)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline
                return run_pipeline()

            result = run()
            self.assertIsNotNone(result)
            self.assertTrue(os.path.exists(fg_path),
                            f"Function graph not written to {fg_path}")

            html = Path(fg_path).read_text()
            data = _extract_graph_data(html)

            # Verify project functions are in the graph
            proj_funcs = {n["func"] for n in data["nodes"] if n["is_project"]}
            self.assertIn("run_pipeline", proj_funcs,
                          f"run_pipeline not in {proj_funcs}")

    def test_both_graphs_produced(self):
        with tempfile.TemporaryDirectory() as td:
            ig_path = os.path.join(td, "imports.html")
            fg_path = os.path.join(td, "functions.html")

            @profile_code(override=True, min_time=0.0,
                          graph=True, graph_file=ig_path,
                          function_graph=True, function_graph_file=fg_path)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_nested
                return run_nested()

            run()
            self.assertTrue(os.path.exists(ig_path))
            self.assertTrue(os.path.exists(fg_path))


class TestE2EFunctionGraphDataIntegrity(unittest.TestCase):
    """End-to-end: verify the full data pipeline from profile → graph → JSON."""

    def test_call_chain_integrity(self):
        """Verify: run_pipeline → compute (20×), run_nested → compute (10×)."""
        with tempfile.TemporaryDirectory() as td:
            fg_path = os.path.join(td, "fg.html")

            @profile_code(override=True, min_time=0.0,
                          graph=False,
                          function_graph=True, function_graph_file=fg_path)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline, run_nested
                run_pipeline()
                run_nested()

            run()
            html = Path(fg_path).read_text()
            data = _extract_graph_data(html)

            # Find compute node
            compute_nodes = [n for n in data["nodes"]
                             if n["func"] == "compute" and n["is_project"]]
            self.assertEqual(len(compute_nodes), 1,
                             f"Expected 1 compute node, got {len(compute_nodes)}")
            compute = compute_nodes[0]

            # Check total calls
            self.assertEqual(compute["ncalls"], 30,
                             f"compute should have 30 calls, got {compute['ncalls']}")

            # Check edges: run_pipeline → compute and run_nested → compute
            incoming = [e for e in data["edges"] if e["target"] == compute["id"]]
            self.assertGreater(len(incoming), 0,
                               "compute should have callers")
            total_edge_calls = sum(e["calls"] for e in incoming)
            self.assertEqual(total_edge_calls, 30,
                             f"Total edge calls to compute should be 30, got {total_edge_calls}")

    def test_timing_consistency(self):
        """tottime <= cumtime, avgtime <= tottime for all nodes."""
        with tempfile.TemporaryDirectory() as td:
            fg_path = os.path.join(td, "fg.html")

            @profile_code(override=True, min_time=0.0,
                          graph=False,
                          function_graph=True, function_graph_file=fg_path)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline
                return run_pipeline()

            run()
            html = Path(fg_path).read_text()
            data = _extract_graph_data(html)

            for node in data["nodes"]:
                self.assertLessEqual(
                    node["tottime"], node["cumtime"] + 1e-9,
                    f"{node['func']}: tottime ({node['tottime']}) > cumtime ({node['cumtime']})"
                )
                if node["ncalls"] > 0:
                    self.assertLessEqual(
                        node["avgtime"], node["tottime"] + 1e-9,
                        f"{node['func']}: avgtime ({node['avgtime']}) > tottime ({node['tottime']})"
                    )

    def test_no_orphan_edges(self):
        """Every edge source/target must reference a valid node ID."""
        with tempfile.TemporaryDirectory() as td:
            fg_path = os.path.join(td, "fg.html")

            @profile_code(override=True, min_time=0.0,
                          graph=False,
                          function_graph=True, function_graph_file=fg_path)
            def run():
                from toolboxv2.utils.extras.code_analyzer.test_fixtures.caller import run_pipeline
                return run_pipeline()

            run()
            data = _extract_graph_data(Path(fg_path).read_text())
            node_ids = {n["id"] for n in data["nodes"]}
            for edge in data["edges"]:
                self.assertIn(edge["source"], node_ids)
                self.assertIn(edge["target"], node_ids)


class TestE2ENoOpWhenDisabled(unittest.TestCase):
    """When PROFILING env is not set, decorator should be transparent."""

    def test_noop_returns_result(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("PROFILING", None)

            @profile_code(override=False, graph=False, function_graph=False)
            def add(a, b):
                return a + b

            self.assertEqual(add(2, 3), 5)


if __name__ == "__main__":
    unittest.main()
