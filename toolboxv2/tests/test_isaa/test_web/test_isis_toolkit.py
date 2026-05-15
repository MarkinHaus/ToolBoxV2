"""
Tests for IsisToolkit.

All WebAgentToolkit interactions are mocked — no browser, no Playwright needed.
Tests verify correct tool routing, JS generation, login flow logic, and tool export.
"""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch



class MockTool:
    """Mocks a ToolDefinition with an async func."""
    def __init__(self, name):
        self.name = name
        self.func = AsyncMock(return_value={})


class MockToolkit:
    """Mocks WebAgentToolkit with tool lookup."""
    def __init__(self):
        self._tool_map = {}
        for name in [
            "goto", "click", "type", "wait", "current_url",
            "screenshot", "scroll", "execute_js",
            "session_save", "session_load",
        ]:
            self._tool_map[name] = MockTool(name)

    def get_tool(self, name):
        return self._tool_map.get(name)

    async def start_browser(self):
        pass

    async def stop_browser(self):
        pass

    def get_tools(self):
        return []


def _make_isis(mock_tk=None):
    """Create IsisToolkit with mocked WebAgentToolkit — no real imports."""
    if mock_tk is None:
        mock_tk = MockToolkit()

    from toolboxv2.mods.isaa.extras.toolkit.isis_toolkit import IsisToolkit
    isis = IsisToolkit.__new__(IsisToolkit)
    isis._tk = mock_tk
    isis._state_name = "isis_session"
    isis._started = False
    return isis, mock_tk


class TestHelperRouting(unittest.TestCase):
    """Verify helpers call correct toolkit tools."""

    def test_goto_calls_toolkit(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://isis.tu-berlin.de/my/", "title": "Dashboard"}

        async def _run():
            result = await isis._goto("https://isis.tu-berlin.de/my/")
            tk.get_tool("goto").func.assert_called_once_with(
                url="https://isis.tu-berlin.de/my/", wait_until="domcontentloaded"
            )
            self.assertEqual(result["url"], "https://isis.tu-berlin.de/my/")

        asyncio.run(_run())

    def test_click_calls_toolkit(self):
        isis, tk = _make_isis()

        async def _run():
            await isis._click("#submit")
            tk.get_tool("click").func.assert_called_once_with(selector="#submit")

        asyncio.run(_run())

    def test_type_calls_toolkit(self):
        isis, tk = _make_isis()

        async def _run():
            await isis._type("input#user", "testuser")
            tk.get_tool("type").func.assert_called_once_with(
                selector="input#user", text="testuser"
            )

        asyncio.run(_run())

    def test_js_extracts_result(self):
        isis, tk = _make_isis()
        tk.get_tool("execute_js").func.return_value = {"result": 42}

        async def _run():
            r = await isis._js("return 42")
            self.assertEqual(r, 42)

        asyncio.run(_run())

    def test_url_returns_string(self):
        isis, tk = _make_isis()
        tk.get_tool("current_url").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}

        async def _run():
            url = await isis._url()
            self.assertEqual(url, "https://isis.tu-berlin.de/my/")

        asyncio.run(_run())


class TestFindFirstVisible(unittest.TestCase):
    """Selector discovery logic."""

    def test_returns_first_visible(self):
        isis, tk = _make_isis()

        call_count = 0
        async def mock_js(script):
            nonlocal call_count
            call_count += 1
            # First selector invisible, second visible
            return {"result": call_count == 2}

        tk.get_tool("execute_js").func.side_effect = mock_js

        async def _run():
            sel = await isis._find_first_visible(["input#a", "input#b", "input#c"])
            self.assertEqual(sel, "input#b")

        asyncio.run(_run())

    def test_returns_none_if_none_visible(self):
        isis, tk = _make_isis()
        tk.get_tool("execute_js").func.return_value = {"result": False}

        async def _run():
            sel = await isis._find_first_visible(["input#a", "input#b"])
            self.assertIsNone(sel)

        asyncio.run(_run())


class TestSessionValid(unittest.TestCase):
    """Session validation logic."""

    def test_valid_session(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("current_url").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("wait").func.return_value = {"status": "ready"}
        # usermenu count = 1
        tk.get_tool("execute_js").func.return_value = {"result": 1}

        async def _run():
            valid = await isis._session_valid()
            self.assertTrue(valid)

        asyncio.run(_run())

    def test_expired_session_shibboleth_redirect(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://shibboleth.tu-berlin.de/idp/..."}
        tk.get_tool("current_url").func.return_value = {"url": "https://shibboleth.tu-berlin.de/idp/..."}
        tk.get_tool("wait").func.side_effect = Exception("timeout")

        async def _run():
            valid = await isis._session_valid()
            self.assertFalse(valid)

        asyncio.run(_run())

    def test_expired_session_login_redirect(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://isis.tu-berlin.de/login/index.php"}
        tk.get_tool("current_url").func.return_value = {"url": "https://isis.tu-berlin.de/login/index.php"}
        tk.get_tool("wait").func.side_effect = Exception("timeout")

        async def _run():
            valid = await isis._session_valid()
            self.assertFalse(valid)

        asyncio.run(_run())


class TestLoginFlow(unittest.TestCase):
    """Login orchestration."""

    def test_login_raises_without_credentials(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaises(ValueError):
                    await isis.login()

        asyncio.run(_run())

    def test_login_calls_type_click_sequence(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://shibboleth..."}
        tk.get_tool("current_url").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("wait").func.return_value = {"status": "ready"}

        # _find_first_visible: always return True
        call_idx = [0]
        async def mock_js(script):
            call_idx[0] += 1
            # First 3 calls: find_first_visible (return True for all)
            # Then consent check (isis in url)
            # Then session_valid check (usermenu count)
            if call_idx[0] <= 3:
                return {"result": True}
            if call_idx[0] == 4:
                return {"result": 1}  # usermenu count
            return {"result": None}

        tk.get_tool("execute_js").func.side_effect = mock_js

        async def _run():
            with patch.dict("os.environ", {"ISIS_USERNAME": "user", "ISIS_PASSWORD": "pass"}):
                result = await isis.login()

            # Verify type was called for username + password
            type_calls = tk.get_tool("type").func.call_args_list
            self.assertEqual(len(type_calls), 2)
            # Verify click was called for submit
            tk.get_tool("click").func.assert_called()

        asyncio.run(_run())


class TestStartLifecycle(unittest.TestCase):
    """Start/stop lifecycle."""

    def test_start_with_cached_session(self):
        isis, tk = _make_isis()
        tk.get_tool("session_load").func.return_value = {"success": True}
        tk.get_tool("goto").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("current_url").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("wait").func.return_value = {"status": "ready"}
        tk.get_tool("execute_js").func.return_value = {"result": 1}  # usermenu

        async def _run():
            tk.start_browser = AsyncMock()
            result = await isis.start()
            self.assertEqual(result["login"], "cached")
            self.assertTrue(isis._started)

        asyncio.run(_run())

    def test_stop(self):
        isis, tk = _make_isis()
        isis._started = True

        async def _run():
            tk.stop_browser = AsyncMock()
            result = await isis.stop()
            self.assertEqual(result["status"], "stopped")
            self.assertFalse(isis._started)

        asyncio.run(_run())


class TestToolExport(unittest.TestCase):
    """Tool export compatibility."""

    def test_exports_all_tools(self):
        isis, tk = _make_isis()
        tools = isis.get_tools()

        names = [t["name"] for t in tools]
        expected = [
            "isis_start", "isis_stop", "isis_list_courses", "isis_list_chat",
            "isis_course_overview", "isis_course_activities",
            "isis_course_sections_md", "isis_course_shallow",
            "isis_course_deep", "isis_scrape_activity",
        ]
        self.assertEqual(names, expected)

    def test_tool_dict_structure(self):
        isis, tk = _make_isis()
        tools = isis.get_tools()

        for t in tools:
            self.assertIn("name", t)
            self.assertIn("description", t)
            self.assertIn("tool_func", t)
            self.assertIn("category", t)
            self.assertEqual(t["category"], "isis")
            self.assertEqual(t["source"], "isis-toolkit")
            self.assertTrue(callable(t["tool_func"]))

    def test_tool_funcs_are_bound_methods(self):
        isis, tk = _make_isis()
        tools = isis.get_tools()

        # isis_list_courses should point to isis.list_courses
        courses_tool = next(t for t in tools if t["name"] == "isis_list_courses")
        # Bound methods create new objects each access, compare __func__ + __self__
        self.assertIs(courses_tool["tool_func"].__func__, isis.list_courses.__func__)
        self.assertIs(courses_tool["tool_func"].__self__, isis)


class TestScrapeActivityDispatch(unittest.TestCase):
    """Type-based dispatcher routing."""

    def test_dispatches_to_forum(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.object(isis, "_scrape_forum", new_callable=AsyncMock) as mock:
                mock.return_value = {"type": "forum", "threads": []}
                result = await isis.scrape_activity({"url": "http://x", "type": "forum"})
                mock.assert_called_once_with("http://x")
                self.assertTrue(result["scraped"])

        asyncio.run(_run())

    def test_dispatches_to_assign(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.object(isis, "_scrape_assign", new_callable=AsyncMock) as mock:
                mock.return_value = {"type": "assign", "dates": {}}
                result = await isis.scrape_activity({"url": "http://x", "type": "assign"})
                mock.assert_called_once_with("http://x")

        asyncio.run(_run())

    def test_unknown_type_uses_generic(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.object(isis, "_scrape_generic", new_callable=AsyncMock) as mock:
                mock.return_value = {"type": "unknown_thing", "markdown": "..."}
                result = await isis.scrape_activity({"url": "http://x", "type": "unknown_thing"})
                mock.assert_called_once_with("http://x")

        asyncio.run(_run())

    def test_error_handling(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.object(isis, "_scrape_forum", new_callable=AsyncMock) as mock:
                mock.side_effect = RuntimeError("connection lost")
                result = await isis.scrape_activity({"url": "http://x", "type": "forum"})
                self.assertFalse(result["scraped"])
                self.assertIn("connection lost", result["error"])

        asyncio.run(_run())


class TestJsExtractMd(unittest.TestCase):
    """JS generation for markdown extraction."""

    def test_js_contains_nodeToMd(self):
        from toolboxv2.mods.isaa.extras.toolkit.isis_toolkit import _js_extract_md
        js = _js_extract_md("#region-main")
        self.assertIn("nodeToMd", js)
        self.assertIn("cleanClone", js)
        self.assertIn("#region-main", js)

    def test_js_remove_nav_flag(self):
        from toolboxv2.mods.isaa.extras.toolkit.isis_toolkit import _js_extract_md
        js_no_nav = _js_extract_md("body", remove_nav=False)
        js_with_nav = _js_extract_md("body", remove_nav=True)
        self.assertIn("false", js_no_nav)
        self.assertIn("true", js_with_nav)


class TestListCourses(unittest.TestCase):
    """Course listing via execute_js."""

    def test_returns_courses(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "https://isis.tu-berlin.de/my/"}
        tk.get_tool("wait").func.return_value = {"status": "ready"}
        tk.get_tool("execute_js").func.return_value = {"result": [
            {"id": "123", "title": "Logik WS25", "url": "https://isis.tu-berlin.de/course/view.php?id=123"},
            {"id": "456", "title": "AI SS26", "url": "https://isis.tu-berlin.de/course/view.php?id=456"},
        ]}

        async def _run():
            courses = await isis.list_courses()
            self.assertEqual(len(courses), 2)
            self.assertEqual(courses[0]["id"], "123")
            self.assertEqual(courses[1]["title"], "AI SS26")

        asyncio.run(_run())

    def test_returns_empty_on_no_courses(self):
        isis, tk = _make_isis()
        tk.get_tool("goto").func.return_value = {"url": "..."}
        tk.get_tool("wait").func.side_effect = Exception("timeout")
        tk.get_tool("execute_js").func.return_value = {"result": None}

        async def _run():
            courses = await isis.list_courses()
            self.assertEqual(courses, [])

        asyncio.run(_run())


class TestShallowScrape(unittest.TestCase):
    """Shallow scrape orchestration."""

    def test_combines_overview_and_activities(self):
        isis, tk = _make_isis()

        async def _run():
            with patch.object(isis, "get_course_overview", new_callable=AsyncMock) as mock_ov, \
                 patch.object(isis, "get_course_activities", new_callable=AsyncMock) as mock_act:
                mock_ov.return_value = {
                    "id": "123", "url": "...", "title": "Test Course",
                    "sections": [{"name": "S1", "activity_count": 2}],
                }
                mock_act.return_value = [
                    {"name": "A1", "type": "forum", "url": "..."},
                    {"name": "A2", "type": "assign", "url": "..."},
                ]

                result = await isis.scrape_course_shallow("123")
                self.assertEqual(result["title"], "Test Course")
                self.assertEqual(result["activity_count"], 2)
                self.assertEqual(result["type_counts"], {"forum": 1, "assign": 1})

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
