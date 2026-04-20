# tests/test_coder_apply_back_hypotheses.py
"""
Isolated tests to confirm/refute hypotheses about the
'changes vanish on /coder accept' bug.

Each test maps to ONE hypothesis from the analysis.
No mocks of git itself — we run real git in temp dirs.
"""
import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from toolboxv2.mods.isaa.CodingAgent.coder import GitWorktree


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if not asyncio.get_event_loop().is_running() \
        else asyncio.run(coro)


def _git(cwd, *args, check=True):
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr}")
    return r


def _init_repo(path: Path, with_subfolder: bool = False) -> Path:
    """Init a git repo. If with_subfolder, return path/sub instead of path."""
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.email", "t@t.com")
    _git(path, "config", "user.name", "T")
    (path / "root_file.py").write_text("root")
    if with_subfolder:
        sub = path / "sub"
        sub.mkdir()
        (sub / "main.py").write_text("original")
        _git(path, "add", ".")
        _git(path, "commit", "-m", "init")
        return sub
    (path / "main.py").write_text("original")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "init")
    return path


# =============================================================================
# H1: changed_files() empty after commit (git mode)
# =============================================================================

class TestH1_ChangedFilesAfterCommit(unittest.TestCase):
    """
    Hypothesis: After self.worktree.commit(...) in execute(), the worktree
    has no diff against HEAD, so changed_files() returns []. This means
    result.changed_files is always [] in git-mode after a successful task.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        _init_repo(self.tmp)
        self.wt = GitWorktree(str(self.tmp))
        self.wt.setup()

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_changed_files_nonempty_before_commit(self):
        """Sanity: pre-commit, changes ARE detected."""
        (self.wt.path / "main.py").write_text("modified")
        (self.wt.path / "newfile.py").write_text("new")
        changed = run(self.wt.changed_files())
        self.assertIn("main.py", changed)
        self.assertIn("newfile.py", changed)

    def test_changed_files_empty_after_commit(self):
        """H1 CONFIRMED if this passes: post-commit returns []."""
        (self.wt.path / "main.py").write_text("modified")
        (self.wt.path / "newfile.py").write_text("new")
        ok = self.wt.commit("test")
        self.assertTrue(ok)
        changed = run(self.wt.changed_files())
        # If H1 is true: changed == []
        # If H1 is false: changed contains files
        self.assertEqual(
            changed, [],
            f"H1 REFUTED: changed_files returned {changed} after commit"
        )

    def test_branch_has_new_commit(self):
        """Verify the commit actually went to the coder-branch."""
        (self.wt.path / "main.py").write_text("modified via worktree")
        self.wt.commit("worktree-edit")
        log = _git(self.wt.path, "log", "--oneline").stdout
        self.assertIn("worktree-edit", log)


# =============================================================================
# H2: _is_git=True for subfolder (origin_root != git_root)
# =============================================================================

class TestH2_IsGitForSubfolder(unittest.TestCase):
    """
    Hypothesis: GitWorktree._detect_git() returns _is_git=True when the
    origin is a subfolder of a git repo (via parent walk), AND _git_root
    points to the actual git root. setup() then must enter subfolder mode.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        self.subfolder = _init_repo(self.tmp, with_subfolder=True)
        self.wt = GitWorktree(str(self.subfolder))

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detect_git_returns_true_for_subfolder(self):
        """H2 part A: _is_git is True even though origin != git_root."""
        self.assertTrue(self.wt._is_git)
        self.assertIsNotNone(self.wt._git_root)
        self.assertEqual(
            self.wt._git_root.resolve(), self.tmp.resolve(),
            "git_root should be the actual repo root, not the subfolder"
        )
        self.assertNotEqual(
            self.wt.origin_root.resolve(), self.wt._git_root.resolve(),
            "origin_root must differ from git_root in subfolder mode"
        )

    def test_setup_enters_subfolder_mode(self):
        """H2 part B: setup() detects mismatch and switches to copy mode."""
        self.wt.setup()
        # In subfolder mode, _is_git should be flipped to False
        self.assertTrue(
            self.wt._is_subfolder_mode,
            "Subfolder mode flag must be set"
        )
        self.assertFalse(
            self.wt._is_git,
            "_is_git must be False after subfolder setup (copy mode)"
        )


# =============================================================================
# H3: apply_back in subfolder mode — does it land changes in the right place?
# =============================================================================

class TestH3_ApplyBackSubfolder(unittest.TestCase):
    """
    Hypothesis: When origin is a subfolder of a git repo, apply_back()
    correctly copies files back to the SUBFOLDER, not the git root.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        self.subfolder = _init_repo(self.tmp, with_subfolder=True)
        self.wt = GitWorktree(str(self.subfolder))
        self.wt.setup()

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_back_modifies_subfolder_file(self):
        """Modified file should appear in subfolder, NOT in git root."""
        (self.wt.path / "main.py").write_text("modified in worktree")
        n = run(self.wt.apply_back())
        # subfolder/main.py should be updated
        self.assertEqual(
            (self.subfolder / "main.py").read_text(),
            "modified in worktree",
            f"Subfolder file not updated. apply_back returned {n}"
        )
        # root_file.py outside subfolder must NOT be touched
        self.assertEqual((self.tmp / "root_file.py").read_text(), "root")

    def test_apply_back_creates_new_subfolder_file(self):
        """New file from worktree must land in subfolder."""
        (self.wt.path / "newfile.py").write_text("new content")
        n = run(self.wt.apply_back())
        self.assertTrue(
            (self.subfolder / "newfile.py").exists(),
            f"New file not created in subfolder. apply_back returned {n}"
        )
        self.assertFalse(
            (self.tmp / "newfile.py").exists(),
            "New file leaked into git root"
        )


# =============================================================================
# H4: apply_back falls back to copy when branch has no diff vs HEAD
# =============================================================================

class TestH4_ApplyBackFallback(unittest.TestCase):
    """
    Hypothesis: When apply_back's diff_check finds no diff between HEAD
    and the coder-branch (because nothing was committed to it), it falls
    back to _apply_back_copy() — and THAT is where the actual file
    propagation happens.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        _init_repo(self.tmp)
        self.wt = GitWorktree(str(self.tmp))
        self.wt.setup()

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_uncommitted_changes_apply_via_fallback(self):
        """If files were edited but never committed, apply_back must still
        propagate them via the copy fallback (after pre-merge checkpoint)."""
        (self.wt.path / "main.py").write_text("uncommitted edit")
        (self.wt.path / "added.py").write_text("uncommitted new")
        n = run(self.wt.apply_back())

        # apply_back creates a 'pre-merge checkpoint' commit, so HEAD on
        # branch DOES diverge from main → merge path is taken (n == -1).
        # If H4 holds: either path (-1 or copy) must propagate files.
        self.assertEqual((self.tmp / "main.py").read_text(), "uncommitted edit",
                         f"main.py not propagated. n={n}")
        self.assertTrue((self.tmp / "added.py").exists(),
                        f"added.py not propagated. n={n}")

    def test_explicit_no_commit_no_changes_returns_zero(self):
        """No edits at all → apply_back is a no-op."""
        n = run(self.wt.apply_back())
        # Either git merge with empty branch (-1) or copy with no diffs (0)
        self.assertIn(n, (0, -1), f"Expected 0 or -1, got {n}")
        self.assertEqual((self.tmp / "main.py").read_text(), "original")


# =============================================================================
# H5: Files lost when commit fails to stage anything
# =============================================================================

class TestH5_CommitStagingFailure(unittest.TestCase):
    """
    Hypothesis: commit() returns True even when 'staged_files' is empty,
    masking the case where files were never actually committed. Then
    apply_back's diff_check sees no diff and the fallback must save us.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        _init_repo(self.tmp)
        # Add a .gitignore that EXCLUDES newly-created files
        (self.tmp / ".gitignore").write_text("ignored_*.py\n")
        _git(self.tmp, "add", ".gitignore")
        _git(self.tmp, "commit", "-m", "ignore")
        self.wt = GitWorktree(str(self.tmp))
        self.wt.setup()

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ignored_file_not_committed_but_still_in_worktree(self):
        """Gitignored file: commit() reports success but file is NOT staged."""
        (self.wt.path / "ignored_secret.py").write_text("secret")
        ok = self.wt.commit("try ignore")
        self.assertTrue(ok)  # commit returns True
        # But the file is not in HEAD
        log = _git(self.wt.path, "ls-tree", "HEAD", "ignored_secret.py").stdout
        self.assertEqual(log.strip(), "", "Ignored file should not be in HEAD")

    def test_ignored_file_NOT_propagated_by_apply_back(self):
        """If commit silently dropped the file AND apply_back uses git merge,
        the file vanishes. This confirms a real failure mode."""
        (self.wt.path / "ignored_secret.py").write_text("secret")
        n = run(self.wt.apply_back())
        # Expect: file is LOST (because git merge can't see it, and
        # _apply_back_copy uses git ls-files which excludes it too).
        propagated = (self.tmp / "ignored_secret.py").exists()
        # We're not asserting either direction — we're DOCUMENTING behavior.
        # If propagated=False, H5 is a real bug surface.
        self.assertEqual(
            propagated, False,
            f"H5 status: propagated={propagated}, n={n}. "
            "If True, ignored files DO survive (good). "
            "If False, gitignore'd files are silently dropped (bug surface)."
        )


# =============================================================================
# H6: _apply_back_copy in subfolder mode lists files correctly
# =============================================================================

class TestH6_ApplyBackCopySubfolderListing(unittest.TestCase):
    """
    Hypothesis: _list_tracked_files in subfolder+git mode runs
    `git ls-files` in _git_root and filters by origin_root prefix.
    Verify this filter actually picks up worktree-created files.
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        self.subfolder = _init_repo(self.tmp, with_subfolder=True)
        self.wt = GitWorktree(str(self.subfolder))
        self.wt.setup()  # forces subfolder mode (copy)

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_tracked_files_in_worktree_finds_new_file(self):
        """New file in worktree must appear in _list_tracked_files."""
        (self.wt.path / "fresh.py").write_text("fresh")
        files = self.wt._list_tracked_files(self.wt.path)
        names = [f.name for f in files]
        self.assertIn(
            "fresh.py", names,
            f"fresh.py missing from listing: {names}"
        )

    def test_subfolder_apply_back_propagates_new_nested_file(self):
        """Nested new file in worktree must reach the subfolder."""
        nested = self.wt.path / "nested" / "deep.py"
        nested.parent.mkdir(parents=True, exist_ok=True)
        nested.write_text("deep")
        n = run(self.wt.apply_back())
        target = self.subfolder / "nested" / "deep.py"
        self.assertTrue(
            target.exists(),
            f"Nested file not propagated. apply_back={n}, "
            f"subfolder contents: {list(self.subfolder.rglob('*'))}"
        )


# =============================================================================
# H7: accept-flow state — cleanup right after apply_back
# =============================================================================

class TestH7_AcceptFlowCleanupRace(unittest.TestCase):
    """
    Hypothesis: The CLI does
        n = await wt.apply_back()
        wt.cleanup()
        wt.setup()
    Verify that apply_back fully completes (files on disk) BEFORE
    cleanup runs. Also verify that the second setup() starts fresh
    (new branch, new temp dir, no leftover state).
    """

    def setUp(self):
        if not shutil.which("git"):
            self.skipTest("git missing")
        self.tmp = Path(tempfile.mkdtemp())
        _init_repo(self.tmp)
        self.wt = GitWorktree(str(self.tmp))
        self.wt.setup()

    def tearDown(self):
        try: self.wt.cleanup()
        except Exception: pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_apply_back_persists_before_cleanup(self):
        """Files must already be on origin disk before cleanup wipes worktree."""
        (self.wt.path / "main.py").write_text("survive cleanup")
        n = run(self.wt.apply_back())

        # Snapshot origin BEFORE cleanup
        before = (self.tmp / "main.py").read_text()
        self.assertEqual(before, "survive cleanup", f"apply_back failed pre-cleanup, n={n}")

        old_path = self.wt.path
        self.wt.cleanup()
        self.assertIsNone(self.wt.path)

        # Origin must still have the change
        after = (self.tmp / "main.py").read_text()
        self.assertEqual(after, "survive cleanup", "Cleanup wiped origin change")

    def test_setup_after_cleanup_creates_fresh_worktree(self):
        """Second setup() after cleanup must NOT resume old (deleted) worktree."""
        old_branch = self.wt._branch
        old_path = self.wt.path
        self.wt.cleanup()

        # state_file may still exist with stale path
        state_file = self.tmp / ".coder_worktree.json"
        if state_file.exists():
            stale = json.loads(state_file.read_text()).get("worktree_path", "")
            self.assertEqual(
                stale, str(old_path.resolve()),
                "State file should still point at deleted path (resume guard relies on .exists())"
            )

        self.wt.setup()
        self.assertIsNotNone(self.wt.path)
        self.assertNotEqual(
            self.wt.path, old_path,
            "Second setup must create a NEW temp dir (old one was deleted)"
        )


if __name__ == "__main__":
    unittest.main()
