"""
Validate TB Language documentation consistency
Checks that all documentation is consistent with code changes

Version: 1.0.1
Last Updated: 2025-11-10
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


class DocumentationValidator:
    """Validates documentation consistency"""

    def __init__(self):
        self.root = Path(__file__).parent.parent.parent.parent
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []

    def check_file_extensions_documented(self) -> bool:
        """Check that both .tbx and .tb extensions are documented"""
        print("\n📄 Checking file extension documentation...")
        
        # Check Lang.md
        lang_md = self.root / "tb-exc" / "src" / "Lang.md"
        if lang_md.exists():
            content = lang_md.read_text(encoding='utf-8')
            if ".tbx" in content and ".tb" in content:
                self.successes.append("✓ Lang.md documents both .tbx and .tb extensions")
            else:
                self.errors.append("✗ Lang.md missing .tbx or .tb extension documentation")
        else:
            self.warnings.append("⚠ Lang.md not found")

        # Check development guide
        dev_guide = self.root / "tb-exc" / "TB_LANG_DEVELOPMENT_GUIDE.md"
        if dev_guide.exists():
            content = dev_guide.read_text(encoding='utf-8')
            if ".tbx" in content or ".tb" in content:
                self.successes.append("✓ Development guide mentions file extensions")
            else:
                self.warnings.append("⚠ Development guide doesn't mention file extensions")
        else:
            self.warnings.append("⚠ Development guide not found")

        return len(self.errors) == 0

    def check_comment_syntax_documented(self) -> bool:
        """Check that comment syntax is correctly documented"""
        print("\n💬 Checking comment syntax documentation...")
        
        lang_md = self.root / "tb-exc" / "src" / "Lang.md"
        if lang_md.exists():
            content = lang_md.read_text(encoding='utf-8')
            
            # Check for correct comment syntax
            if "//" in content and "/*" in content:
                self.successes.append("✓ Lang.md documents C-style comments (// and /* */)")
            else:
                self.errors.append("✗ Lang.md missing comment syntax documentation")
            
            # Make sure old # syntax is not documented as valid
            if "# comment" in content.lower() or "#comment" in content.lower():
                self.warnings.append("⚠ Lang.md may reference # comments (should be //)")
        else:
            self.errors.append("✗ Lang.md not found")

        return len(self.errors) == 0

    def check_version_consistency(self) -> bool:
        """Check that version numbers are consistent"""
        print("\n🔢 Checking version consistency...")
        
        files_to_check = [
            ("setup.py", self.root / "utils" / "tbx" / "setup.py"),
            ("install_support.py", self.root / "utils" / "tbx" / "install_support.py"),
            ("package.json", self.root / "utils" / "tbx" / "tb-lang-support" / "package.json"),
            ("plugin.xml", self.root / "utils" / "tbx" / "tb-lang-pycharm" / "src" / "main" / "resources" / "META-INF" / "plugin.xml"),
        ]
        
        versions = {}
        for name, path in files_to_check:
            if path.exists():
                content = path.read_text(encoding='utf-8')
                
                # Extract version
                if "1.0.1" in content:
                    versions[name] = "1.0.1"
                    self.successes.append(f"✓ {name} has version 1.0.1")
                elif "1.0.0" in content:
                    versions[name] = "1.0.0"
                    self.warnings.append(f"⚠ {name} still at version 1.0.0")
                else:
                    self.warnings.append(f"⚠ {name} version not found")
            else:
                self.warnings.append(f"⚠ {name} not found at {path}")

        return True

    def check_execution_modes_documented(self) -> bool:
        """Check that JIT and AOT execution modes are documented"""
        print("\n⚡ Checking execution mode documentation...")
        
        lang_md = self.root / "tb-exc" / "src" / "Lang.md"
        if lang_md.exists():
            content = lang_md.read_text(encoding='utf-8')
            
            if "JIT" in content or "jit" in content:
                self.successes.append("✓ Lang.md documents JIT mode")
            else:
                self.warnings.append("⚠ Lang.md doesn't mention JIT mode")
            
            if "AOT" in content or "aot" in content or "Ahead-Of-Time" in content:
                self.successes.append("✓ Lang.md documents AOT mode")
            else:
                self.warnings.append("⚠ Lang.md doesn't mention AOT mode")
        else:
            self.errors.append("✗ Lang.md not found")

        return len(self.errors) == 0

    def check_keywords_consistency(self) -> bool:
        """Check that keywords are consistent across implementations"""
        print("\n🔑 Checking keyword consistency...")
        
        # Expected keywords from Lang.md
        expected_keywords = {
            "fn", "let", "if", "else", "while", "for", "in", 
            "return", "break", "continue", "match", "true", "false",
            "and", "or", "not"
        }
        
        # Check VS Code syntax file
        vscode_syntax = self.root / "utils" / "tbx" / "tb-lang-support" / "syntaxes" / "tb.tmLanguage.json"
        if vscode_syntax.exists():
            content = vscode_syntax.read_text(encoding='utf-8')
            missing = []
            for keyword in expected_keywords:
                if keyword not in content:
                    missing.append(keyword)
            
            if not missing:
                self.successes.append("✓ VS Code syntax file has all keywords")
            else:
                self.warnings.append(f"⚠ VS Code syntax missing keywords: {', '.join(missing)}")
        else:
            self.errors.append("✗ VS Code syntax file not found")

        # Check PyCharm file type
        pycharm_filetype = self.root / "utils" / "tbx" / "tb-lang-pycharm" / "src" / "main" / "resources" / "fileTypes" / "TB.xml"
        if pycharm_filetype.exists():
            content = pycharm_filetype.read_text(encoding='utf-8')
            missing = []
            for keyword in expected_keywords:
                if keyword not in content:
                    missing.append(keyword)
            
            if not missing:
                self.successes.append("✓ PyCharm file type has all keywords")
            else:
                self.warnings.append(f"⚠ PyCharm file type missing keywords: {', '.join(missing)}")
        else:
            self.errors.append("✗ PyCharm file type not found")

        return len(self.errors) == 0

    def run_all_checks(self) -> bool:
        """Run all validation checks"""
        print("=" * 70)
        print("TB Language Documentation Validation")
        print("=" * 70)

        checks = [
            self.check_file_extensions_documented,
            self.check_comment_syntax_documented,
            self.check_version_consistency,
            self.check_execution_modes_documented,
            self.check_keywords_consistency,
        ]

        for check in checks:
            check()

        # Print results
        print("\n" + "=" * 70)
        print("Validation Results")
        print("=" * 70)

        if self.successes:
            print(f"\n{GREEN}Successes ({len(self.successes)}):{RESET}")
            for success in self.successes:
                print(f"  {success}")

        if self.warnings:
            print(f"\n{YELLOW}Warnings ({len(self.warnings)}):{RESET}")
            for warning in self.warnings:
                print(f"  {warning}")

        if self.errors:
            print(f"\n{RED}Errors ({len(self.errors)}):{RESET}")
            for error in self.errors:
                print(f"  {error}")

        print("\n" + "=" * 70)
        
        if self.errors:
            print(f"{RED}❌ Validation FAILED with {len(self.errors)} error(s){RESET}")
            return False
        elif self.warnings:
            print(f"{YELLOW}⚠️  Validation PASSED with {len(self.warnings)} warning(s){RESET}")
            return True
        else:
            print(f"{GREEN}✅ Validation PASSED - All checks successful!{RESET}")
            return True


if __name__ == "__main__":
    validator = DocumentationValidator()
    success = validator.run_all_checks()
    sys.exit(0 if success else 1)

