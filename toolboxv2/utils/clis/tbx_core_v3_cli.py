"""
TB Lang Core Runtime v3.0.0 - CLI Interface
Main entry point for TB Lang Core Runtime with Server Plugin Management
"""

import argparse
import os
import platform
import sys
import subprocess
import json
import time
import signal
import shutil
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

# Directories
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
CORE_DIR = WORKSPACE_ROOT / "toolboxv2" / "utils" / "tbx" / "core"
TESTS_DIR = CORE_DIR / "tests"
TB_EXC_DIR = WORKSPACE_ROOT / "toolboxv2" / "tb-exc"
TBX_EXECUTABLE = WORKSPACE_ROOT / "toolboxv2" / "bin" / ("tbx.exe" if platform.system() == "Windows" else "tbx")
SERVER_PLUGIN_DIR = TB_EXC_DIR / "src" / "builtin-plugins" / "server"
DIST_DIR = WORKSPACE_ROOT / "toolboxv2" / "dist"
BIN_DIR = WORKSPACE_ROOT / "toolboxv2" / "bin"
CORE_EXECUTABLE_NAME = "tb-core" + (".exe" if platform.system() == "Windows" else "")


class TBXCoreManager:
    """Manager for TB Lang Core Runtime v3.0.0"""

    def __init__(self):
        self.workspace_root = WORKSPACE_ROOT
        self.core_dir = CORE_DIR
        self.tests_dir = TESTS_DIR
        self.tb_exc_dir = TB_EXC_DIR
        self.tbx_executable = TBX_EXECUTABLE
        self.server_plugin_dir = SERVER_PLUGIN_DIR
        self.dist_dir = DIST_DIR
        self.main_tbx = self.core_dir / "main.tbx"
        self.config_file = self.core_dir / "config.json"
        self.state_file = self.core_dir / ".state.json"
        self.server_lib = self.server_plugin_dir / "target" / "release" / self._get_lib_name()

    def _get_lib_name(self) -> str:
        """Get platform-specific library name"""
        if os.name == 'nt':
            return "server.dll"
        elif sys.platform == 'darwin':
            return "libserver.dylib"
        else:
            return "libserver.so"

    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        issues = []

        # Check TB Lang executable
        if not self.tbx_executable.exists():
            issues.append(f"❌ TB Lang executable not found: {self.tbx_executable}")
            issues.append("   Build it with: cd toolboxv2/tb-exc/src && cargo build --release")

        # Check main.tbx
        if not self.main_tbx.exists():
            issues.append(f"❌ Core runtime not found: {self.main_tbx}")

        # Check dist directory
        if not self.dist_dir.exists():
            issues.append(f"⚠️  Static files directory not found: {self.dist_dir}")
            issues.append("   Server will create it automatically")

        # Check server plugin (optional for JIT mode)
        if not self.server_lib.exists():
            issues.append(f"⚠️  Server plugin not compiled: {self.server_lib}")
            issues.append("   Build it with: cd toolboxv2/tb-exc/src/builtin-plugins/server && cargo build --release")
            issues.append("   Note: Server plugin is required for FFI mode")

        if issues:
            print("╔════════════════════════════════════════════════════════════╗")
            print("║       Prerequisites Check                                  ║")
            print("╚════════════════════════════════════════════════════════════╝")
            for issue in issues:
                print(issue)
            print()
            return False

        return True

    def load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self.get_default_config()

    def save_config(self, config: Dict[str, Any]):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)

    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8080,
                "workers": 4,
                "static_dir": str(self.dist_dir),
                "enable_websocket": True,
                "enable_cors": True
            },
            "security": {
                "rate_limit": 100,
                "rate_limit_window": 60,
                "session_timeout": 3600,
                "require_auth": True,
                "cors_enabled": True,
                "allowed_origins": ["*"]
            },
            "auth": {
                "jwt_validation_module": "CloudM.AuthManager",
                "jwt_validation_function": "jwt_check_claim_server_side",
                "session_validation_endpoint": "/validateSession",
                "anonymous_allowed": False
            },
            "runtime": {
                "mode": "jit",
                "optimize": True
            }
        }

    def load_state(self) -> Dict[str, Any]:
        """Load runtime state"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {"pid": None, "status": "stopped", "started_at": None}

    def save_state(self, state: Dict[str, Any]):
        """Save runtime state"""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def build_core(self, release: bool = True) -> bool:
        """
        Build (compile) the TB Lang Core Runtime to a standalone executable

        NOTE: Currently the TB Lang compiler has issues with complex type inference
        in main.tbx. This feature is experimental and may not work until the compiler
        is improved. For now, use JIT mode with 'start' command.

        Args:
            release: Build in release mode (optimized)

        Returns:
            True if build successful, False otherwise
        """
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       Building TB Lang Core Runtime (EXPERIMENTAL)        ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()
        print("⚠️  WARNING: AOT compilation is currently experimental!")
        print("   The TB Lang compiler has issues with complex type inference.")
        print("   For production use, run in JIT mode with 'start' command.")
        print()
        print(f"Mode: {'Release (Optimized)' if release else 'Debug'}")
        print(f"Source: {self.main_tbx}")
        print()

        if not self.main_tbx.exists():
            print(f"❌ Error: main.tbx not found: {self.main_tbx}")
            return False

        if not self.tbx_executable.exists():
            print(f"❌ Error: TB Lang compiler not found: {self.tbx_executable}")
            print("   Build it with: cd toolboxv2/tb-exc/src && cargo build --release")
            return False

        # Create temporary output file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe' if os.name == 'nt' else '', mode='w') as f:
            temp_output = Path(f.name)

        try:
            print("🔨 Compiling main.tbx...")
            compile_start = time.perf_counter()

            # Compile command
            # Note: tbx compile doesn't have --release flag, it always optimizes
            cmd = [
                str(self.tbx_executable),
                "compile",
                "--output",
                str(temp_output),
                str(self.main_tbx)
            ]

            print(f"   Command: {' '.join(cmd)}")
            print()

            result = subprocess.run(
                cmd,
                cwd=str(self.core_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            compile_time = (time.perf_counter() - compile_start) * 1000

            if result.returncode != 0:
                print("❌ Compilation failed!")
                print()
                print("═══════════════════════════════════════════════════════════")
                print("  KNOWN ISSUE: TB Lang Compiler Type Inference Limitations")
                print("═══════════════════════════════════════════════════════════")
                print()
                print("The TB Lang compiler currently has issues with:")
                print("  • Complex type inference in nested function calls")
                print("  • DictValue vs primitive type conversions")
                print("  • HashMap<String, DictValue> vs HashMap<String, String>")
                print()
                print("WORKAROUND: Use JIT mode instead:")
                print("  python -m toolboxv2.utils.clis.tbx_core_v3_cli start")
                print()
                print("═══════════════════════════════════════════════════════════")
                print()

                # Show compilation output for debugging
                if result.stdout:
                    print("Compilation output:")
                    print(result.stdout)
                    print()
                if result.stderr:
                    print("Compilation errors:")
                    print(result.stderr)
                    print()

                return False

            print(f"✅ Compiled successfully in {compile_time:.2f}ms")

            # Make executable on Unix
            if os.name != 'nt':
                os.chmod(temp_output, 0o755)

            # Store temp path for deployment
            self._compiled_binary = temp_output

            return True

        except Exception as e:
            print(f"❌ Build failed: {e}")
            if temp_output.exists():
                try:
                    os.unlink(temp_output)
                except:
                    pass
            return False

    def deploy_core(self) -> bool:
        """
        Deploy the compiled core runtime to bin directory

        Returns:
            True if deployment successful, False otherwise
        """
        print()
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       Deploying TB Lang Core Runtime                      ║")
        print("╚════════════════════════════════════════════════════════════╝")

        if not hasattr(self, '_compiled_binary') or not self._compiled_binary.exists():
            print("❌ Error: No compiled binary found. Run 'build' first.")
            return False

        # Ensure bin directory exists
        BIN_DIR.mkdir(parents=True, exist_ok=True)

        dest_path = BIN_DIR / CORE_EXECUTABLE_NAME

        try:
            # Remove old version if exists
            if dest_path.exists():
                print(f"🗑️  Removing old version: {dest_path}")
                os.remove(dest_path)

            # Copy new version
            print(f"📦 Deploying to: {dest_path}")
            shutil.copy(self._compiled_binary, dest_path)

            # Make executable on Unix
            if os.name != 'nt':
                os.chmod(dest_path, 0o755)

            # Clean up temp file
            try:
                os.unlink(self._compiled_binary)
            except:
                pass

            print(f"✅ Deployed successfully!")
            print()
            print(f"   Executable: {dest_path}")
            print(f"   Size: {dest_path.stat().st_size / 1024:.2f} KB")
            print()
            print("   Run with:")
            print(f"   $ {dest_path}")

            return True

        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False

    def run_compiled_core(self, args: List[str] = None) -> int:
        """
        Run the compiled core runtime executable

        Args:
            args: Additional arguments to pass to the executable

        Returns:
            Exit code from the executable
        """
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       Running TB Lang Core Runtime (Compiled)             ║")
        print("╚════════════════════════════════════════════════════════════╝")

        core_exe = BIN_DIR / CORE_EXECUTABLE_NAME

        if not core_exe.exists():
            print(f"❌ Error: Compiled core not found: {core_exe}")
            print("   Build and deploy first with:")
            print("   $ python -m toolboxv2.utils.clis.tbx_core_v3_cli build")
            print("   $ python -m toolboxv2.utils.clis.tbx_core_v3_cli deploy")
            return 1

        cmd = [str(core_exe)]
        if args:
            cmd.extend(args)

        print(f"🚀 Executing: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, cwd=str(self.workspace_root))
            return result.returncode
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            return 1

    def run_tbx_script(self, script_path: Path, args: List[str] = None, mode: str = "jit") -> int:
        """Run a .tbx script using TB Lang compiler"""
        if not script_path.exists():
            print(f"❌ Error: Script not found: {script_path}")
            return 1

        if not self.tbx_executable.exists():
            print(f"❌ Error: TB Lang executable not found: {self.tbx_executable}")
            print("   Build it with: cd toolboxv2/tb-exc/src && cargo build --release")
            return 1

        # Build command - use 'run' directly without 'x' parameter
        cmd = [str(self.tbx_executable), "run", str(script_path)]
        if mode:
            cmd.extend(["--mode", mode])
        if args:
            cmd.extend(args)

        print(f"🚀 Running: {' '.join(cmd)}")
        print(f"📂 Working directory: {self.core_dir}")
        print()

        try:
            result = subprocess.run(cmd, cwd=str(self.core_dir))
            return result.returncode
        except FileNotFoundError:
            print(f"❌ Error: TB Lang compiler not found: {self.tbx_executable}")
            return 1
        except Exception as e:
            print(f"❌ Error running script: {e}")
            return 1

    def start_server(self, background: bool = False, mode: str = "jit"):
        """Start TB Lang Core Runtime server"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       TB Lang Core Runtime v3.0.0 - Starting Server       ║")
        print("╚════════════════════════════════════════════════════════════╝")

        # Important note about server mode
        if mode == "jit":
            print("\n⚠️  IMPORTANT NOTE:")
            print("   Rust plugins (including server) are NOT supported in JIT mode!")
            print("   The core will run but server functionality will be limited (stub only).")
            print("   For full server functionality, use AOT compilation:")
            print("   $ tbx compile main.tbx --output core_server")
            print("   $ ./core_server")
            print()
            response = input("Continue in JIT mode anyway? (y/N): ")
            if response.lower() != 'y':
                print("Cancelled.")
                return 0

        # Check prerequisites
        if not self.check_prerequisites():
            if mode == "ffi" and not self.server_lib.exists():
                print("❌ Cannot start in FFI mode without server plugin")
                return 1

        # Check if already running
        state = self.load_state()
        if state["status"] == "running" and state["pid"]:
            print(f"⚠️  Server already running (PID: {state['pid']})")
            return 0

        # Load config
        config = self.load_config()
        print(f"📋 Configuration:")
        print(f"   Host: {config['server']['host']}")
        print(f"   Port: {config['server']['port']}")
        print(f"   Mode: {mode.upper()}")
        print(f"   Static Dir: {config['server']['static_dir']}")
        print(f"   Auth Required: {config['security']['require_auth']}")
        print()

        if background:
            # Start in background
            cmd = [str(self.tbx_executable), "run", str(self.main_tbx), "--mode", mode]
            process = subprocess.Popen(
                cmd,
                cwd=str(self.core_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Save state
            self.save_state({
                "pid": process.pid,
                "status": "running",
                "started_at": time.time(),
                "mode": mode
            })

            print(f"✅ Server started in background (PID: {process.pid})")
            print(f"   Logs: {self.core_dir / 'server.log'}")
            print(f"   URL: http://{config['server']['host']}:{config['server']['port']}")
            return 0
        else:
            # Run in foreground
            print("🚀 Starting server in foreground mode...")
            print("   Press Ctrl+C to stop")
            print()
            return self.run_tbx_script(self.main_tbx, mode=mode)

    def stop_server(self):
        """Stop TB Lang Core Runtime server"""
        state = self.load_state()

        if state["status"] != "running" or not state["pid"]:
            print("⚠️  Server is not running")
            return 0

        print(f"🛑 Stopping server (PID: {state['pid']})...")

        try:
            # Try graceful shutdown first
            if os.name == 'nt':
                # Windows
                subprocess.run(['taskkill', '/PID', str(state['pid']), '/F'], check=False)
            else:
                # Unix-like
                os.kill(state['pid'], signal.SIGTERM)

            # Wait for process to stop
            time.sleep(2)

            # Update state
            self.save_state({
                "pid": None,
                "status": "stopped",
                "started_at": None
            })

            print("✅ Server stopped")
            return 0
        except ProcessLookupError:
            print("⚠️  Process not found (already stopped?)")
            self.save_state({
                "pid": None,
                "status": "stopped",
                "started_at": None
            })
            return 0
        except Exception as e:
            print(f"❌ Error stopping server: {e}")
            return 1

    def status(self):
        """Show server status"""
        state = self.load_state()
        config = self.load_config()

        print("╔════════════════════════════════════════════════════════════╗")
        print("║       TB Lang Core Runtime v3.0.0 - Status                ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print(f"║  Status:        {state['status']:<40} ║")

        if state.get("pid"):
            print(f"║  PID:           {state['pid']:<40} ║")

        if state.get("mode"):
            print(f"║  Mode:          {state['mode'].upper():<40} ║")

        if state.get("started_at"):
            uptime = int(time.time() - state["started_at"])
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            seconds = uptime % 60
            uptime_str = f"{hours}h {minutes}m {seconds}s"
            print(f"║  Uptime:        {uptime_str:<40} ║")

        print(f"║  Host:          {config['server']['host']:<40} ║")
        print(f"║  Port:          {config['server']['port']:<40} ║")
        print(f"║  CORS:          {str(config['server']['enable_cors']):<40} ║")
        print(f"║  WebSocket:     {str(config['server']['enable_websocket']):<40} ║")
        print(f"║  Auth Required: {str(config['security']['require_auth']):<40} ║")
        print(f"║  Static Dir:    {config['server']['static_dir']:<40} ║")
        print("╚════════════════════════════════════════════════════════════╝")

        # Check if process is actually running
        if state.get("pid"):
            try:
                if os.name == 'nt':
                    # Windows
                    result = subprocess.run(['tasklist', '/FI', f'PID eq {state["pid"]}'],
                                          capture_output=True, text=True)
                    if str(state["pid"]) not in result.stdout:
                        print("\n⚠️  Warning: Process not found (server may have crashed)")
                else:
                    # Unix-like
                    os.kill(state["pid"], 0)
            except (ProcessLookupError, subprocess.CalledProcessError):
                print("\n⚠️  Warning: Process not found (server may have crashed)")

        return 0

    def run_tests(self, test_type: str = "all", verbose: bool = False, report_file: str = None):
        """Run tests and generate detailed error report"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       TB Lang Core Runtime v3.0.0 - Running Tests         ║")
        print("╚════════════════════════════════════════════════════════════╝")

        # Only check prerequisites for TBX tests
        if test_type in ["all", "tbx", "security", "e2e"] and not self.tbx_executable.exists():
            print("❌ TB Lang executable not found. Cannot run TBX tests.")
            print(f"   Expected: {self.tbx_executable}")
            return 1

        # Collect all test files
        all_test_files = {
            "python": [],
            "tbx": [],
            "security": [],
            "e2e": []
        }

        # Discover all test files
        if self.tests_dir.exists():
            for test_file in self.tests_dir.iterdir():
                if test_file.is_file():
                    if test_file.suffix == ".py" and test_file.name.startswith("test_"):
                        all_test_files["python"].append(test_file)
                        # Categorize E2E tests
                        if "e2e" in test_file.name or "welcome" in test_file.name:
                            all_test_files["e2e"].append(test_file)
                    elif test_file.suffix == ".tbx" and test_file.name.startswith("test_"):
                        all_test_files["tbx"].append(test_file)
                        # Categorize security tests
                        if "security" in test_file.name or "path_traversal" in test_file.name:
                            all_test_files["security"].append(test_file)

        # Filter tests based on type
        tests_to_run = {"python": [], "tbx": []}

        if test_type == "all":
            tests_to_run["python"] = all_test_files["python"]
            tests_to_run["tbx"] = all_test_files["tbx"]
        elif test_type == "python":
            tests_to_run["python"] = all_test_files["python"]
        elif test_type == "tbx":
            tests_to_run["tbx"] = all_test_files["tbx"]
        elif test_type == "security":
            tests_to_run["tbx"] = all_test_files["security"]
        elif test_type == "e2e":
            tests_to_run["python"] = all_test_files["e2e"]
        elif test_type == "integration":
            tests_to_run["python"] = all_test_files["python"]

        # Test results tracking
        test_results = []
        total_passed = 0
        total_failed = 0
        total_skipped = 0

        print(f"\n📋 Found {len(tests_to_run['python'])} Python tests and {len(tests_to_run['tbx'])} TBX tests")
        print()

        # Run Python tests
        for test_file in sorted(tests_to_run["python"]):
            print(f"\n{'='*60}")
            print(f"🧪 Running Python test: {test_file.name}")
            print(f"{'='*60}")

            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=str(self.tests_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            duration = time.time() - start_time

            test_result = {
                "name": test_file.name,
                "type": "python",
                "path": str(test_file),
                "returncode": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }
            test_results.append(test_result)

            if verbose or result.returncode != 0:
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

            if result.returncode == 0:
                total_passed += 1
                print(f"✅ {test_file.name} PASSED ({duration:.2f}s)")
            else:
                total_failed += 1
                print(f"❌ {test_file.name} FAILED ({duration:.2f}s)")

        # Run TB Lang tests
        for test_file in sorted(tests_to_run["tbx"]):
            print(f"\n{'='*60}")
            print(f"🧪 Running TBX test: {test_file.name}")
            print(f"{'='*60}")

            start_time = time.time()

            # Build command
            cmd = [str(self.tbx_executable), "run", str(test_file), "--mode", "jit"]

            result = subprocess.run(
                cmd,
                cwd=str(self.core_dir),
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            duration = time.time() - start_time

            test_result = {
                "name": test_file.name,
                "type": "tbx",
                "path": str(test_file),
                "returncode": result.returncode,
                "duration": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "status": "PASSED" if result.returncode == 0 else "FAILED"
            }
            test_results.append(test_result)

            if verbose or result.returncode != 0:
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

            if result.returncode == 0:
                total_passed += 1
                print(f"✅ {test_file.name} PASSED ({duration:.2f}s)")
            else:
                total_failed += 1
                print(f"❌ {test_file.name} FAILED ({duration:.2f}s)")

        # Generate detailed report
        self._generate_test_report(test_results, total_passed, total_failed, total_skipped, report_file)

        # Summary
        print("\n╔════════════════════════════════════════════════════════════╗")
        print("║       Test Summary                                         ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print(f"║  Total Tests:   {total_passed + total_failed:<40} ║")
        print(f"║  Passed:        {total_passed:<40} ║")
        print(f"║  Failed:        {total_failed:<40} ║")
        print(f"║  Skipped:       {total_skipped:<40} ║")
        success_rate = (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0
        print(f"║  Success Rate:  {success_rate:.1f}%{'':<36} ║")
        print("╚════════════════════════════════════════════════════════════╝")

        if report_file:
            print(f"\n📄 Detailed report saved to: {report_file}")

        return 0 if total_failed == 0 else 1

    def _generate_test_report(self, test_results: List[Dict], passed: int, failed: int, skipped: int, report_file: str = None):
        """Generate detailed test report"""
        if not report_file:
            report_file = str(self.tests_dir / f"TEST_REPORT_{time.strftime('%Y%m%d_%H%M%S')}.md")
        else:
            # Ensure absolute path
            report_file = str(Path(report_file).resolve())

        # Ensure directory exists
        Path(report_file).parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TB Lang Core Runtime v3.0.0 - Test Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            f.write("## Summary\n\n")
            f.write(f"- **Total Tests:** {passed + failed + skipped}\n")
            f.write(f"- **Passed:** {passed} ✅\n")
            f.write(f"- **Failed:** {failed} ❌\n")
            f.write(f"- **Skipped:** {skipped} ⚠️\n")
            success_rate = (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0
            f.write(f"- **Success Rate:** {success_rate:.1f}%\n\n")

            # Failed tests details
            failed_tests = [t for t in test_results if t["status"] == "FAILED"]
            if failed_tests:
                f.write("## ❌ Failed Tests\n\n")
                for test in failed_tests:
                    f.write(f"### {test['name']}\n\n")
                    f.write(f"- **Type:** {test['type']}\n")
                    f.write(f"- **Path:** `{test['path']}`\n")
                    f.write(f"- **Duration:** {test['duration']:.2f}s\n")
                    f.write(f"- **Return Code:** {test['returncode']}\n\n")

                    if test['stdout']:
                        f.write("**Output:**\n```\n")
                        f.write(test['stdout'][:5000])  # Limit output
                        if len(test['stdout']) > 5000:
                            f.write("\n... (truncated)")
                        f.write("\n```\n\n")

                    if test['stderr']:
                        f.write("**Errors:**\n```\n")
                        f.write(test['stderr'][:5000])
                        if len(test['stderr']) > 5000:
                            f.write("\n... (truncated)")
                        f.write("\n```\n\n")

                    f.write("---\n\n")

            # Passed tests
            passed_tests = [t for t in test_results if t["status"] == "PASSED"]
            if passed_tests:
                f.write("## ✅ Passed Tests\n\n")
                f.write("| Test Name | Type | Duration |\n")
                f.write("|-----------|------|----------|\n")
                for test in passed_tests:
                    f.write(f"| {test['name']} | {test['type']} | {test['duration']:.2f}s |\n")
                f.write("\n")

            # All test details
            f.write("## 📋 All Test Details\n\n")
            for test in test_results:
                status_icon = "✅" if test["status"] == "PASSED" else "❌"
                f.write(f"### {status_icon} {test['name']}\n\n")
                f.write(f"- **Type:** {test['type']}\n")
                f.write(f"- **Status:** {test['status']}\n")
                f.write(f"- **Duration:** {test['duration']:.2f}s\n")
                f.write(f"- **Return Code:** {test['returncode']}\n\n")

                if test['status'] == "PASSED" and test['stdout']:
                    # Show brief output for passed tests
                    lines = test['stdout'].split('\n')
                    if len(lines) > 20:
                        f.write("<details>\n<summary>Show output</summary>\n\n```\n")
                        f.write(test['stdout'][:2000])
                        f.write("\n```\n</details>\n\n")
                    else:
                        f.write("**Output:**\n```\n")
                        f.write(test['stdout'])
                        f.write("\n```\n\n")

                f.write("---\n\n")

            # System info
            f.write("## 🖥️ System Information\n\n")
            f.write(f"- **Workspace:** `{self.workspace_root}`\n")
            f.write(f"- **Core Dir:** `{self.core_dir}`\n")
            f.write(f"- **TB Executable:** `{self.tbx_executable}`\n")
            f.write(f"- **TB Exec Exists:** {self.tbx_executable.exists()}\n")
            f.write(f"- **Python Version:** {sys.version}\n")
            f.write(f"- **Platform:** {sys.platform}\n\n")

    def build_server_plugin(self, release: bool = True):
        """Build the Rust server plugin"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       Building Server Plugin                               ║")
        print("╚════════════════════════════════════════════════════════════╝")

        if not self.server_plugin_dir.exists():
            print(f"❌ Server plugin directory not found: {self.server_plugin_dir}")
            return 1

        print(f"📂 Plugin directory: {self.server_plugin_dir}")
        print(f"🔨 Build mode: {'Release' if release else 'Debug'}")
        print()

        cmd = ["cargo", "build"]
        if release:
            cmd.append("--release")

        print(f"🚀 Running: {' '.join(cmd)}")
        print()

        try:
            result = subprocess.run(cmd, cwd=str(self.server_plugin_dir))
            if result.returncode == 0:
                print("\n✅ Server plugin built successfully!")
                print(f"📦 Library: {self.server_lib}")
                return 0
            else:
                print("\n❌ Build failed!")
                return 1
        except FileNotFoundError:
            print("❌ Error: Cargo not found. Please install Rust first.")
            return 1
        except Exception as e:
            print(f"❌ Error building plugin: {e}")
            return 1

    def info(self):
        """Show system information"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       TB Lang Core Runtime v3.0.0 - System Info            ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print(f"║  Workspace:     {str(self.workspace_root)[:40]:<40} ║")
        print(f"║  Core Dir:      {str(self.core_dir)[:40]:<40} ║")
        print(f"║  TB Executable: {str(self.tbx_executable)[:40]:<40} ║")
        print(f"║  Server Plugin: {str(self.server_lib)[:40]:<40} ║")
        print(f"║  Static Dir:    {str(self.dist_dir)[:40]:<40} ║")
        print("╠════════════════════════════════════════════════════════════╣")
        print(f"║  TB Exec Exists: {str(self.tbx_executable.exists()):<39}  ║")
        print(f"║  Main.tbx Exists: {str(self.main_tbx.exists()):<38} ║")
        print(f"║  Plugin Exists:  {str(self.server_lib.exists()):<39} ║")
        print(f"║  Dist Exists:    {str(self.dist_dir.exists()):<39} ║")
        print("╚════════════════════════════════════════════════════════════╝")
        return 0

    def validate(self):
        """Validate the installation"""
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       TB Lang Core Runtime v3.0.0 - Validation            ║")
        print("╚════════════════════════════════════════════════════════════╝")

        all_ok = True

        # Check TB executable
        print("\n1. Checking TB Lang executable...")
        if self.tbx_executable.exists():
            print(f"   ✅ Found: {self.tbx_executable}")
        else:
            print(f"   ❌ Not found: {self.tbx_executable}")
            print("      Build with: cd toolboxv2/tb-exc/src && cargo build --release")
            all_ok = False

        # Check main.tbx
        print("\n2. Checking core runtime...")
        if self.main_tbx.exists():
            print(f"   ✅ Found: {self.main_tbx}")
            # Check version
            try:
                with open(self.main_tbx, 'r', encoding='utf-8') as f:
                    content = f.read(500)
                    if "v3.0.0" in content:
                        print("   ✅ Version: v3.0.0")
                    else:
                        print("   ⚠️  Version check failed (expected v3.0.0)")
            except Exception as e:
                print(f"   ⚠️  Could not read file: {e}")
        else:
            print(f"   ❌ Not found: {self.main_tbx}")
            all_ok = False

        # Check server plugin
        print("\n3. Checking server plugin...")
        if self.server_lib.exists():
            print(f"   ✅ Found: {self.server_lib}")
        else:
            print(f"   ⚠️  Not found: {self.server_lib}")
            print("      Build with: cd toolboxv2/tb-exc/src/builtin-plugins/server && cargo build --release")
            print("      Note: Required for FFI mode, optional for JIT mode")

        # Check dist directory
        print("\n4. Checking static files directory...")
        if self.dist_dir.exists():
            print(f"   ✅ Found: {self.dist_dir}")
            # Count files
            file_count = len(list(self.dist_dir.rglob('*')))
            print(f"   📁 Files: {file_count}")
        else:
            print(f"   ⚠️  Not found: {self.dist_dir}")
            print("      Will be created automatically when needed")

        # Check Python dependencies
        print("\n5. Checking Python dependencies...")
        try:
            sys.path.insert(0, str(self.workspace_root))
            from toolboxv2.utils.toolbox import App
            print("   ✅ ToolBoxV2 framework available (App class)")
            # Try to import other key components
            from toolboxv2.utils.system.types import Result, ApiResult
            print("   ✅ ToolBoxV2 types available (Result, ApiResult)")
        except ImportError as e:
            print(f"   ❌ ToolBoxV2 framework not found: {e}")
            all_ok = False

        # Summary
        print("\n╔════════════════════════════════════════════════════════════╗")
        if all_ok:
            print("║  ✅ Validation PASSED - System ready                      ║")
        else:
            print("║  ❌ Validation FAILED - Please fix issues above           ║")
        print("╚════════════════════════════════════════════════════════════╝")

        return 0 if all_ok else 1


def cli_tbx_core():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="🚀 TB Lang Core Runtime v3.0.0 - Multi-Language Plugin System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='tb core',
        epilog="""
╔══════════════════════════════════════════╗
║          Command Examples                ║
╠══════════════════════════════════════════╣
║                                          ║
║  Build & Deploy:                         ║
║    $ tb core build                       ║
║    $ tb core build --debug               ║
║    $ tb core deploy                      ║
║    $ tb core build-deploy                ║
║    $ tb core run-compiled                ║
║                                          ║
║  Server Management:                      ║
║    $ tb core start                       ║
║    $ tb core start --background          ║
║    $ tb core start --mode ffi            ║
║    $ tb core stop                        ║
║    $ tb core status                      ║
║                                          ║
║  Testing:                                ║
║    $ tb core test                        ║
║    $ tb core test --type python          ║
║    $ tb core test --type tbx             ║
║    $ tb core test --type security        ║
║    $ tb core test --type e2e             ║
║    $ tb core test --report report.md     ║
║                                          ║
║  Validation:                             ║
║    $ tb core validate                    ║
║    $ tb core info                        ║
║    $ tb core build-plugin                ║
║                                          ║
╚══════════════════════════════════════════╝
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command (compile main.tbx)
    build_parser = subparsers.add_parser('build', help='Build (compile) TB Lang Core Runtime to executable')
    build_parser.add_argument('--debug', '-d', action='store_true',
                             help='Build in debug mode (default: release/optimized)')

    # Deploy command (move to bin/)
    subparsers.add_parser('deploy', help='Deploy compiled core runtime to bin directory')

    # Build-Deploy command (build + deploy in one step)
    bd_parser = subparsers.add_parser('build-deploy', help='Build and deploy in one step')
    bd_parser.add_argument('--debug', '-d', action='store_true',
                          help='Build in debug mode (default: release/optimized)')

    # Run compiled command
    run_compiled_parser = subparsers.add_parser('run-compiled', help='Run the compiled core runtime')
    run_compiled_parser.add_argument('args', nargs='*', help='Additional arguments to pass')

    # Start command
    start_parser = subparsers.add_parser('start', help='Start TB Lang Core Runtime server (JIT mode)')
    start_parser.add_argument('--background', '-b', action='store_true',
                             help='Run server in background')
    start_parser.add_argument('--mode', '-m', choices=['jit', 'ffi'], default='jit',
                             help='Execution mode (jit=Python JIT, ffi=Rust FFI)')

    # Stop command
    subparsers.add_parser('stop', help='Stop TB Lang Core Runtime server')

    # Status command
    subparsers.add_parser('status', help='Show server status')

    # Test command
    test_parser = subparsers.add_parser('test', help='Run tests')
    test_parser.add_argument('--type', '-t',
                            choices=['all', 'python', 'tbx', 'integration', 'security', 'e2e'],
                            default='all',
                            help='Type of tests to run (all=all tests, python=Python tests, '
                                 'tbx=TB Lang tests, integration=integration tests, '
                                 'security=path traversal & security tests, '
                                 'e2e=end-to-end Welcome module tests)')
    test_parser.add_argument('--verbose', '-v', action='store_true',
                            help='Verbose output')
    test_parser.add_argument('--report', '-r', type=str, metavar='FILE',
                            help='Save detailed report to file (default: auto-generated)')

    # Build plugin command (build server plugin)
    plugin_parser = subparsers.add_parser('build-plugin', help='Build server plugin (Rust)')
    plugin_parser.add_argument('--debug', '-d', action='store_true',
                             help='Build in debug mode')

    # Validate command
    subparsers.add_parser('validate', help='Validate installation')

    # Info command
    subparsers.add_parser('info', help='Show system information')

    args = parser.parse_args()

    # Create manager
    manager = TBXCoreManager()

    # Execute command
    if args.command == 'build':
        # Build (compile) the core runtime
        success = manager.build_core(release=not args.debug)
        return 0 if success else 1

    elif args.command == 'deploy':
        # Deploy compiled core to bin/
        success = manager.deploy_core()
        return 0 if success else 1

    elif args.command == 'build-deploy':
        # Build and deploy in one step
        print("╔════════════════════════════════════════════════════════════╗")
        print("║       Build & Deploy TB Lang Core Runtime                 ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print()

        # Step 1: Build
        if not manager.build_core(release=not args.debug):
            print("\n❌ Build failed! Deployment cancelled.")
            return 1

        # Step 2: Deploy
        if not manager.deploy_core():
            print("\n❌ Deployment failed!")
            return 1

        print("\n✅ Build & Deploy completed successfully!")
        return 0

    elif args.command == 'run-compiled':
        # Run the compiled core runtime
        return manager.run_compiled_core(args=args.args)

    elif args.command == 'start':
        return manager.start_server(background=args.background, mode=args.mode)
    elif args.command == 'stop':
        return manager.stop_server()
    elif args.command == 'status':
        return manager.status()
    elif args.command == 'test':
        return manager.run_tests(test_type=args.type, verbose=args.verbose, report_file=args.report)
    elif args.command == 'build-plugin':
        return manager.build_server_plugin(release=not args.debug)
    elif args.command == 'validate':
        return manager.validate()
    elif args.command == 'info':
        return manager.info()
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(cli_tbx_core())


