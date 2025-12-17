"""
Comprehensive Test Suite for Flow Management System
===================================================

Tests all aspects of the production-ready flow management:
- Flow discovery and registration
- Session management and lifecycle
- I/O interception and control
- MCP tool handlers
- Edge cases and error handling
"""

import asyncio
import time
from typing import Dict, Any


# =========================================================================
# TEST FLOWS
# =========================================================================

class TestFlows:
    """Collection of test flows covering different scenarios"""
    
    @staticmethod
    async def simple_flow(app, args_sto, message: str = "Hello", **kwargs):
        """Simple flow that just prints and returns"""
        print(f"Simple flow: {message}")
        return {"status": "success", "message": message}
    
    @staticmethod
    async def interactive_flow(app, args_sto, **kwargs):
        """Interactive flow with multiple input prompts"""
        print("=== Interactive Flow Started ===")
        
        name = input("What's your name? ")
        print(f"Hello, {name}!")
        
        age = input("How old are you? ")
        print(f"You are {age} years old.")
        
        hobby = input("What's your hobby? ")
        print(f"Interesting! {hobby} sounds fun.")
        
        return {
            "name": name,
            "age": age,
            "hobby": hobby
        }
    
    @staticmethod
    async def error_flow(app, args_sto, should_fail: bool = True, **kwargs):
        """Flow that can trigger errors"""
        print("Error flow started")
        
        if should_fail:
            raise ValueError("Intentional test error")
        
        return {"status": "success"}
    
    @staticmethod
    async def long_running_flow(app, args_sto, duration: int = 2, **kwargs):
        """Flow that takes some time to complete"""
        print(f"Starting long task ({duration}s)")
        
        for i in range(duration):
            await asyncio.sleep(1)
            print(f"Progress: {i+1}/{duration}")
        
        print("Task completed!")
        return {"duration": duration, "status": "completed"}
    
    @staticmethod
    async def conditional_input_flow(app, args_sto, ask_questions: bool = True, **kwargs):
        """Flow with conditional input"""
        print("Conditional flow started")
        
        if ask_questions:
            answer = input("Do you want to continue? (yes/no) ")
            print(f"You answered: {answer}")
            
            if answer.lower() == "yes":
                detail = input("Provide more details: ")
                print(f"Details: {detail}")
                return {"continued": True, "details": detail}
            else:
                return {"continued": False}
        else:
            print("Skipping questions")
            return {"continued": False}


# =========================================================================
# TEST SUITE
# =========================================================================

class FlowSystemTests:
    """Comprehensive test suite"""
    
    def __init__(self):
        self.app = None
        self.gateway = None
        self.session_manager = None
        self.flow_handlers = None
        self.results = []
    
    async def setup(self):
        """Setup test environment"""
        from flow_management import FlowGateway, FlowSessionManager
        from flow_integration import FlowToolHandlers
        
        # Create mock app
        class MockApp:
            def __init__(self):
                self.flows = {
                    "simple": TestFlows.simple_flow,
                    "interactive": TestFlows.interactive_flow,
                    "error": TestFlows.error_flow,
                    "long_running": TestFlows.long_running_flow,
                    "conditional": TestFlows.conditional_input_flow,
                }
        
        self.app = MockApp()
        self.session_manager = FlowSessionManager(max_sessions=50, timeout=300)
        self.gateway = FlowGateway(self.app, self.session_manager)
        self.flow_handlers = FlowToolHandlers(self.gateway)
        
        await self.session_manager.start_cleanup()
        
        print("✅ Test environment setup complete\n")
    
    async def teardown(self):
        """Cleanup test environment"""
        await self.session_manager.stop_cleanup()
        print("\n✅ Test environment cleaned up")
    
    def log_result(self, test_name: str, passed: bool, message: str = ""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append((test_name, passed, message))
        print(f"{status}: {test_name}")
        if message:
            print(f"  → {message}")
    
    # =====================================================================
    # DISCOVERY TESTS
    # =====================================================================
    
    async def test_flow_discovery(self):
        """Test that flows are discovered correctly"""
        flows = self.gateway.get_available_flows()
        
        expected_flows = ["simple", "interactive", "error", "long_running", "conditional"]
        discovered = list(flows.keys())
        
        passed = all(f in discovered for f in expected_flows)
        self.log_result(
            "Flow Discovery",
            passed,
            f"Discovered {len(discovered)} flows: {discovered}"
        )
    
    async def test_flow_metadata(self):
        """Test flow metadata extraction"""
        flows = self.gateway.get_available_flows()
        simple_meta = flows.get("simple")
        
        passed = (
            simple_meta is not None and
            simple_meta.has_async and
            "message" in simple_meta.parameters
        )
        
        self.log_result(
            "Flow Metadata",
            passed,
            f"Metadata: async={simple_meta.has_async}, params={list(simple_meta.parameters.keys())}"
        )
    
    # =====================================================================
    # SESSION MANAGEMENT TESTS
    # =====================================================================
    
    async def test_session_creation(self):
        """Test session creation and retrieval"""
        session = await self.session_manager.create("simple", message="test")
        retrieved = await self.session_manager.get(session.session_id)
        
        passed = (
            retrieved is not None and
            retrieved.session_id == session.session_id and
            retrieved.flow_name == "simple"
        )
        
        self.log_result(
            "Session Creation",
            passed,
            f"Session ID: {session.session_id}"
        )
        
        await self.session_manager.delete(session.session_id)
    
    async def test_session_expiration(self):
        """Test session expiration mechanism"""
        # Create session with very short timeout
        old_timeout = self.session_manager._timeout
        self.session_manager._timeout = 1  # 1 second
        
        session = await self.session_manager.create("simple")
        session_id = session.session_id
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Manually check expiration
        expired = session.is_expired(1)
        
        # Restore timeout
        self.session_manager._timeout = old_timeout
        
        self.log_result(
            "Session Expiration",
            expired,
            "Session correctly identified as expired"
        )
        
        await self.session_manager.delete(session_id)
    
    async def test_max_sessions_limit(self):
        """Test max sessions enforcement"""
        old_max = self.session_manager._max_sessions
        self.session_manager._max_sessions = 3
        
        # Create 4 sessions
        sessions = []
        for i in range(4):
            s = await self.session_manager.create("simple", message=f"test{i}")
            sessions.append(s.session_id)
            await asyncio.sleep(0.1)  # Ensure different timestamps
        
        # First session should be removed
        first_exists = await self.session_manager.get(sessions[0]) is not None
        last_exists = await self.session_manager.get(sessions[-1]) is not None
        
        passed = not first_exists and last_exists
        
        # Cleanup
        for sid in sessions[1:]:
            await self.session_manager.delete(sid)
        self.session_manager._max_sessions = old_max
        
        self.log_result(
            "Max Sessions Limit",
            passed,
            "Oldest session removed when limit exceeded"
        )
    
    # =====================================================================
    # FLOW EXECUTION TESTS
    # =====================================================================
    
    async def test_simple_flow_execution(self):
        """Test simple flow execution"""
        session = await self.gateway.start_flow("simple", message="Test Message")
        
        # Wait for execution
        await asyncio.sleep(0.5)
        
        status = await self.gateway.get_session_status(session.session_id)
        
        passed = (
            status['state'] == 'completed' and
            status['has_result'] and
            len(status['output_buffer']) > 0
        )
        
        self.log_result(
            "Simple Flow Execution",
            passed,
            f"State: {status['state']}, Output: {status['output_buffer']}"
        )
        
        await self.session_manager.delete(session.session_id)
    
    async def test_interactive_flow_with_input(self):
        """Test interactive flow with programmatic input"""
        session = await self.gateway.start_flow("interactive")
        
        # Wait for first input request
        await asyncio.sleep(0.5)
        status1 = await self.gateway.get_session_status(session.session_id)
        
        # Provide first input
        await self.gateway.continue_flow(session.session_id, "Alice")
        await asyncio.sleep(0.5)
        
        # Provide second input
        await self.gateway.continue_flow(session.session_id, "25")
        await asyncio.sleep(0.5)
        
        # Provide third input
        await self.gateway.continue_flow(session.session_id, "Programming")
        await asyncio.sleep(0.5)
        
        final_status = await self.gateway.get_session_status(session.session_id)
        
        passed = (
            status1['state'] == 'waiting_input' and
            final_status['state'] == 'completed' and
            final_status['has_result']
        )
        
        self.log_result(
            "Interactive Flow",
            passed,
            f"Completed with result: {final_status['has_result']}"
        )
        
        await self.session_manager.delete(session.session_id)
    
    async def test_error_handling(self):
        """Test error handling in flows"""
        session = await self.gateway.start_flow("error", should_fail=True)
        
        # Wait for execution and error
        await asyncio.sleep(0.5)
        
        status = await self.gateway.get_session_status(session.session_id)
        
        passed = (
            status['state'] == 'error' and
            status['has_error']
        )
        
        self.log_result(
            "Error Handling",
            passed,
            f"Error state correctly captured"
        )
        
        await self.session_manager.delete(session.session_id)
    
    async def test_long_running_flow(self):
        """Test long-running flow"""
        start_time = time.time()
        session = await self.gateway.start_flow("long_running", duration=2)
        
        # Check it's running
        await asyncio.sleep(0.5)
        status_running = await self.gateway.get_session_status(session.session_id)
        
        # Wait for completion
        await asyncio.sleep(2)
        status_done = await self.gateway.get_session_status(session.session_id)
        
        elapsed = time.time() - start_time
        
        passed = (
            status_running['state'] == 'running' and
            status_done['state'] == 'completed' and
            elapsed >= 2
        )
        
        self.log_result(
            "Long Running Flow",
            passed,
            f"Executed in {elapsed:.1f}s, outputs: {len(status_done['output_buffer'])}"
        )
        
        await self.session_manager.delete(session.session_id)
    
    # =====================================================================
    # MCP TOOL HANDLER TESTS
    # =====================================================================
    
    async def test_flow_list_handler(self):
        """Test flow_list MCP handler"""
        result = await self.flow_handlers.handle_flow_list()
        
        passed = (
            result.success and
            result.data is not None and
            'flows' in result.data
        )
        
        self.log_result(
            "MCP flow_list Handler",
            passed,
            f"Listed {len(result.data['flows'])} flows"
        )
    
    async def test_flow_start_handler(self):
        """Test flow_start MCP handler"""
        result = await self.flow_handlers.handle_flow_start({
            "flow_name": "simple",
            "kwargs": {"message": "MCP Test"}
        })
        
        await asyncio.sleep(0.5)
        
        passed = (
            result.success and
            result.data is not None and
            'session_id' in result.data
        )
        
        if passed:
            await self.session_manager.delete(result.data['session_id'])
        
        self.log_result(
            "MCP flow_start Handler",
            passed,
            f"Session created: {result.data.get('session_id', 'N/A') if result.data else 'N/A'}"
        )
    
    async def test_flow_status_handler(self):
        """Test flow_status MCP handler"""
        # Start a flow first
        start_result = await self.flow_handlers.handle_flow_start({
            "flow_name": "simple"
        })
        
        await asyncio.sleep(0.5)
        
        if start_result.success:
            session_id = start_result.data['session_id']
            status_result = await self.flow_handlers.handle_flow_status({
                "session_id": session_id
            })
            
            passed = (
                status_result.success and
                status_result.data is not None and
                status_result.data['state'] == 'completed'
            )
            
            await self.session_manager.delete(session_id)
        else:
            passed = False
        
        self.log_result(
            "MCP flow_status Handler",
            passed,
            "Status retrieved successfully"
        )
    
    async def test_flow_continue_handler(self):
        """Test flow_continue MCP handler"""
        # Start interactive flow
        start_result = await self.flow_handlers.handle_flow_start({
            "flow_name": "interactive"
        })
        
        await asyncio.sleep(0.5)
        
        if start_result.success:
            session_id = start_result.data['session_id']
            
            # Continue with input
            continue_result = await self.flow_handlers.handle_flow_continue({
                "session_id": session_id,
                "input_data": {"user_input": "TestUser"}
            })
            
            passed = (
                continue_result.success and
                continue_result.data['state'] in ['running', 'waiting_input', 'completed']
            )
            
            await self.session_manager.delete(session_id)
        else:
            passed = False
        
        self.log_result(
            "MCP flow_continue Handler",
            passed,
            "Flow continued successfully"
        )
    
    async def test_flow_cancel_handler(self):
        """Test flow_cancel handler"""
        # Start a flow
        start_result = await self.flow_handlers.handle_flow_start({
            "flow_name": "long_running",
            "kwargs": {"duration": 10}
        })
        
        if start_result.success:
            session_id = start_result.data['session_id']
            
            # Cancel it
            cancel_result = await self.flow_handlers.handle_flow_cancel({
                "session_id": session_id
            })
            
            passed = cancel_result.success
            
            await self.session_manager.delete(session_id)
        else:
            passed = False
        
        self.log_result(
            "MCP flow_cancel Handler",
            passed,
            "Flow cancelled successfully"
        )
    
    # =====================================================================
    # EDGE CASE TESTS
    # =====================================================================
    
    async def test_invalid_flow_name(self):
        """Test handling of invalid flow name"""
        result = await self.flow_handlers.handle_flow_start({
            "flow_name": "nonexistent_flow"
        })
        
        passed = not result.success and result.error == "FlowNotFound"
        
        self.log_result(
            "Invalid Flow Name",
            passed,
            "Correctly rejected invalid flow"
        )
    
    async def test_invalid_session_id(self):
        """Test handling of invalid session ID"""
        result = await self.flow_handlers.handle_flow_status({
            "session_id": "invalid_session_id"
        })
        
        passed = not result.success
        
        self.log_result(
            "Invalid Session ID",
            passed,
            "Correctly rejected invalid session"
        )
    
    async def test_missing_parameters(self):
        """Test handling of missing required parameters"""
        result1 = await self.flow_handlers.handle_flow_start({})
        result2 = await self.flow_handlers.handle_flow_status({})
        
        passed = (
            not result1.success and result1.error == "MissingParameter" and
            not result2.success and result2.error == "MissingParameter"
        )
        
        self.log_result(
            "Missing Parameters",
            passed,
            "Correctly validated required parameters"
        )
    
    # =====================================================================
    # RUN ALL TESTS
    # =====================================================================
    
    async def run_all(self):
        """Run all tests"""
        print("\n" + "=" * 70)
        print("  FLOW MANAGEMENT SYSTEM - COMPREHENSIVE TEST SUITE")
        print("=" * 70 + "\n")
        
        await self.setup()
        
        print("\n--- Discovery Tests ---")
        await self.test_flow_discovery()
        await self.test_flow_metadata()
        
        print("\n--- Session Management Tests ---")
        await self.test_session_creation()
        await self.test_session_expiration()
        await self.test_max_sessions_limit()
        
        print("\n--- Flow Execution Tests ---")
        await self.test_simple_flow_execution()
        await self.test_interactive_flow_with_input()
        await self.test_error_handling()
        await self.test_long_running_flow()
        
        print("\n--- MCP Tool Handler Tests ---")
        await self.test_flow_list_handler()
        await self.test_flow_start_handler()
        await self.test_flow_status_handler()
        await self.test_flow_continue_handler()
        await self.test_flow_cancel_handler()
        
        print("\n--- Edge Case Tests ---")
        await self.test_invalid_flow_name()
        await self.test_invalid_session_id()
        await self.test_missing_parameters()
        
        await self.teardown()
        
        # Print summary
        print("\n" + "=" * 70)
        print("  TEST SUMMARY")
        print("=" * 70)
        
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        percentage = (passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success Rate: {percentage:.1f}%")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! 🎉")
        else:
            print("\n⚠️  Some tests failed. Review output above.")
            print("\nFailed tests:")
            for name, p, msg in self.results:
                if not p:
                    print(f"  ❌ {name}: {msg}")
        
        print("\n" + "=" * 70 + "\n")


# =========================================================================
# MAIN
# =========================================================================

if __name__ == "__main__":
    async def main():
        tests = FlowSystemTests()
        await tests.run_all()
    
    asyncio.run(main())
