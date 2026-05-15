import asyncio
import time
import json
from typing import Dict, List
import statistics

async def test_mcp_performance():
    """Test MCP server tool performance"""

    # Test data for different tools
    test_cases = [
        {
            "tool": "toolbox_info",
            "args": {"info_type": "modules"},
            "description": "Get modules info"
        },
        {
            "tool": "toolbox_status",
            "args": {"include_performance": True},
            "description": "System status check"
        },
        {
            "tool": "docs_reader",
            "args": {"query": "toolbox", "max_results": 5},
            "description": "Documentation search"
        },
        {
            "tool": "toolbox_execute",
            "args": {
                "module_name": "core_utils",
                "function_name": "get_system_info"
            },
            "description": "Execute simple function"
        },
        {
            "tool": "python_execute",
            "args": {"code": "print('Hello, World!')"},
            "description": "Python code execution"
        },{
            "tool": "get_update_suggestions",
            "args": {},
            "description": "Get update suggestions"
        },{
            "tool": "source_code_lookup",
            "args": {"element_name": "get_update_suggestions", "element_type": "function"},
            "description": "Source code lookup"
        }
    ]

    results = {}

    print("🚀 Starting MCP Performance Tests\n")

    for test_case in test_cases:
        tool_name = test_case["tool"]
        description = test_case["description"]

        print(f"Testing: {tool_name} - {description}")

        # Run multiple iterations
        times = []
        for i in range(5):
            start_time = time.perf_counter()

            try:
                # Simulate MCP tool call (you'll need to adapt this to your actual MCP client)
                # For now, just measure a simple operation
                await asyncio.sleep(0.001)  # Minimal async operation

                end_time = time.perf_counter()
                execution_time = (end_time - start_time) * 1000  # Convert to ms
                times.append(execution_time)

                print(f"  Run {i+1}: {execution_time:.2f}ms")

            except Exception as e:
                print(f"  Run {i+1}: ERROR - {e}")
                times.append(float('inf'))

        # Calculate statistics
        valid_times = [t for t in times if t != float('inf')]
        if valid_times:
            results[tool_name] = {
                "description": description,
                "avg_time": statistics.mean(valid_times),
                "min_time": min(valid_times),
                "max_time": max(valid_times),
                "median_time": statistics.median(valid_times),
                "success_rate": len(valid_times) / len(times) * 100
            }

        print(f"  Average: {results[tool_name]['avg_time']:.2f}ms\n")

    # Print summary
    print("📊 Performance Summary:")
    print("=" * 60)

    for tool_name, stats in results.items():
        print(f"{tool_name}:")
        print(f"  Description: {stats['description']}")
        print(f"  Average: {stats['avg_time']:.2f}ms")
        print(f"  Range: {stats['min_time']:.2f}ms - {stats['max_time']:.2f}ms")
        print(f"  Success Rate: {stats['success_rate']:.1f}%")
        print()

    # Identify performance issues
    slow_tools = {k: v for k, v in results.items() if v['avg_time'] > 1000}  # > 1 second
    if slow_tools:
        print("⚠️  Slow Tools (>1s):")
        for tool, stats in slow_tools.items():
            print(f"  - {tool}: {stats['avg_time']:.2f}ms")

    fast_tools = {k: v for k, v in results.items() if v['avg_time'] < 100}  # < 100ms
    if fast_tools:
        print("✅ Fast Tools (<100ms):")
        for tool, stats in fast_tools.items():
            print(f"  - {tool}: {stats['avg_time']:.2f}ms")

    return results

# Run the test
if __name__ == "__main__":
    results = asyncio.run(test_mcp_performance())
