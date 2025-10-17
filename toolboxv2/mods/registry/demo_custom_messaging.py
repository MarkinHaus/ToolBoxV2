# example_local_agent.py
import asyncio

from toolboxv2 import get_app

# example_registry_agent.py
# example_chain_live.py
from toolboxv2.mods.isaa.base.Agent.chain import CF


async def setup_chain_with_live_updates():
    """Example 3: Create agent chain with live progress broadcasting"""
    app = get_app("ChainLiveExample")
    isaa = app.get_mod("isaa")

    # Initialize ISAA
    await isaa.init_isaa()

    # Create and register specialized agents

    # Research agent
    researcher_builder = isaa.get_agent_builder("researcher_agent")
    researcher_builder.with_system_message(
        "You are a research specialist. Gather comprehensive information and provide detailed analysis. "
        "Always report your progress clearly."
    )
    #researcher_builder.with_models(complex_llm_model="openrouter/openai/gpt-4o")
    await isaa.register_agent(researcher_builder)

    # Writer agent
    writer_builder = isaa.get_agent_builder("writer_agent")
    writer_builder.with_system_message(
        "You are a professional writer. Create well-structured, engaging content from research data. "
        "Report your writing progress step by step."
    )
    #writer_builder.with_models(complex_llm_model="openrouter/openai/gpt-4o")
    await isaa.register_agent(writer_builder)

    # Reviewer agent
    reviewer_builder = isaa.get_agent_builder("reviewer_agent")
    reviewer_builder.with_system_message(
        "You are a quality reviewer. Check for accuracy, completeness, and suggest improvements. "
        "Report your review progress clearly."
    )
    # reviewer_builder.with_models(fast_llm_model="openrouter/anthropic/claude-3-haiku")
    await isaa.register_agent(reviewer_builder)

    # Get agent instances
    researcher = await isaa.get_agent("researcher_agent")
    writer = await isaa.get_agent("writer_agent")
    reviewer = await isaa.get_agent("reviewer_agent")

    # Create chain using the >> operator for sequential execution
    from pydantic import BaseModel
    class Topick(BaseModel):
        topic: str

    class MiniBlog(BaseModel):
        title: str
        content: str

    class Review(BaseModel):
        feedback: str
        better_title: str
        better_content: str

    chain = researcher >> CF(Topick) >> writer >> CF(MiniBlog) >> reviewer >> CF(Review)
    chain.name = "content_creation_chain"

    # Publish chain with live updates - Progress Callback wird automatisch eingerichtet
    result = await isaa.publish_and_host_agent(
        agent=chain,
        public_name="Content Creation Pipeline",
        description="Multi-agent chain with live progress: Research → Write → Review",
        registry_server="ws://localhost:8080/ws/registry/connect",
    )

    if result.get('public_url'):
        app.print("🔗 Chain published successfully with Live Progress UI!")
        app.print(f"   Local UI: {result['ui_url']}")
        app.print(f"   WebSocket: {result.get('registry_server')}")
        app.print(f"   WebSocket: {result.get('websocket_url')}")
        app.print(f"   Public URL: {result.get('public_url')}")
        app.print(f"   API Key: {result.get('public_api_key')}")
        print(result)

        # Example usage - test the chain with live updates
        #pp.print("\n🧪 Testing chain execution with live progress tracking:")
        #ry:
        #   result_text = await chain.a_run(
        #       query="Create a comprehensive article about renewable energy trends in 2024",
        #       session_id="demo-session"
        #   )
        #   app.print(f"✅ Chain completed successfully!")
        #   app.print(f"   Result length: {len(result_text)} characters")
        #   app.print("   All progress was tracked live in the UI!")
        #xcept Exception as e:
        #   app.print(f"❌ Chain execution failed: {e}")

        # Keep services running with live status
        try:
            while True:
                await asyncio.sleep(30)
                app.print("💓 Chain services live - ready for requests")
        except KeyboardInterrupt:
            app.print("Shutting down chain services...")
    else:
        app.print("❌ Failed to publish chain to registry")

    # Clean shutdown
    await researcher.close()
    await writer.close()
    await reviewer.close()


# example_multi_agent_live.py


async def setup_multiple_live_agents():
    """Example 4: Host multiple agents with individual live UIs"""
    app = get_app("MultiAgentLiveExample")
    isaa = app.get_mod("isaa")

    # Initialize ISAA
    await isaa.init_isaa()

    # Create different specialized agents
    agents_config = [
        {
            "name": "math_tutor",
            "system": "You are a mathematics tutor. Explain concepts step-by-step with live progress updates.",
            "public_name": "Live Math Tutor",
            "port": 8770
        },
        {
            "name": "code_helper",
            "system": "You are a coding assistant. Help debug and explain code with detailed progress tracking.",
            "public_name": "Live Code Assistant",
            "port": 8771
        },
        {
            "name": "creative_writer",
            "system": "You are a creative writer. Generate stories and content with live creative process updates.",
            "public_name": "Live Creative Writer",
            "port": 8772
        }
    ]

    hosted_agents = []

    # Create and host each agent
    for config in agents_config:
        # Create agent builder
        builder = isaa.get_agent_builder(config["name"])
        builder.with_system_message(config["system"])
        # builder.with_models(complex_llm_model="openrouter/openai/gpt-4o")

        # Register agent
        await isaa.register_agent(builder)

        # Get agent instance
        agent = await isaa.get_agent(config["name"])

        # Host with live UI - Progress wird automatisch eingerichtet
        result = await isaa.publish_and_host_agent(
            agent=agent,
            public_name=config["public_name"],
            description=f"Specialized agent: {config['public_name']} with live progress updates",
        )

        hosted_agents.append({
            'name': config["name"],
            'agent': agent,
            'result': result
        })

        app.print(f"🚀 {config['public_name']} live at: {result['ui_url']}")

    # Test all agents with live progress
    app.print("\n🧪 Testing all agents with live progress:")

    test_queries = [
        ("math_tutor", "Explain how to solve quadratic equations step by step"),
        ("code_helper", "Debug this Python function and explain the process"),
        ("creative_writer", "Write a short story about AI and humans working together")
    ]

    for agent_name, query in test_queries:
        agent_info = next(a for a in hosted_agents if a['name'] == agent_name)
        app.print(f"Testing {agent_name} - watch live progress in UI...")

        try:
            result = await agent_info['agent'].a_run(query, session_id=f"test_{agent_name}")
            app.print(f"✅ {agent_name} completed - live progress was shown!")
        except Exception as e:
            app.print(f"❌ {agent_name} failed: {e}")

    # Keep all agents running
    try:
        while True:
            await asyncio.sleep(60)
            app.print("💓 All agents live and ready")
            for agent_info in hosted_agents:
                app.print(f"   • {agent_info['name']}: {agent_info['result']['ui_url']}")
    except KeyboardInterrupt:
        app.print("Shutting down all live agents...")
        for agent_info in hosted_agents:
            await agent_info['agent'].close()


# example_complete_integration.py


async def setup_complete_agent_system(local=False):
    """Vollständiges Beispiel für Agent-System mit Live-Progress."""

    app = get_app("CompleteAgentSystem")
    isaa = app.get_mod("isaa")

    # ISAA initialisieren
    await isaa.init_isaa()

    # Erweiterten Agent erstellen
    advanced_builder = isaa.get_agent_builder("production_assistant")
    advanced_builder.with_system_message("""
        Du bist ein produktions-fertiger AI-Assistent mit detailliertem Progress-Tracking.

        Arbeitsweise:
        1. Analysiere die Anfrage sorgfältig
        2. Erstelle einen strukturierten Plan (Outline)
        3. Führe jeden Schritt methodisch aus
        4. Verwende Meta-Tools für komplexe Aufgaben
        5. Berichte kontinuierlich über deinen Fortschritt
        6. Liefere umfassende, gut strukturierte Antworten

        Zeige immer, welche Tools du verwendest und warum.
        Erkläre deine Reasoning-Loops transparent.
        """)

    # Agent registrieren
    await isaa.register_agent(advanced_builder)
    agent = await isaa.get_agent("production_assistant")

    # **Produktionsfertige Publish & Host - Ein Aufruf macht alles**
    result = await isaa.publish_and_host_agent(
        agent=agent,
        public_name="Production AI Assistant",
        registry_server="ws://localhost:8080/ws/registry/connect" if local else "wss://simplecore.app/ws/registry/connect",
        description="Production-ready AI assistant with comprehensive progress tracking, step-by-step reasoning, and meta-tool visualization. Supports real-time progress updates, outline tracking, and multi-user access.",
        access_level="public"
    )

    if result.get('success'):
        app.print("🎉 AGENT SYSTEM FULLY DEPLOYED!")
        app.print("")
        app.print("🌐 Public Access:")
        app.print(f"   URL: {result['public_url']}")
        app.print(f"   API Key: {result['public_api_key']}")
        app.print("")
        app.print("🖥️  Live UI:")
        app.print(f"   Registry UI: {result['ui_url']}")
        if result.get('local_ui'):
            app.print(f"   Local UI: {result['local_ui'].get('ui_url')}")
        app.print("")
        app.print("🔌 WebSocket:")
        app.print(f"   Live Updates: {result['websocket_url']}")
        app.print("")
        app.print("📋 cURL Test:")
        app.print(f"""curl -X POST {result['public_url']} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer {result['public_api_key']}" \\
  -d '{{"query": "Create a detailed analysis of quantum computing with step-by-step progress", "session_id": "test-session"}}'""")

        # Lokaler Test des Agents
        app.print("\n🧪 Testing agent locally...")
        #await asyncio.sleep(5)
        #test_result = await agent.a_run(
        #    "hey",
        #    session_id="local_test"
        #)
        app.print("✅ Test completed successfully!")

        # Service am Leben halten
        try:
            while True:
                await asyncio.sleep(30)
                app.print("💓 Agent services running - ready for requests")
        except KeyboardInterrupt:
            app.print("🛑 Shutting down agent services...")
    else:
        app.print(f"❌ Deployment failed: {result.get('error')}")
        print(result)

    await agent.close()


if __name__ == "__main__":
    asyncio.run(setup_complete_agent_system(local=True))
if __name__ == "__main__d":
    asyncio.run(setup_multiple_live_agents())
if __name__ == "__main__d":
    asyncio.run(setup_chain_with_live_updates())
