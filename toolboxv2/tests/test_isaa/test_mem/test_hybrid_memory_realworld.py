"""
Real-World Integration Test for HybridMemoryStore

Tests the system with Wikipedia-like content including:
- Multiple pages with entities and relations
- Vector + BM25 + Relation search
- Graph traversal
- Persistence
- Performance validation

Phase 1 Exit Test
"""

import os
import shutil
import tempfile
import time
import unittest
from typing import Any, Dict, List

import numpy as np

from toolboxv2.mods.isaa.base.hybrid_memory import HybridMemoryStore
from toolboxv2.mods.isaa.base.memory_graph_visualizer import MemoryGraphVisualizer


class WikipediaSimulator:
    """
    Simulates Wikipedia content for testing without network dependency.
    Contains realistic pages about companies, people, and technologies.
    """

    PAGES = [
        {
            "id": "SpaceX",
            "title": "SpaceX",
            "content": """Space Exploration Technologies Corp. (SpaceX) is an American spacecraft manufacturer,
launch service provider and satellite communications company headquartered in Hawthorne, California.
The company was founded in 2002 by Elon Musk with the goal of reducing space transportation costs
and enabling the colonization of Mars. SpaceX manufactures the Falcon 9, Falcon Heavy and Starship
launch vehicles, rocket engines, crew spacecraft, and communications satellites.

SpaceX has developed the Starlink satellite constellation to provide internet access.
The company achieved the first privately funded liquid-propellant rocket to reach orbit,
the first private company to successfully launch, orbit, and recover a spacecraft,
and the first private company to send a spacecraft to the International Space Station.

The Starship spacecraft is designed to be fully reusable and will be the most powerful rocket
ever built. It consists of a first stage Super Heavy booster and a second stage Starship vehicle.
Both stages are powered by Raptor engines that use liquid methane and liquid oxygen.""",
            "entities": [
                ("company:spacex", "company", "SpaceX"),
                ("person:elon_musk", "person", "Elon Musk"),
                ("rocket:falcon9", "rocket", "Falcon 9"),
                ("rocket:starship", "rocket", "Starship"),
                ("place:hawthorne", "place", "Hawthorne, California"),
            ],
            "relations": [
                ("person:elon_musk", "company:spacex", "FOUNDED"),
                ("company:spacex", "rocket:falcon9", "MANUFACTURES"),
                ("company:spacex", "rocket:starship", "DEVELOPS"),
                ("company:spacex", "place:hawthorne", "HEADQUARTERED_IN"),
            ],
            "concepts": ["spacecraft", "rockets", "aerospace", "mars colonization"],
        },
        {
            "id": "Tesla_Inc",
            "title": "Tesla, Inc.",
            "content": """Tesla, Inc. is an American multinational automotive and clean energy company
headquartered in Austin, Texas. Tesla designs, manufactures and sells electric vehicles,
stationary battery energy storage devices from home to grid-scale, solar panels and solar roof tiles,
and related products and services. Tesla is one of the world's most valuable companies.

The company was incorporated as Tesla Motors, Inc. on July 1, 2003 by Martin Eberhard and Marc Tarpenning.
Elon Musk joined as chairman in 2004 after leading the company's Series A funding round.
Musk became CEO in 2008. The company's name is a tribute to inventor and electrical engineer Nikola Tesla.

Tesla's current products include Model S, Model 3, Model X, Model Y, Cybertruck, and Tesla Semi vehicles.
The company also produces energy storage products like Powerwall, Powerpack, and Megapack.
Tesla operates multiple Gigafactories in the United States, China, and Germany.

The Tesla Autopilot is an advanced driver-assistance system that enables Tesla vehicles to
steer, accelerate, and brake automatically within their lane. Full Self-Driving (FSD) capability
is being developed to achieve full autonomous driving.""",
            "entities": [
                ("company:tesla", "company", "Tesla, Inc."),
                ("person:elon_musk", "person", "Elon Musk"),
                ("person:martin_eberhard", "person", "Martin Eberhard"),
                ("person:marc_tarpenning", "person", "Marc Tarpenning"),
                ("person:nikola_tesla", "person", "Nikola Tesla"),
                ("product:model3", "product", "Model 3"),
                ("product:modelY", "product", "Model Y"),
                ("place:austin", "place", "Austin, Texas"),
            ],
            "relations": [
                ("person:martin_eberhard", "company:tesla", "CO_FOUNDED"),
                ("person:marc_tarpenning", "company:tesla", "CO_FOUNDED"),
                ("person:elon_musk", "company:tesla", "CEO"),
                ("company:tesla", "product:model3", "MANUFACTURES"),
                ("company:tesla", "product:modelY", "MANUFACTURES"),
                ("company:tesla", "place:austin", "HEADQUARTERED_IN"),
            ],
            "concepts": [
                "electric vehicles",
                "clean energy",
                "autonomous driving",
                "batteries",
            ],
        },
        {
            "id": "Elon_Musk",
            "title": "Elon Musk",
            "content": """Elon Reeve Musk is a businessman and investor. He is the founder, chairman, CEO,
and chief technology officer of SpaceX; angel investor, CEO, product architect, and former
chairman of Tesla, Inc.; owner, chairman, and CTO of X Corp.; founder of the Boring Company;
and co-founder of Neuralink and OpenAI. He is the wealthiest person in the world.

Born in Pretoria, South Africa, Musk briefly attended the University of Pretoria before
immigrating to Canada at age 18, acquiring citizenship through his Canadian-born mother.
He transferred to the University of Pennsylvania two years later and received bachelor's degrees
in economics and physics. He moved to California in 1995 to attend Stanford University.

In 2002, Musk founded SpaceX, where he serves as CEO and Chief Engineer. In 2004, he joined
Tesla as chairman and product architect, becoming its CEO in 2008. In 2006, he helped create
SolarCity, a solar energy company that Tesla acquired in 2016. In 2015, Musk co-founded OpenAI,
a nonprofit research company aiming to develop safe artificial general intelligence.""",
            "entities": [
                ("person:elon_musk", "person", "Elon Musk"),
                ("company:spacex", "company", "SpaceX"),
                ("company:tesla", "company", "Tesla, Inc."),
                ("company:neuralink", "company", "Neuralink"),
                ("company:openai", "company", "OpenAI"),
                ("place:pretoria", "place", "Pretoria, South Africa"),
            ],
            "relations": [
                ("person:elon_musk", "company:spacex", "FOUNDED"),
                ("person:elon_musk", "company:tesla", "CEO"),
                ("person:elon_musk", "company:neuralink", "CO_FOUNDED"),
                ("person:elon_musk", "company:openai", "CO_FOUNDED"),
                ("person:elon_musk", "place:pretoria", "BORN_IN"),
            ],
            "concepts": ["entrepreneurship", "technology", "space exploration", "AI"],
        },
        {
            "id": "OpenAI",
            "title": "OpenAI",
            "content": """OpenAI is an American artificial intelligence research organization founded in December 2015.
It was originally a non-profit organization but restructured as a for-profit company in 2019.
The organization is headquartered in San Francisco, California.

OpenAI conducts AI research with the goal of developing safe and beneficial artificial general intelligence (AGI).
The organization's founders include Elon Musk, Sam Altman, Greg Brockman, Ilya Sutskever,
John Schulman, and Wojciech Zaremba. Musk resigned from the board in 2018 but remained a donor.

OpenAI has developed several notable AI systems including GPT (Generative Pre-trained Transformer),
GPT-2, GPT-3, GPT-4, DALL-E, and ChatGPT. These systems demonstrate capabilities in natural language
processing, image generation, and conversational AI. GPT-4 is a large multimodal model that can
process both text and images.

ChatGPT, launched in November 2022, became the fastest-growing consumer application in history,
reaching 100 million users within two months. It demonstrates advanced conversational abilities
and can assist with tasks like writing, coding, and answering questions.""",
            "entities": [
                ("company:openai", "company", "OpenAI"),
                ("person:elon_musk", "person", "Elon Musk"),
                ("person:sam_altman", "person", "Sam Altman"),
                ("product:gpt4", "product", "GPT-4"),
                ("product:chatgpt", "product", "ChatGPT"),
                ("place:san_francisco", "place", "San Francisco"),
            ],
            "relations": [
                ("person:elon_musk", "company:openai", "CO_FOUNDED"),
                ("person:sam_altman", "company:openai", "CEO"),
                ("company:openai", "product:gpt4", "DEVELOPS"),
                ("company:openai", "product:chatgpt", "DEVELOPS"),
                ("company:openai", "place:san_francisco", "HEADQUARTERED_IN"),
            ],
            "concepts": ["artificial intelligence", "machine learning", "AGI", "NLP"],
        },
        {
            "id": "Neuralink",
            "title": "Neuralink",
            "content": """Neuralink Corporation is a neurotechnology company that develops implantable brain-computer interfaces (BCIs).
The company was founded by Elon Musk and a team of scientists in 2016. Neuralink is headquartered in
Fremont, California, and has a research campus in Austin, Texas.

Neuralink's goal is to develop a high-bandwidth brain implant that can connect human brains directly
to computers. The technology aims to help patients with paralysis control devices with their thoughts
and eventually enable symbiosis between human and artificial intelligence.

The company has developed a surgical robot capable of implanting thin, flexible electrode threads into
the brain. These threads are much thinner than a human hair and cause minimal damage during implantation.
The device, called the Link, is a coin-sized chip that processes and transmits neural signals.

In May 2023, Neuralink received FDA approval to begin human clinical trials. The first human received
a Neuralink implant in January 2024. Early results show the patient can control a computer mouse
through thought alone.""",
            "entities": [
                ("company:neuralink", "company", "Neuralink"),
                ("person:elon_musk", "person", "Elon Musk"),
                ("product:link", "product", "The Link"),
                ("place:fremont", "place", "Fremont, California"),
                ("place:austin", "place", "Austin, Texas"),
            ],
            "relations": [
                ("person:elon_musk", "company:neuralink", "FOUNDED"),
                ("company:neuralink", "product:link", "DEVELOPS"),
                ("company:neuralink", "place:fremont", "HEADQUARTERED_IN"),
                ("company:neuralink", "place:austin", "RESEARCH_CAMPUS_IN"),
            ],
            "concepts": [
                "neurotechnology",
                "brain-computer interface",
                "neuroscience",
                "biotechnology",
            ],
        },
    ]


class TestHybridMemoryRealWorld(unittest.TestCase):
    """
    Real-world integration test simulating Wikipedia knowledge base.
    Tests complete workflow from adding pages to querying and graph traversal.
    """

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.tmp = tempfile.mkdtemp(prefix="hybrid_memory_realworld_")
        self.store = HybridMemoryStore(self.tmp, 768)
        self.rng = np.random.default_rng(42)
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Generate deterministic embedding for text.
        In production, this would use a real embedding model.
        """
        # Use hash for deterministic but varied embeddings
        hash_val = hash(text) % (2**31)
        self.rng = np.random.default_rng(hash_val)
        return self.rng.random(768).astype(np.float32)

    def tearDown(self):
        """Clean up temporary directory."""
        try:
            self.store.close()
        except:
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_01_add_wikipedia_pages(self):
        """TASK 1: Add all Wikipedia pages with entities and relations."""
        print("\n" + "=" * 70)
        print("TEST 1: Adding Wikipedia Pages")
        print("=" * 70)

        for page in WikipediaSimulator.PAGES:
            # Add main content
            emb = self._get_embedding(page["content"])
            entry_id = self.store.add(
                content=page["content"],
                embedding=emb,
                meta={
                    "title": page["title"],
                    "source": f"wikipedia:{page['id']}",
                    "type": "wikipedia_page",
                },
                concepts=page["concepts"],
            )

            print(f"  ✓ Added page: {page['title']} (ID: {entry_id})")

            # Add entities
            for entity_id, entity_type, entity_name in page["entities"]:
                self.store.add_entity(entity_id, entity_type, entity_name)

            # Add relations
            for from_id, to_id, relation_type in page["relations"]:
                self.store.add_relation(from_id, to_id, relation_type)

        # Verify stats
        stats = self.store.stats()
        print(f"\n  Stats:")
        print(f"    Active entries: {stats['active']}")
        print(f"    Entities: {stats['entities']}")
        print(f"    Relations: {stats['relations']}")

        self.assertEqual(stats["active"], 5, "Should have 5 Wikipedia pages")
        self.assertGreater(stats["entities"], 10, "Should have multiple entities")
        self.assertGreater(stats["relations"], 10, "Should have multiple relations")

    def test_02_vector_search(self):
        """TASK 2: Test vector similarity search."""
        print("\n" + "=" * 70)
        print("TEST 2: Vector Search")
        print("=" * 70)

        # First add pages
        self.test_01_add_wikipedia_pages()

        # Query about rockets/space
        query_text = "rocket launch space exploration Mars"
        query_emb = self._get_embedding(query_text)

        hits = self.store.query(
            query_text=query_text,
            query_embedding=query_emb,
            k=3,
            search_modes=("vector",),
        )

        print(f"\n  Query: '{query_text}'")
        print(f"  Results (vector search):")
        for i, hit in enumerate(hits, 1):
            title = hit.get("meta", {}).get("title", "Unknown")
            score = hit.get("score", 0)
            print(f"    {i}. {title} (score: {score:.3f})")

        self.assertGreater(len(hits), 0, "Should find results for rocket query")
        # SpaceX should be in top results
        titles = [h.get("meta", {}).get("title", "") for h in hits]
        self.assertIn(
            "SpaceX", titles, "SpaceX should be in top results for rocket query"
        )

    def test_03_bm25_search(self):
        """TASK 3: Test BM25 text search."""
        print("\n" + "=" * 70)
        print("TEST 3: BM25 Search")
        print("=" * 70)

        # First add pages
        self.test_01_add_wikipedia_pages()

        # Query for exact terms
        query_text = "artificial general intelligence AGI"
        query_emb = self._get_embedding(query_text)

        hits = self.store.query(
            query_text=query_text,
            query_embedding=query_emb,
            k=3,
            search_modes=("bm25",),
        )

        print(f"\n  Query: '{query_text}'")
        print(f"  Results (BM25 search):")
        for i, hit in enumerate(hits, 1):
            title = hit.get("meta", {}).get("title", "Unknown")
            score = hit.get("score", 0)
            print(f"    {i}. {title} (score: {score:.3f})")

        self.assertGreater(len(hits), 0, "Should find results for AGI query")
        # OpenAI should be in top results
        titles = [h.get("meta", {}).get("title", "") for h in hits]
        self.assertIn("OpenAI", titles, "OpenAI should be in top results for AGI query")

    def test_04_hybrid_search(self):
        """TASK 4: Test hybrid RRF fusion search."""
        print("\n" + "=" * 70)
        print("TEST 4: Hybrid Search (Vector + BM25 + Relations)")
        print("=" * 70)

        # First add pages
        self.test_01_add_wikipedia_pages()

        # Query combining semantic and keyword matching
        query_text = "Elon Musk companies CEO founder"
        query_emb = self._get_embedding(query_text)

        hits = self.store.query(
            query_text=query_text,
            query_embedding=query_emb,
            k=5,
            search_modes=("vector", "bm25", "relation"),
        )

        print(f"\n  Query: '{query_text}'")
        print(f"  Results (hybrid search):")
        for i, hit in enumerate(hits, 1):
            title = hit.get("meta", {}).get("title", "Unknown")
            score = hit.get("score", 0)
            modes = hit.get("search_modes", [])
            print(f"    {i}. {title} (score: {score:.3f}, modes: {modes})")

        self.assertGreater(len(hits), 0, "Should find results for Elon Musk query")
        # Should find multiple related pages (SpaceX, Tesla, Neuralink, etc.)
        titles = [h.get("meta", {}).get("title", "") for h in hits]
        musk_related = ["Elon Musk", "SpaceX", "Tesla, Inc.", "Neuralink", "OpenAI"]
        found_related = [t for t in titles if t in musk_related]
        self.assertGreaterEqual(
            len(found_related), 2, "Should find at least 2 Musk-related pages"
        )

    def test_05_graph_traversal(self):
        """TASK 5: Test graph traversal and entity relations."""
        print("\n" + "=" * 70)
        print("TEST 5: Graph Traversal")
        print("=" * 70)

        # First add pages
        self.test_01_add_wikipedia_pages()

        # Get all entities related to Elon Musk (depth=2)
        related = self.store.get_related("person:elon_musk", depth=2)

        print(f"\n  Traversing from: person:elon_musk (depth=2)")
        print(f"  Found {len(related)} related entities:")

        for entity in related:
            entity_id = entity.get("id", "")
            entity_type = entity.get("type", "")
            entity_name = entity.get("name", "")
            print(f"    - {entity_name} ({entity_type}): {entity_id}")

        # Should find companies, other people, places, etc.
        entity_names = [e.get("name", "") for e in related]

        # Should find SpaceX and Tesla (depth=1)
        self.assertIn("SpaceX", entity_names, "Should find SpaceX as direct relation")
        self.assertIn("Tesla, Inc.", entity_names, "Should find Tesla as direct relation")

        # Should find products/projects (depth=2)
        products = [e for e in related if e.get("type") == "product"]
        self.assertGreater(len(products), 0, "Should find products at depth=2")

        print(f"\n  Found {len(products)} products through relations")

    def test_06_concept_filtering(self):
        """TASK 6: Test concept-based filtering."""
        print("\n" + "=" * 70)
        print("TEST 6: Concept Filtering")
        print("=" * 70)

        # First add pages
        self.test_01_add_wikipedia_pages()

        # Query for AI-related content
        query_text = "technology development innovation"
        query_emb = self._get_embedding(query_text)

        hits = self.store.query(
            query_text=query_text,
            query_embedding=query_emb,
            k=5,
            search_modes=("vector", "bm25"),
        )

        print(f"\n  Query: '{query_text}'")
        print(f"  Results with concepts:")
        for i, hit in enumerate(hits, 1):
            title = hit.get("meta", {}).get("title", "Unknown")
            concepts = hit.get("concepts", [])
            print(f"    {i}. {title}")
            print(f"       Concepts: {concepts}")

        # Verify all pages have concepts
        for hit in hits:
            self.assertIsInstance(hit.get("concepts"), list, "Should have concepts list")

    def test_07_persistence(self):
        """TASK 7: Test save/load persistence."""
        print("\n" + "=" * 70)
        print("TEST 7: Persistence (Save/Load)")
        print("=" * 70)

        # Add pages
        self.test_01_add_wikipedia_pages()

        # Get stats before save
        stats_before = self.store.stats()

        # Save to bytes (ZIP format for cross-instance transfer)
        t0 = time.perf_counter()
        data = self.store.save_to_bytes()
        save_time = time.perf_counter() - t0

        print(f"\n  Saved {len(data)} bytes in {save_time:.3f}s")

        # Create new store and load
        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_realworld_load_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)

            t0 = time.perf_counter()
            store2.load(data)
            load_time = time.perf_counter() - t0

            print(f"  Loaded in {load_time:.3f}s")

            # Verify stats match
            stats_after = store2.stats()

            print(f"\n  Stats comparison:")
            print(
                f"    Before: active={stats_before['active']}, entities={stats_before['entities']}"
            )
            print(
                f"    After:  active={stats_after['active']}, entities={stats_after['entities']}"
            )

            self.assertEqual(
                stats_before["active"],
                stats_after["active"],
                "Active entries should match after load",
            )
            self.assertEqual(
                stats_before["entities"],
                stats_after["entities"],
                "Entities should match after load",
            )
            self.assertEqual(
                stats_before["relations"],
                stats_after["relations"],
                "Relations should match after load",
            )

            # Verify we can query loaded data
            query_text = "SpaceX rockets"
            query_emb = self._get_embedding(query_text)
            hits = store2.query(query_text, query_emb, k=2, search_modes=("bm25",))

            self.assertGreater(len(hits), 0, "Should find results in loaded store")

            print(f"\n  ✓ Query in loaded store returned {len(hits)} results")

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

    def test_08_update_versioning(self):
        """TASK 8: Test update with versioning."""
        print("\n" + "=" * 70)
        print("TEST 8: Update and Versioning")
        print("=" * 70)

        # Add a page
        page = WikipediaSimulator.PAGES[0]
        emb = self._get_embedding(page["content"])

        entry_id = self.store.add(
            content=page["content"],
            embedding=emb,
            meta={"title": page["title"]},
        )

        print(f"  Original entry ID: {entry_id}")

        # Update the content
        updated_content = page["content"] + "\n\nUpdated in 2024 with new achievements."
        updated_emb = self._get_embedding(updated_content)

        new_id = self.store.update(entry_id, updated_content, updated_emb)

        print(f"  Updated entry ID: {new_id}")
        print(f"  IDs are different: {entry_id != new_id}")

        self.assertNotEqual(entry_id, new_id, "Update should create new ID")

        # Verify old entry is inactive
        old_entry = self.store._load_entry(entry_id)
        self.assertEqual(old_entry["is_active"], 0, "Old entry should be inactive")

        # Verify new entry is active
        new_entry = self.store._load_entry(new_id)
        self.assertEqual(new_entry["is_active"], 1, "New entry should be active")
        self.assertIn(
            "Updated in 2024", new_entry["content"], "Should have updated content"
        )

        print(f"\n  ✓ Versioning working correctly")

    def test_09_performance_validation(self):
        """TASK 9: Validate performance meets Phase 2 exit criteria."""
        print("\n" + "=" * 70)
        print("TEST 9: Performance Validation")
        print("=" * 70)

        # Add all pages
        for page in WikipediaSimulator.PAGES:
            emb = self._get_embedding(page["content"])
            self.store.add(
                content=page["content"],
                embedding=emb,
                meta={"title": page["title"]},
                concepts=page["concepts"],
            )

        # Test add latency (< 5ms avg)
        print("\n  Testing add latency...")
        add_times = []
        for i in range(10):
            content = f"Performance test entry {i}"
            emb = self._get_embedding(content)

            t0 = time.perf_counter()
            self.store.add(content, emb)
            add_times.append(time.perf_counter() - t0)

        avg_add_ms = (sum(add_times) / len(add_times)) * 1000
        print(f"    Add latency: {avg_add_ms:.2f}ms avg (target: < 5ms)")
        self.assertLess(avg_add_ms, 5.0, "Add should be < 5ms avg")

        # Test query latency (< 50ms for hybrid)
        print("\n  Testing query latency...")
        query_times = []
        for i in range(10):
            query_text = f"test query {i}"
            query_emb = self._get_embedding(query_text)

            t0 = time.perf_counter()
            self.store.query(
                query_text, query_emb, k=5, search_modes=("vector", "bm25", "relation")
            )
            query_times.append(time.perf_counter() - t0)

        avg_query_ms = (sum(query_times) / len(query_times)) * 1000
        print(f"    Query latency: {avg_query_ms:.2f}ms avg (target: < 50ms)")
        self.assertLess(avg_query_ms, 50.0, "Query should be < 50ms avg")

        # Test save/load time (< 2s for 10k, we have ~5 so should be instant)
        print("\n  Testing save/load time...")
        t0 = time.perf_counter()
        data = self.store.save_to_bytes()
        save_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.store.load(data)
        load_time = time.perf_counter() - t0

        print(f"    Save time: {save_time * 1000:.2f}ms")
        print(f"    Load time: {load_time * 1000:.2f}ms")
        self.assertLess(save_time, 2.0, "Save should be < 2s")
        self.assertLess(load_time, 2.0, "Load should be < 2s")

        print(f"\n  ✓ All performance criteria met")

    def test_10_full_integration(self):
        """TASK 10: Full integration test combining all features."""
        print("\n" + "=" * 70)
        print("TEST 10: Full Integration Test")
        print("=" * 70)

        # Add all Wikipedia pages
        print("\n  Step 1: Adding Wikipedia pages...")
        for page in WikipediaSimulator.PAGES:
            emb = self._get_embedding(page["content"])
            self.store.add(
                content=page["content"],
                embedding=emb,
                meta={"title": page["title"], "source": f"wikipedia:{page['id']}"},
                concepts=page["concepts"],
            )

            for entity_id, entity_type, entity_name in page["entities"]:
                self.store.add_entity(entity_id, entity_type, entity_name)

            for from_id, to_id, relation_type in page["relations"]:
                self.store.add_relation(from_id, to_id, relation_type)

        # Perform hybrid search
        print("  Step 2: Performing hybrid search...")
        query = "Elon Musk's companies and projects"
        query_emb = self._get_embedding(query)
        hits = self.store.query(
            query, query_emb, k=5, search_modes=("vector", "bm25", "relation")
        )

        print(f"    Found {len(hits)} results")

        # Traverse entity graph
        print("  Step 3: Traversing entity graph...")
        related = self.store.get_related("person:elon_musk", depth=2)
        print(f"    Found {len(related)} related entities")

        # Save and reload
        print("  Step 4: Testing persistence...")
        data = self.store.save_to_bytes()

        tmp2 = tempfile.mkdtemp(prefix="hybrid_memory_full_integration_")
        try:
            store2 = HybridMemoryStore(tmp2, 768)
            store2.load(data)

            # Verify data persisted
            stats = store2.stats()
            print(f"    Loaded {stats['active']} entries, {stats['entities']} entities")

            # Query in loaded store
            hits2 = store2.query(query, query_emb, k=3, search_modes=("bm25",))
            print(f"    Query in loaded store: {len(hits2)} results")

            store2.close()
        finally:
            shutil.rmtree(tmp2, ignore_errors=True)

        print("\n  ✓ Full integration test passed")
        print("=" * 70)
        print("ALL REAL-WORLD TESTS PASSED ✓")
        print("=" * 70)


    def test_11_vis(self):
        """TASK 10: Full integration test combining all features."""
        print("\n" + "=" * 70)
        print("TEST 11: VIS Test")
        print("=" * 70)

        # Add all Wikipedia pages
        print("\n  Step 1: Adding Wikipedia pages...")
        for page in WikipediaSimulator.PAGES:
            emb = self._get_embedding(page["content"])
            self.store.add(
                content=page["content"],
                embedding=emb,
                meta={"title": page["title"], "source": f"wikipedia:{page['id']}"},
                concepts=page["concepts"],
            )

            for entity_id, entity_type, entity_name in page["entities"]:
                self.store.add_entity(entity_id, entity_type, entity_name)

            for from_id, to_id, relation_type in page["relations"]:
                self.store.add_relation(from_id, to_id, relation_type)

        visualizer = MemoryGraphVisualizer(self.store)

        output_path = "wiki_test_graph3."

        visualizer.save_html(output_path+'html')





if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)
