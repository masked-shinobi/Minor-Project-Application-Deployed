"""
RAG-Based Academic Research Paper Analyzer
==========================================
Main CLI interface for ingesting papers and querying the knowledge base.

Usage:
    python main.py ingest             # Ingest all PDFs from data/papers
    python main.py query "question"   # Query the knowledge base
    python main.py evaluate           # Run evaluation metrics
    python main.py security           # Run security tests
    python main.py stats              # Show system statistics
"""

import os
import sys
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def build_ingestion_pipeline():
    """Build and return the full ingestion pipeline components."""
    from ingestion.pdf_loader import PDFLoader
    from ingestion.document_parser import DocumentParser
    from processing.structure_analyzer import StructureAnalyzer
    from processing.boundary_detector import BoundaryDetector
    from processing.keyword_extractor import KeywordExtractor
    from processing.summary_generator import SummaryGenerator
    from processing.question_generator import QuestionGenerator
    from processing.table_parser import TableParser

    return {
        "loader": PDFLoader(),
        "parser": DocumentParser(),
        "structure_analyzer": StructureAnalyzer(),
        "boundary_detector": BoundaryDetector(max_chunk_size=512, overlap_size=50),
        "keyword_extractor": KeywordExtractor(max_keywords=10),
        "summary_generator": SummaryGenerator(),  # No LLM for offline mode
        "question_generator": QuestionGenerator(),
        "table_parser": TableParser(),
    }


def build_query_pipeline():
    """Build and return the full query pipeline components."""
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB
    from retrieval.retriever import Retriever
    from reasoning.llm_client import LLMClient
    from reasoning.planner import Planner
    from agents.retrieval_agent import RetrievalAgent
    from agents.summary_agent import SummaryAgent
    from agents.explanation_agent import ExplanationAgent
    from agents.verification_agent import VerificationAgent
    from reasoning.router import Router

    # Initialize components
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    # Load existing index if available
    index_dir = os.path.join(PROJECT_ROOT, "data")
    try:
        faiss_store.load(index_dir)
    except FileNotFoundError:
        print("[Main] No existing FAISS index found. Run 'ingest' first.")

    retriever = Retriever(faiss_store, metadata_db, embedder)

    # LLM-powered components
    llm_client = LLMClient()
    planner = Planner(llm_client)
    retrieval_agent = RetrievalAgent(retriever)
    summary_agent = SummaryAgent(llm_client)
    explanation_agent = ExplanationAgent(llm_client)
    verification_agent = VerificationAgent(llm_client)
    router = Router(planner, retrieval_agent, summary_agent, explanation_agent, verification_agent)

    return {
        "embedder": embedder,
        "faiss_store": faiss_store,
        "metadata_db": metadata_db,
        "retriever": retriever,
        "llm_client": llm_client,
        "router": router,
    }


def cmd_ingest(args):
    """Ingest PDF papers into the knowledge base."""
    print("=" * 60)
    print("  📄 Ingesting Research Papers")
    print("=" * 60)

    pipeline = build_ingestion_pipeline()
    from embeddings.embedder import Embedder
    from vectorstore.faiss_store import FAISSStore
    from database.metadata_db import MetadataDB

    # Discover PDFs
    pdf_paths = pipeline["loader"].discover_pdfs()
    if not pdf_paths:
        print("No PDF files found in data/papers/. Add PDFs and try again.")
        return

    print(f"Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"  • {os.path.basename(p)}")

    # Initialize embedding and storage
    embedder = Embedder()
    faiss_store = FAISSStore(embedding_dim=embedder.embedding_dim)
    metadata_db = MetadataDB()

    total_chunks = 0
    t_start = time.time()

    for pdf_path in pdf_paths:
        paper_name = pipeline["loader"].get_paper_name(pdf_path)
        print(f"\n--- Processing: {paper_name} ---")

        # Parse PDF
        doc = pipeline["parser"].parse(pdf_path, paper_id=paper_name)
        print(f"  Pages: {doc.total_pages} | Characters: {len(doc.full_text)}")

        # Analyze structure
        sections = pipeline["structure_analyzer"].analyze(doc.full_text)
        print(f"  Sections detected: {len(sections)}")
        for sec in sections:
            print(f"    • {sec.heading} ({len(sec.content)} chars)")

        # Chunk sections
        chunks = pipeline["boundary_detector"].chunk_document(sections, paper_id=paper_name)
        print(f"  Chunks created: {len(chunks)}")

        # Extract keywords for each chunk
        chunk_keywords = pipeline["keyword_extractor"].extract_from_chunks(chunks)

        # Generate summaries
        summaries = pipeline["summary_generator"].summarize_chunks(chunks)

        # Generate embeddings
        embeddings = embedder.embed_chunks(chunks)
        chunk_ids = [c.chunk_id for c in chunks]

        # Add to FAISS
        faiss_store.add_embeddings(embeddings, chunk_ids)

        # Store metadata in SQLite
        metadata_db.add_paper(
            paper_id=paper_name,
            title=doc.title,
            file_path=pdf_path,
            total_pages=doc.total_pages,
            metadata=doc.metadata
        )

        from database.metadata_db import ChunkMetadata
        chunk_meta_list = []
        for i, chunk in enumerate(chunks):
            keywords = chunk_keywords.get(chunk.chunk_id, [])
            chunk_meta_list.append(ChunkMetadata(
                chunk_id=chunk.chunk_id,
                paper_id=paper_name,
                paper_title=doc.title,
                section_heading=chunk.section_heading,
                content=chunk.content,
                summary=summaries[i] if i < len(summaries) else "",
                keywords=",".join(keywords),
                page_numbers="",
                char_start=chunk.char_start,
                char_end=chunk.char_end
            ))
        metadata_db.add_chunks_batch(chunk_meta_list)

        total_chunks += len(chunks)

    # Save FAISS index
    index_dir = os.path.join(PROJECT_ROOT, "data")
    faiss_store.save(index_dir)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  ✅ Ingestion Complete!")
    print(f"  Papers: {len(pdf_paths)} | Chunks: {total_chunks}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'=' * 60}")


def cmd_query(args):
    """Query the knowledge base."""
    query = args.query
    if not query:
        print("Please provide a query. Usage: python main.py query 'your question'")
        return

    print(f"\n🔍 Query: {query}\n")

    pipeline = build_query_pipeline()
    result = pipeline["router"].route(query)

    print(f"\n{'=' * 60}")
    print(f"  📝 Answer (Confidence: {result['confidence']})")
    print(f"{'=' * 60}")
    print(result["answer"])

    timing = result.get("timing", {})
    if timing:
        print(f"\n⏱  Timing: {sum(timing.values()):.2f}s total")
        for step, t in timing.items():
            print(f"   • {step}: {t:.3f}s")


def cmd_interactive(args):
    """Interactive query mode."""
    print("=" * 60)
    print("  🤖 RAG Research Paper Analyzer — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    pipeline = build_query_pipeline()

    while True:
        try:
            query = input("\n❓ Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline["router"].route(query)

        print(f"\n📝 Answer (Confidence: {result['confidence']}):")
        print("-" * 40)
        print(result["answer"])

        timing = result.get("timing", {})
        if timing:
            print(f"\n⏱  {sum(timing.values()):.2f}s")


def cmd_stats(args):
    """Show system statistics."""
    from database.metadata_db import MetadataDB
    db = MetadataDB()
    stats = db.get_stats()

    print(f"\n📊 System Statistics:")
    print(f"  Papers indexed: {stats['total_papers']}")
    print(f"  Total chunks: {stats['total_chunks']}")

    papers = db.list_papers()
    if papers:
        print(f"\n  Papers:")
        for paper in papers:
            print(f"    • {paper['title']} ({paper['total_pages']} pages)")


def cmd_evaluate(args):
    """Run evaluation metrics (13 metrics across 3 categories)."""
    from evaluation.evaluation_runner import EvaluationRunner

    pipeline = build_query_pipeline()
    runner = EvaluationRunner(
        router=pipeline["router"],
        embedder=pipeline.get("embedder"),
        llm_client=pipeline.get("llm_client")
    )

    if args.all:
        # Run full test dataset evaluation
        results = runner.run_all()
    elif args.query:
        # Run single query evaluation
        result = runner.run_single(
            query=args.query,
            ground_truth=""  # No ground truth for ad-hoc queries
        )

        # Print detailed single-query results
        print(f"\n{'=' * 70}")
        print(f"  [*] Evaluation for: '{args.query}'")
        print(f"{'=' * 70}")

        metrics = result["metrics"]
        for category in ["retrieval", "generation"]:
            if metrics.get(category):
                print(f"\n  -- {category.title()} --")
                for name, data in metrics[category].items():
                    if isinstance(data, dict) and "score" in data:
                        print(f"    {name:20s}  {data['score']:.3f}  {data.get('explanation', '')}")

        print(f"\n  Overall: {metrics.get('overall_score', 0):.3f}")

        results = {"per_query": [result], "aggregate": {}, "dataset_size": 1}
    else:
        # Default: run full dataset
        results = runner.run_all()

    # Save results if requested
    if args.save:
        runner.save_results(results)


def main():
    parser = argparse.ArgumentParser(
        description="RAG-Based Academic Research Paper Analyzer"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    subparsers.add_parser("ingest", help="Ingest PDFs from data/papers/")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("query", type=str, help="Your question")

    # Interactive command
    subparsers.add_parser("interactive", help="Interactive query mode")

    # Stats command
    subparsers.add_parser("stats", help="Show system statistics")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation metrics")
    eval_parser.add_argument("--query", type=str, default=None,
                             help="Single query to evaluate")
    eval_parser.add_argument("--all", action="store_true",
                             help="Run full test dataset evaluation")
    eval_parser.add_argument("--save", action="store_true",
                             help="Save results to data/evaluation_results.json")

    # Security command
    subparsers.add_parser("security", help="Run security tests")

    args = parser.parse_args()

    if args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "query":
        cmd_query(args)
    elif args.command == "interactive":
        cmd_interactive(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "security":
        print("Running security tests...")
        pipeline = build_query_pipeline()
        from security.prompt_injection_test import PromptInjectionTest
        tester = PromptInjectionTest(router=pipeline["router"])
        results = tester.run_all_tests()
        for category, data in results.items():
            if isinstance(data, dict) and "pass_rate" in data:
                print(f"  {category}: {data['pass_rate']:.0%} pass rate")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
