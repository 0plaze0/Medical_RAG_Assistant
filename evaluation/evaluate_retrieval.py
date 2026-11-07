
import argparse
import csv
import json
import statistics
import time

from pathlib import Path

from medical_rag.chunking import ChunkRecord
from medical_rag.embeddings import (
    DEFAULT_MODEL_NAME,
    LocalEmbeddingModel,
)
from medical_rag.retrieval import VectorRetriever

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    ndcg_at_k,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "evaluation" / "datasets"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

CORPUS_FILE = DATASET_DIR / "retrieval_corpus.json"
QUERIES_FILE = DATASET_DIR / "retrieval_queries.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def convert_to_chunk_records(corpus: list[dict]) -> list[ChunkRecord]:
    """
    Convert RAGCare-QA corpus records into the existing
    ChunkRecord structure used by MedicalRAG.
    """

    chunks = []

    for index, record in enumerate(corpus):

        text = record["text"]
        metadata = record.get("metadata", {})

        try:
            page_number = int(metadata.get("page") or 0)
        except (ValueError, TypeError):
            page_number = 0

        chunk: ChunkRecord = {
            "chunk_id": record["chunk_id"],
            "document_name": (
                metadata.get("reference") or "RAGCare-QA"
            ),
            "page_number": page_number,
            "chunk_index": index,
            "start_char": 0,
            "end_char": len(text),
            "text": text,
        }

        chunks.append(chunk)

    return chunks


def evaluate_query(
    retriever: VectorRetriever,
    query_record: dict,
    ks: list[int],
):
    question = query_record["question"]
    relevant_ids = query_record["relevant_chunk_ids"]

    max_k = max(ks)

    start_time = time.perf_counter()

    retrieved_chunks = retriever.search(
        query=question,
        top_k=max_k,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    result = {
        "query_id": query_record["query_id"],
        "question": question,
        "specialty": query_record["metadata"].get("specialty"),
        "complexity": query_record["metadata"].get("complexity"),
        "rag_pipeline": query_record["metadata"].get("rag_pipeline"),
        "relevant_chunk_ids": relevant_ids,
        "retrieved_chunk_ids": retrieved_ids,
        "retrieval_latency_ms": round(latency_ms, 3),
    }

    for k in ks:

        result[f"precision@{k}"] = precision_at_k(
            retrieved_ids, relevant_ids, k
        )

        result[f"recall@{k}"] = recall_at_k(
            retrieved_ids, relevant_ids, k
        )

        result[f"mrr@{k}"] = reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, k
        )

        result[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_ids, relevant_ids, k
        )

    return result


def calculate_summary(results: list[dict], ks: list[int]):

    summary = {
        "total_queries": len(results),
        "metrics": {},
    }

    for k in ks:

        summary["metrics"][f"precision@{k}"] = round(
            statistics.mean(
                row[f"precision@{k}"] for row in results
            ),
            4,
        )

        summary["metrics"][f"recall@{k}"] = round(
            statistics.mean(
                row[f"recall@{k}"] for row in results
            ),
            4,
        )

        summary["metrics"][f"mrr@{k}"] = round(
            statistics.mean(
                row[f"mrr@{k}"] for row in results
            ),
            4,
        )

        summary["metrics"][f"ndcg@{k}"] = round(
            statistics.mean(
                row[f"ndcg@{k}"] for row in results
            ),
            4,
        )

    summary["average_retrieval_latency_ms"] = round(
        statistics.mean(
            row["retrieval_latency_ms"] for row in results
        ),
        3,
    )

    return summary


def save_results(results: list[dict], summary: dict):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "retrieval_evaluation.csv"
    json_path = RESULTS_DIR / "retrieval_evaluation.json"

    # Save individual query results.
    with open(csv_path, "w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # Save aggregate metrics and individual results.
    output = {
        "experiment": {
            "name": "RAGCare-QA Dense Retrieval Baseline",
            "retrieval_method": "FAISS cosine similarity",
            "embedding_model": summary["embedding_model"],
            "top_k_values": summary["top_k_values"],
        },
        "summary": summary["summary"],
        "per_query_results": results,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print(f"\nCSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate MedicalRAG dense retrieval."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Sentence Transformers embedding model.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu or cuda. Default: automatic.",
    )

    parser.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for retrieval metrics.",
    )

    args = parser.parse_args()

    ks = sorted(set(args.ks))

    if not ks or any(k <= 0 for k in ks):
        parser.error("All K values must be positive.")

    if not CORPUS_FILE.exists() or not QUERIES_FILE.exists():
        raise FileNotFoundError(
            "Evaluation datasets not found. "
            "Run prepare_ragcare.py first."
        )

    print("Loading evaluation datasets...")

    corpus = load_json(CORPUS_FILE)
    queries = load_json(QUERIES_FILE)

    if not corpus or not queries:
        raise ValueError("Corpus or query dataset is empty.")

    print(f"Corpus size: {len(corpus)}")
    print(f"Number of queries: {len(queries)}")

    print("\nConverting corpus into ChunkRecord format...")

    chunks = convert_to_chunk_records(corpus)

    print("\nLoading embedding model...")

    embedding_model = LocalEmbeddingModel(
        model_name=args.model,
        device=args.device,
    )

    print(f"Embedding model: {args.model}")
    print(f"Embedding dimension: {embedding_model.dimension}")

    print("\nBuilding FAISS index...")

    retriever = VectorRetriever(embedding_model)
    retriever.build_index(chunks)

    print(f"FAISS index size: {retriever.size}")

    print("\nRunning retrieval evaluation...")

    results = []

    for index, query in enumerate(queries, start=1):

        result = evaluate_query(
            retriever=retriever,
            query_record=query,
            ks=ks,
        )

        results.append(result)

        if index % 25 == 0 or index == len(queries):
            print(
                f"Processed {index}/{len(queries)} queries"
            )

    summary = calculate_summary(results, ks)

    summary["embedding_model"] = args.model
    summary["top_k_values"] = ks

    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 55)

    print(f"Total queries: {summary['total_queries']}")

    for metric, value in summary["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print(
        "Average retrieval latency: "
        f"{summary['average_retrieval_latency_ms']} ms"
    )

    save_results(results, {
        "embedding_model": args.model,
        "top_k_values": ks,
        "summary": summary,
    })


if __name__ == "__main__":
    main()
