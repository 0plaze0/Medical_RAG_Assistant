
import hashlib
from pathlib import Path

import streamlit as st

from medical_rag.config import Settings
from medical_rag.embeddings import LocalEmbeddingModel
from medical_rag.generation import (
    GenerationError,
    GeminiAnswerGenerator,
)
from medical_rag.ingestion import PDFIngestionError
from medical_rag.persistence import (
    PersistenceError,
    load_knowledge_base,
    save_knowledge_base,
)
from medical_rag.pipeline import MedicalRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parent
STORAGE_DIR = PROJECT_ROOT / "data" / "processed" / "knowledge_base"


st.set_page_config(
    page_title="MedicalRAG",
    layout="wide",
)

st.title("MedicalRAG")
st.caption(
    "Ask questions about your medical documents and inspect "
    "the source passages supporting each answer."
)


@st.cache_resource(show_spinner="Loading the local embedding model...")
def load_embedding_model(model_name: str) -> LocalEmbeddingModel:
    return LocalEmbeddingModel(model_name=model_name)


def initialize_pipeline(
    settings: Settings,
) -> tuple[MedicalRAGPipeline, dict[str, str]]:
    """Create the pipeline and restore the saved knowledge base."""
    embedding_model = load_embedding_model(settings.embedding_model)

    answer_generator = GeminiAnswerGenerator(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model,
    )

    pipeline = MedicalRAGPipeline(
        embedding_model=embedding_model,
        answer_generator=answer_generator,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    document_hashes = load_knowledge_base(
        pipeline=pipeline,
        storage_dir=STORAGE_DIR,
    )

    return pipeline, document_hashes


try:
    settings = Settings.from_env()
except ValueError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

if not settings.gemini_api_key:
    st.error(
        "Gemini API key is missing. Add GEMINI_API_KEY to your "
        ".env file and restart the application."
    )
    st.stop()


# Initialize the pipeline only once per Streamlit session.
if "pipeline" not in st.session_state:
    try:
        with st.spinner("Initializing MedicalRAG..."):
            pipeline, document_hashes = initialize_pipeline(settings)

        st.session_state.pipeline = pipeline
        st.session_state.processed_files = document_hashes
        st.session_state.chat_history = []

        if pipeline.chunk_count:
            st.toast("Saved knowledge base restored.")

    except PersistenceError as exc:
        st.error(
            f"Could not restore the saved knowledge base: {exc}"
        )
        st.info(
            "Inspect the files in data/processed/knowledge_base "
            "before attempting recovery. The application has not "
            "automatically overwritten them."
        )
        st.stop()

    except Exception as exc:
        st.error(f"Could not initialize MedicalRAG: {exc}")
        st.stop()


pipeline: MedicalRAGPipeline = st.session_state.pipeline
processed_files: dict[str, str] = st.session_state.processed_files


# Sidebar: document upload and knowledge-base status.
with st.sidebar:
    st.header("Knowledge base")

    uploaded_files = st.file_uploader(
        "Upload medical PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload documents you are authorized to use.",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            pdf_bytes = uploaded_file.getvalue()
            digest = hashlib.sha256(pdf_bytes).hexdigest()

            if processed_files.get(uploaded_file.name) == digest:
                continue

            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    chunk_count = pipeline.ingest_pdf(
                        source=pdf_bytes,
                        document_name=uploaded_file.name,
                    )

                    updated_hashes = {
                        **processed_files,
                        uploaded_file.name: digest,
                    }

                    # Persist the updated index and metadata before
                    # marking the upload as successfully processed.
                    save_knowledge_base(
                        pipeline=pipeline,
                        document_hashes=updated_hashes,
                        storage_dir=STORAGE_DIR,
                    )

                    st.session_state.processed_files = updated_hashes
                    processed_files = updated_hashes

                if chunk_count:
                    st.success(
                        f"Processed {uploaded_file.name}: "
                        f"{chunk_count} chunks"
                    )
                else:
                    st.warning(
                        f"No extractable text was found in "
                        f"{uploaded_file.name}. Scanned PDFs may "
                        "require OCR."
                    )

            except PDFIngestionError as exc:
                st.error(str(exc))

            except PersistenceError as exc:
                st.error(
                    f"Could not save the knowledge base: {exc}"
                )

            except Exception as exc:
                st.error(
                    f"Could not process {uploaded_file.name}: {exc}"
                )

    st.divider()

    st.subheader("Collection status")

    col1, col2 = st.columns(2)
    col1.metric("Documents", pipeline.document_count)
    col2.metric("Chunks", pipeline.chunk_count)

    st.caption(
        f"Persistent storage: {STORAGE_DIR.relative_to(PROJECT_ROOT)}"
    )

    st.caption(
        "Embeddings and vector search run locally. Questions and "
        "retrieved passages are sent to Gemini for answer generation."
    )


# Main question-answering interface.
st.subheader("Ask a question")

if pipeline.chunk_count == 0:
    st.info("Upload a text-based medical PDF to get started.")

question = st.text_area(
    "Your question",
    placeholder="Example: What symptoms are described in the document?",
    height=100,
)

top_k = st.slider(
    "Number of passages to retrieve",
    min_value=1,
    max_value=10,
    value=min(settings.top_k, 10),
)

ask_button = st.button(
    "Ask MedicalRAG",
    type="primary",
    disabled=(
        pipeline.chunk_count == 0
        or not question.strip()
    ),
)

if ask_button:
    try:
        with st.spinner(
            "Searching documents and generating an answer..."
        ):
            response = pipeline.ask(
                question=question.strip(),
                top_k=top_k,
            )

        result = response["answer"]

        st.session_state.chat_history.append(
            {
                "question": question.strip(),
                "answer": result["answer"],
                "citations": result["citations"],
                "retrieved_chunks": response["retrieved_chunks"],
            }
        )

    except GenerationError as exc:
        st.error(f"Answer generation failed: {exc}")

    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")


# Display previous questions and answers.
for item in reversed(st.session_state.chat_history):
    st.divider()

    st.markdown(f"**Question:** {item['question']}")
    st.markdown("### Answer")
    st.markdown(item["answer"])

    if item["citations"]:
        st.markdown("### Sources")

        for citation in item["citations"]:
            st.markdown(
                f"- **[{citation['source_id']}]** "
                f"{citation['document_name']} — "
                f"page {citation['page_number']}"
            )
    else:
        st.caption("No source citations were returned.")

    with st.expander("Inspect retrieved passages"):
        for chunk in item["retrieved_chunks"]:
            st.markdown(
                f"**{chunk['document_name']}** — "
                f"Page {chunk['page_number']}"
            )
            st.caption(
                f"Chunk ID: {chunk['chunk_id']} | "
                f"Similarity score: {chunk['score']:.4f}"
            )
            st.write(chunk["text"])
            st.divider()


st.caption(
    "This prototype provides document-grounded information, "
    "not a diagnosis or a substitute for professional medical care. "
    "Verify important claims against the original documents."
)
