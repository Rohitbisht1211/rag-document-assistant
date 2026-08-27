import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("Gemini API key not found.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚"
)

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions about it.")


# Upload PDF
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file is not None:

    # Read PDF
    reader = PdfReader(uploaded_file)

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text:

            documents.append(
                Document(
                    page_content=page_text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number
                    }
                )
            )

    st.success("✅ PDF successfully loaded!")

    st.write(
        f"📄 Pages found: {len(documents)}"
    )


    # Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(
        documents
    )

    st.write(
        f"🧩 Chunks created: {len(chunks)}"
    )


    # Create Knowledge Base
    if st.button("Create Knowledge Base"):

        with st.spinner(
            "Creating embeddings and vector database..."
        ):

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vector_store = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state["vector_store"] = vector_store

        st.success(
            "✅ Knowledge base created successfully!"
        )

        st.write(
            "Your PDF is now converted into searchable vectors."
        )


    # Ask Questions
    st.divider()

    st.subheader(
        "💬 Ask a question about your document"
    )

    question = st.text_input(
        "Enter your question:"
    )


    if question:

        if "vector_store" not in st.session_state:

            st.warning(
                "Please create the knowledge base first."
            )

        else:

            vector_store = st.session_state[
                "vector_store"
            ]


            # Retrieve relevant chunks
            results = vector_store.similarity_search(
                question,
                k=3
            )


            # Build context
            context_parts = []

            for doc in results:

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                page = doc.metadata.get(
                    "page",
                    "Unknown"
                )

                context_parts.append(
                    f"Source: {source}\n"
                    f"Page: {page}\n"
                    f"Content:\n{doc.page_content}"
                )

            context = "\n\n".join(
                context_parts
            )


            # Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-3.6-flash",
                google_api_key=api_key
            )


            # RAG prompt
            prompt = f"""
You are an AI assistant that answers questions
about a user's document.

Use the retrieved document context below to
answer the user's question.

IMPORTANT RULES:

1. Answer using information present in the context.
2. Do not invent or assume information.
3. If the answer is present, answer clearly.
4. If the answer is not present, say:
"I could not find this information in the document."
5. Keep the answer concise.
6. Do not mention the retrieval process.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

FINAL ANSWER:
"""


            # Generate answer
            with st.spinner(
                "🤖 Generating answer..."
            ):

                response = llm.invoke(
                    prompt
                )


            # Clean Gemini response
            if isinstance(
                response.content,
                list
            ):

                answer_parts = []

                for item in response.content:

                    if (
                        isinstance(item, dict)
                        and item.get("type") == "text"
                    ):

                        answer_parts.append(
                            item.get("text", "")
                        )

                answer_text = "".join(
                    answer_parts
                )

            else:

                answer_text = str(
                    response.content
                )


            # Display answer
            st.subheader("🤖 Answer")

            st.write(
                answer_text
            )


            # Display sources
            st.subheader("📚 Sources")

            for result in results:

                source = result.metadata.get(
                    "source",
                    "Unknown"
                )

                page = result.metadata.get(
                    "page",
                    "Unknown"
                )

                st.write(
                    f"📄 **{source}** — Page **{page}**"
                )