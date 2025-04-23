def load_documents(data_path="./docs", file_pattern="**/*.md"):
    from langchain_community.document_loaders import DirectoryLoader

    loader = DirectoryLoader(path=data_path, glob=file_pattern)
    documents = loader.load()
    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source')}")
        # print(f"Content (first 150 chars): {doc.page_content[:150]}...")
    return documents
    # Print information about the first few documents


def split_documents(documents):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print("Chunking Done")
    return chunks


def generate_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    print("Inside generate_embeddings")
    return embedding_model


def build_vector_db():
    from langchain_community.vectorstores import Chroma

    documents = load_documents()
    chunks = split_documents(documents)
    embeddings = generate_embeddings()
    vector_db = Chroma.from_documents(
        documents, embeddings, persist_directory="./keploy-docs-coll"
    )
    return vector_db


def get_vector_db():
    from langchain_community.vectorstores import Chroma

    embeddings = generate_embeddings()
    vector_db = Chroma(
        persist_directory="./keploy-docs-coll", embedding_function=embeddings
    )
    return vector_db


# emb_model = generate_embeddings()


def answer_query(query, k=3):
    from dotenv import load_dotenv
    import os
    load_dotenv()
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    db = get_vector_db()
    results = db.similarity_search_with_relevance_scores(query, k=k)
    context_text = "\n\n-------------------------\n\n".join(
        [doc.page_content for doc, _score in results]
    )

    PROMPT_TEMPLATE = [
        """
        You are a smart and helpful AI assistant trained to answer technical questions using documentation.

        You are provided with multiple documentation snippets retrieved from a vector database. These are separated by lines containing only: 
        -------------------------

        Your goal is to:
        - Use the relevant information from any or all of the provided snippets to answer the user’s question.
        - If the question involves implementation or coding, provide a complete and accurate code snippet.
        - Do not exceed the examples to more than 1.
        - Keep the response short, concise, clear, and in a developer-friendly tone.
        - Format your output using Markdown (.md) e.g., code blocks, lists, bold, newlines.
        - Keep the tone formal, don't start with any words like "OKAY". Write direct answer what is required.
        - If the information is not directly present, reason based on the closest matches and explain your assumptions.

        Documentation:
        {context}

        User query:
        {query}

        Answer:

        """
            ]
    prompt_template = ChatPromptTemplate(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, query=query)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-exp-03-25", google_api_key=os.getenv("GEMINI_API_KEY")
    )
    response = llm.invoke(prompt)
    answer = response.content

    # Unique sources
    seen_sources = set()
    sources_list = []
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources:
            sources_list.append(source)
            seen_sources.add(source)
    return {"answer": answer, "sources": sources_list}

