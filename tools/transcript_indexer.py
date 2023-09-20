
from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tools.youtuber_transcriber import transcriber

urls = ["https://www.youtube.com/watch?v=gWTa2lW9Qc4",
        "https://www.youtube.com/watch?v=3tuSis7yMf0",
        "https://www.youtube.com/watch?v=GS9tBtgaDlY",
        "https://www.youtube.com/watch?v=KyB1YdKfhSk"]


def load_docs():
    list_docs = transcriber(urls)

    # Combine doc
    combined_docs = [doc[0].page_content for doc in list_docs]
    text = " ".join(combined_docs)

    # Split them
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_text(text)

    # Build an index
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectordb = FAISS.from_texts(splits, embeddings)
    vectordb.save_local('../index')

    return 0


load_docs()
