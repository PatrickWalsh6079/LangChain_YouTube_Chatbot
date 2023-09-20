
from langchain import FAISS, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
load_dotenv()


def LLM_response():

    embeddings_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_name)
    # embeddings = OpenAIEmbeddings()
    vectordb = FAISS.load_local(embeddings=embeddings, folder_path='../index')

    prompt_template = """You are a helpful virtual AI assistant named VAL. Always provide a source.
            Use the following pieces of context: {context} and {history} to answer the question: {question}.
            Summarize the response, unless the user is asking you to generate something, in which case, be as detailed as possible.
                """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=prompt_template,
    )

    # Build a QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"),
        }
    )

    # Ask a question!
    # query = "Can the VA take away my disability status?"

    query = "Who is Paul?"

    response = qa_chain.run(query)

    return response


print(LLM_response())
