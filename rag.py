import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Inicializar el modelo de lenguaje
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Inicializar los embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Inicializar el vector store
vector_store = InMemoryVectorStore(embeddings)

# Cargar y dividir los contenidos del blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
    ),
)
docs = loader.load()

# Dividir el texto en trozos más pequeños
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Indexar los documentos divididos en el vector store
vector_store.add_documents(documents=all_splits)

# Definir el prompt para la generación de respuestas
prompt = hub.pull("rlm/rag-prompt")

# Definir el estado de la aplicación
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Función de recuperación de documentos relevantes
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

# Función para generar la respuesta
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# Construir y compilar el flujo de trabajo de la aplicación
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])

