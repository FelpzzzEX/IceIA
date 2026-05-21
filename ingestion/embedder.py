import json
from splitter import split
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.storage.docstore.postgres import PostgresDocumentStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# diretorios
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "data" / "embedded_log.txt"

DB_NAME = "rag_database"
DB_USER = "admin"
DB_PASSWORD = "admin"
DB_HOST = "localhost"
DB_PORT = "5432"

DB_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# definição do modelo de embedding
Settings.embed_model = OllamaEmbedding(model_name="embeddinggemma")

# docstore onde os o nó completo é salvo (contexto)
docstore = PostgresDocumentStore.from_uri(DB_URI)

# vector store onde as folhas ficam salvas (busca)
vector_store = PGVectorStore.from_params(
    database=DB_NAME,
    host=DB_HOST,
    user=DB_USER,
    password=DB_PASSWORD,
    port=DB_PORT,
    table_name="iceia_vetores",
    embed_dim=768,
)

# definindo o storage context
storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)

index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store, storage_context=storage_context
)

# funções para ler e salvar log
def load_log() -> set:
    if not LOG_DIR.exists():
        LOG_DIR.touch()
        return set()
    with open(LOG_DIR, "r") as l:
        return set(l.read().splitlines())

def save_log(file_name: str):
    with open(LOG_DIR, "a") as f:
        f.write(f"{file_name}\n")


def embed_and_index(path: str, already_processed: set):
    """
    Lê um JSON processado, recria os nós hierárquicos, extrai os nós folha
    e os indexa no Postgres (pgvector) com embeddings.
    """

    # lendo o log para evitar retrabalho
    file_name = Path(path).stem

    if file_name in already_processed:
        print("Arquivo já processado, pulando...")
        return

    else:
        print(f"Iniciando processamento do arquivo {file_name}...")
        # lendo os dados e passando para a função de split
        with open(path, "r", encoding="utf-8") as r:
            js = json.load(r)

        # função de chunking da etapa anterior
        all_nodes, leaf_nodes = split(js)

        # adiciona o nó completo na docstore
        storage_context.docstore.add_documents(all_nodes)

        # adiciona as folhas no vector store
        index.insert_nodes(leaf_nodes)

        # salva o nome do arquivo no log
        save_log(file_name)

if __name__ == "__main__":
    arquivos = list(PROCESSED_DIR.glob("*.json"))
    print(f"Iniciando embedding de {len(arquivos)} arquivos...")
    already_processed = load_log()
    for doc in arquivos:
        try:
            embed_and_index(str(doc), already_processed)
        except Exception as e:
            print(f"Erro ao processar {doc.name}: {e}")
            continue
