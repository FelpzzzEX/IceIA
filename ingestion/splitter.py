import json
import os
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.ingestion import IngestionPipeline

load_dotenv()

# definição do modelo de embedding
Settings.embed_model = OllamaEmbedding(
    model_name='qwen3-embedding:0.6b',
    base_url='http://localhost:11434'
)

# diretorios (talvez precise mudar, idk)
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
LOG_DIR = BASE_DIR / 'data' / 'chunked_log.txt'

# definição do banco vetorial (usando docker)
qdrant_client = QdrantClient(url='http://localhost:6333')
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name='documentos_iceia'
)

# banco de documentos para salvar os chunks maiores
doc_store = RedisDocumentStore.from_host_and_port(
    host='localhost',
    port=6379,
    namespace='documentos_iceia'
)

# liga docstore (chunks pais) e vector_store (chunks folha) para persistir e buscar
storage_context = StorageContext.from_defaults(
    docstore=doc_store,
    vector_store=vector_store
)

# usando uma abordagem hierarquica, os nós filhos sãp usados para buscar (mais preciso)
# os nós pais são usados para maior contexto
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],
    chunk_overlap=50
)

def load_log():
    if not LOG_DIR.exists():
        LOG_DIR.touch()
        return ""
    with open(LOG_DIR, 'r') as l:
        log = l.read()
    return log
        
def save_log(file_name: str):
    with open(LOG_DIR, 'a') as f:
        f.write(f'{file_name}\n')

def split_and_save(path: str):
    """
    Lê um JSON processado, divide o texto em nós hierárquicos
    e persiste caso ainda não tenha sido processado. 
    - nós pais no Redis Document Store;
    - nós folha no Qdrant para indexação vetorial.

    Args:
        path: caminho para o arquivo JSON processado.
    """
    file_name = Path(path).stem
    already_processed = load_log()
    
    # checa se o arquivo ja foi processado
    if file_name in already_processed:
        return
    
    print(f"Lendo o arquivo: {path}...")
    with open(path, 'r', encoding='utf-8') as j:
        markdown = json.load(j)

    if isinstance(markdown, dict):
        markdown = [markdown]

    documentos_brutos = [
        Document(
            text=item.get('page_content', ''),
            metadata=item.get('metadata', {})
        )
        for item in markdown
        if isinstance(item, dict)
    ]
    
    # todos os chunks gerados são salvos
    all_nodes = node_parser.get_nodes_from_documents(documentos_brutos)
    storage_context.docstore.add_documents(all_nodes)
    
    # separando os leaf e salvando no banco vetorial para busca
    leaf_nodes = get_leaf_nodes(all_nodes)
    index = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context
    )
    
    # salva o arquivo no log
    print(f'Arquivo {file_name} processado e adicionado ao log!')
    save_log(file_name)
        
if __name__ == "__main__":
    arquivos = list(PROCESSED_DIR.glob("*.json"))
    print(f"Iniciando ingestão de {len(arquivos)} arquivos...")
    
    for doc in arquivos:
        try:
            split_and_save(str(doc))
        except Exception as e:
            print(f"Erro ao processar {doc.name}: {e}")
            continue