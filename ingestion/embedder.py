import json
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownNodeParser, SentenceSplitter, get_leaf_nodes
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

# definição do modelo de embedding
Settings.embed_model = OllamaEmbedding(
    model_name='qwen3-embedding:0.6b',
    base_url='http://localhost:11434'
)

# diretorios
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
LOG_DIR = BASE_DIR / 'data' / 'embedded_log.txt'

# banco vetorial (Qdrant via Docker)
qdrant_client = QdrantClient(url='http://localhost:6333')
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name='documentos_iceia'
)

# Redis apenas para leitura dos nós já salvos pelo splitter
doc_store = RedisDocumentStore.from_host_and_port(
    host='localhost',
    port=6379,
    namespace='documentos_iceia'
)

storage_context = StorageContext.from_defaults(
    docstore=doc_store,
    vector_store=vector_store
)

# mesma configuração de parsers do splitter para recriar a hierarquia
node_parsers = [
    MarkdownNodeParser(),
    SentenceSplitter(chunk_size=512, chunk_overlap=80),
    SentenceSplitter(chunk_size=128, chunk_overlap=20)
]

node_parser = HierarchicalNodeParser(
    node_parser_ids=['markdown', 'medium', 'small'],
    node_parser_map={
        'markdown': node_parsers[0],
        'medium': node_parsers[1],
        'small': node_parsers[2]
    }
)


def load_log() -> str:
    if not LOG_DIR.exists():
        LOG_DIR.touch()
        return ""
    with open(LOG_DIR, 'r') as l:
        return l.read()


def save_log(file_name: str):
    with open(LOG_DIR, 'a') as f:
        f.write(f'{file_name}\n')


def embed_and_index(path: str):
    """
    Lê um JSON processado, recria os nós hierárquicos, extrai os nós folha
    e os indexa no Qdrant com embeddings gerados pelo modelo configurado.
    Documentos já indexados são ignorados.

    Args:
        path: caminho para o arquivo JSON processado.
    """
    file_name = Path(path).stem
    log = load_log()
    embedded = set(log.splitlines())

    if file_name in embedded:
        print(f'{file_name} já indexado, pulando...')
        return

    print(f"Lendo o arquivo: {file_name}...")
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

    all_nodes = node_parser.get_nodes_from_documents(documentos_brutos)
    leaf_nodes = get_leaf_nodes(all_nodes)

    print(f'Gerando embeddings e indexando {len(leaf_nodes)} nós folha...')
    _ = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context
    )

    save_log(file_name)
    print(f'Arquivo {file_name} indexado e adicionado ao log!')


if __name__ == "__main__":
    arquivos = list(PROCESSED_DIR.glob("*.json"))
    print(f"Iniciando embedding de {len(arquivos)} arquivos...")
    for doc in arquivos:
        try:
            embed_and_index(str(doc))
        except Exception as e:
            print(f"Erro ao embedar {doc.name}: {e}")
            continue