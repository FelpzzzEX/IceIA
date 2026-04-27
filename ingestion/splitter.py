import json
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from llama_index.core import Document, StorageContext
from llama_index.core.node_parser import HierarchicalNodeParser, MarkdownNodeParser, SentenceSplitter
from llama_index.storage.docstore.redis import RedisDocumentStore

load_dotenv()

# diretorios
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
LOG_DIR = BASE_DIR / 'data' / 'chunked_log.txt'

# banco de documentos para salvar os chunks (todos os níveis da hierarquia)
doc_store = RedisDocumentStore.from_host_and_port(
    host='localhost',
    port=6379,
    namespace='documentos_iceia'
)

storage_context = StorageContext.from_defaults(
    docstore=doc_store
)

# abordagem hierarquica: nó filho para busca (mais preciso),
# nó pai para maior contexto (seção do markdown)
node_parsers = [
    MarkdownNodeParser(),
    SentenceSplitter(chunk_size=512, chunk_overlap=50)
]

node_parser = HierarchicalNodeParser(
    node_parser_ids=['markdown', 'small'],
    node_parser_map={
        'markdown': node_parsers[0],
        'small': node_parsers[1]
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

def split_and_save(path: str):
    """
    Lê um JSON processado, divide o texto em nós hierárquicos
    e persiste todos os nós no Redis Document Store.
    Documentos já processados são ignorados.

    Args:
        path: caminho para o arquivo JSON processado.
    """
    file_name = Path(path).stem
    log = load_log()
    processed = set(log.splitlines())

    if file_name in processed:
        print('Arquivo já processado, pulando...')
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
    storage_context.docstore.add_documents(all_nodes)
    print(f'Salvando {len(all_nodes)} nós no Redis...')

    save_log(file_name)
    print(f'Arquivo {file_name} processado e adicionado ao log!')


if __name__ == "__main__":
    arquivos = list(PROCESSED_DIR.glob("*.json"))
    print(f"Iniciando chunking de {len(arquivos)} arquivos...")
    for doc in arquivos:
        try:
            split_and_save(str(doc))
        except Exception as e:
            print(f"Erro ao processar {doc.name}: {e}")
            continue