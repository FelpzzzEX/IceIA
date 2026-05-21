from pathlib import Path
from llama_index.llms.ollama import Ollama
from llama_index.core.extractors import QuestionsAnsweredExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import Document
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SentenceSplitter,
    get_leaf_nodes
)

PROMPT_PTBR = (
    "Aqui está o contexto:\n"
    "----------------\n"
    "{context_str}\n"
    "----------------\n"
    "Dada a informação contextual acima, gere APENAS {num_questions} perguntas "
    "que podem ser respondidas de forma direta por este contexto. "
    "Escreva as perguntas em português do Brasil. "
    "Retorne apenas as perguntas, uma por linha, sem numeração, "
    "sem marcadores e sem textos adicionais."
)

# diretorios
BASE_DIR = Path(__file__).parent.parent

# abordagem hierarquica: nó filho para busca (mais preciso),
# nó pai para maior contexto (seção do markdown)
node_parsers = [
    MarkdownNodeParser(),
    SentenceSplitter(chunk_size=512, chunk_overlap=50)
]

node_parser = HierarchicalNodeParser(
    node_parser_ids=['markdown', 'sentence'],
    node_parser_map={
        'markdown': node_parsers[0],
        'sentence': node_parsers[1]
    }
)

llm = Ollama(
    model='gemma4:e2b',
    request_timeout=120,
    temperature=0.1
)

def split(json_docs: dict):
    """
    Lê um texto, divide em nós hierárquicos (Markdown -> Sentenças)
    e usa o Reverse HyDE nas folhas.

    Args:
        text: string contendo o documento.
        
    Returns:
        all_nodes: Lista com a hierarquia completa de nós
        leaf_nodes: Lista com os nós folha enriquecidos com as perguntas
    """

    text = json_docs['page_content']
    metadata = json_docs['metadata']

    docs = [
        Document(
            text=text,
            metadata=metadata,
            excluded_llm_metadata_keys=['source']
        )
    ]
    all_nodes = node_parser.get_nodes_from_documents(docs)

    # separa entre nó raiz e nó folha
    leaf_nodes = get_leaf_nodes(all_nodes)

    # filtra nós com conteúdo insuficiente
    leaf_nodes = [
        node for node in leaf_nodes
        if len(node.get_content().strip()) > 100
    ]

    # gera 3 perguntas para cada chunk (reverse hyde)
    qa_extractor = QuestionsAnsweredExtractor(
        llm=llm,
        questions=3,
        prompt_template=PROMPT_PTBR
    )
    
    # aplica as transformações
    pipeline = IngestionPipeline(
        transformations=[qa_extractor]
    )

    enriched_leaf_nodes = pipeline.run(nodes=leaf_nodes)

    return all_nodes, enriched_leaf_nodes