from langchain_docling.loader import ExportType, DoclingLoader
from docling_core.types.doc.document import ContentLayer
from pathlib import Path
import json
from typing import Optional, cast
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
import re
import unicodedata

# para esse arquivo ainda falta criar uma função que lê os arquivos brutos diretamente da fonte. Acredito que também seria interessante criar um dag para fazer esse processo de tempos em tempos.

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'

PROMPT = """Você é um assistente especializado em análise de documentos acadêmicos da UFOP.

Analise o texto abaixo e extraia os seguintes metadados:

- **tipo_documento**: Tipo do documento (ex: "Resolução", "Portaria", "Instrução Normativa")
- **departamento**: Órgão/conselho emissor (ex: "CEPE", "COSI", "Reitoria"). Extraia apenas a sigla ou nome do órgão.
- **curso**: Curso ao qual o documento se refere, se houver (ex: "Sistemas de Informação", "Engenharia de Produção"). Se for para todos os cursos ou não especificado, retorne null.
- **data**: Data de emissão do documento no formato YYYY-MM-DD. Se não encontrar, retorne null.

Retorne apenas os dados encontrados explicitamente no texto. Não invente informações.

Texto:
{text}"""
class MetadadosEstruturados(BaseModel):
    """Metadados extraídos de documentos acadêmicos da UFOP via LLM."""

    tipo_documento: Optional[str] = Field(
        default=None, 
        description='Tipo do documento (ex: "Resolução", "Portaria", "Instrução Normativa", "Edital")'
    )
    departamento: Optional[str] = Field(
        default=None, 
        description='Órgão/conselho emissor (ex: "CEPE", "COSI", "Reitoria"). Extraia apenas a sigla ou nome do órgão.'
    )
    curso: Optional[str] = Field(
        default=None, 
        description='Curso ao qual o documento se refere (ex: "Sistemas de Informação", "Engenharia de Produção"). Retorne null se for geral.'
    )
    data: Optional[str] = Field(
        default=None, 
        description='Data de emissão do documento no formato YYYY-MM-DD. Retorne null se não achar.'
    )

# definição do modelo gemini 2.5 flash para extrair metadados (usando api gratuita)
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-flash-lite',
    temperature=0
)
extrator = llm.with_structured_output(MetadadosEstruturados)

def load_data(path: str):
    """
    Carrega um PDF, extrai seu conteúdo em Markdown via Docling e salva em JSON
    na pasta processed. Metadados estruturados são extraídos via LLM e incorporados
    aos metadados de cada documento. Documentos já processados são ignorados.

    Args:
        path: Caminho absoluto para o arquivo PDF.
    """
    processed_file = PROCESSED_DIR / (Path(path).stem + ".json")
    
    if processed_file.exists():
        return
    
    loader = DoclingLoader(
        file_path=path,
        export_type=ExportType.MARKDOWN,
        md_export_kwargs={
            "included_content_layers": [ContentLayer.BODY]
        }
    )
    docs = loader.load()
    if not docs:
        return
    
    for doc in docs:
        doc.page_content = clean_markdown(doc.page_content)

    content = "\n".join(doc.page_content for doc in docs)
    metadados = extract_metadata(content)
    dict_metadados = metadados.model_dump(exclude_none=True)

    for i, doc in enumerate(docs):
        doc.id = f"{Path(path).stem}_{i}"
        doc.metadata["source"] = Path(path).name
        if dict_metadados:
            doc.metadata.update(dict_metadados)

    with open(processed_file, 'w', encoding='utf-8') as f:
        json.dump([doc.model_dump() for doc in docs], f)

def extract_metadata(text: str) -> MetadadosEstruturados:
    """
    Envia o texto do documento ao LLM para extração de metadados.
    Em caso de falha, aguarda 5 segundos e tenta novamente. Se a segunda tentativa
    também falhar, retorna um objeto vazio.

    Args:
        text: Conteúdo textual do documento.

    Returns:
        MetadadosEstruturados com os campos extraídos preenchidos.
    """
    try:
        return cast(MetadadosEstruturados, extrator.invoke(PROMPT.format(text=text)))
    except Exception as e:
        print(f"Erro na extração via LLM: {e}")
        time.sleep(5)
        try:
            return cast(MetadadosEstruturados, extrator.invoke(PROMPT.format(text=text)))
        except Exception:
            return MetadadosEstruturados()

def clean_markdown(texto: str) -> str:
    """Limpa ruídos de OCR e metadados visuais de PDFs acadêmicos."""
    texto = unicodedata.normalize("NFKC", texto).replace("\u00ad", "")

    # remove tags de imagem do Docling
    texto = texto.replace("<!-- image -->", "\n")

    # corrige hifenização por quebra de linha (informa-\nção -> informação)
    texto = re.sub(r"([A-Za-zÀ-ÖØ-öø-ÿ])-\n([A-Za-zÀ-ÖØ-öø-ÿ])", r"\1\2", texto)

    # remove cabeçalhos/rodapés repetitivos (linha a linha)
    padroes = [
        r"^\s*#*\s*[\·\-\*]?\s*MINIST[ÉE]RIO DA EDUCA[ÇC][ÃA]O.*$",
        r"^\s*#*\s*[\·\-\*]?\s*UNIVERSIDADE FEDERAL DE OURO PRETO.*$",
        r"^\s*#*\s*[\·\-\*]?\s*INSTITUTO DE CI[ÊE]NCIAS EXATAS E APLICADAS.*$",
        r"^\s*#*\s*[\·\-\*]?\s*COLEGIADO DO CURSO DE SISTEMAS DE INFORMA[ÇC][ÃA]O.*$",
        r"^\s*#*\s*REITORIA\s*$",
        r"^\s*P[ÁA]G\.?\s*\d+(\s*de\s*\d+)?\s*$"
    ]

    linhas_limpa = []
    for linha in texto.splitlines():
        l = linha.strip()
        if not l:
            linhas_limpa.append("")
            continue
        if any(re.match(p, l, flags=re.IGNORECASE) for p in padroes):
            continue
        linhas_limpa.append(l)

    texto = "\n".join(linhas_limpa)

    # normaliza espaçamento
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()

if __name__ == "__main__":
    for doc in RAW_DIR.glob("*.pdf"):
        load_data(str(doc))