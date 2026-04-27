from docling.document_converter import DocumentConverter
from llama_parse import LlamaParse, ResultType
from pathlib import Path
import json
from typing import Optional, cast
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import time
import re
import unicodedata
import os
import logging

# para esse arquivo ainda falta criar uma função que lê os arquivos brutos diretamente da fonte. Acredito que também seria interessante criar um dag para fazer esse processo de tempos em tempos.

load_dotenv()

USE_CLOUD = True

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MARKDOWN_DIR = BASE_DIR / 'data' / 'markdown'

FOLDERS = [RAW_DIR, PROCESSED_DIR, MARKDOWN_DIR]

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

# caso a pasta não exista, é criada
for f in FOLDERS:
    os.makedirs(f, exist_ok=True)

# converter docling para processamento local
converter = DocumentConverter()

def save_json(path: Path, content: dict):
    with open(path, 'w', encoding='utf-8') as j:
            json.dump(content, j, ensure_ascii=False, indent=2)
        

def save_markdown(path: Path, content: str):
    with open(path, 'w', encoding='utf-8') as m:
            m.write(content)

def local_load_data(path: str):
    """
    Carrega um PDF, extrai seu conteúdo em Markdown via Docling e salva em JSON
    na pasta processed (incluindo metadados) e o markdown na pasta markdown (inspeção).
    Metadados estruturados são extraídos via LLM e incorporados aos
    metadados de cada documento.
    Documentos já processados são ignorados.
    
    Args:
        path: Caminho absoluto para o arquivo PDF.
    """
    file_path = Path(path)
    
    # arquivo JSON para etapas futuras e .md para inspeção
    PROCESSED_FILE = PROCESSED_DIR / f"{file_path.stem}.json"
    MARKDOWN_FILE = MARKDOWN_DIR / f"{file_path.stem}.md"
    
    # retorna se já tiverem sido processados
    if PROCESSED_FILE.exists() and MARKDOWN_FILE.exists():
        logging.info(f"Documento {file_path.name} já processado. Ignorando.")
        return

    try:
        # conversão
        result = converter.convert(str(file_path))
        doc = result.document
        content_md = doc.export_to_markdown()

        if not content_md:
            logging.warning(f"O documento {file_path.name} retornou vazio.")
            return

        # markdown limpo
        content_md = clean_markdown(content_md)

        # extração estruturada de metadados com LLM
        metadados = extract_metadata(content_md)
        dict_metadados = metadados.model_dump(exclude_none=True)

        documento = {
            "id": f"{file_path.stem}_0",
            "metadata": {
                "source": file_path.name,
                **dict_metadados,
            },
            "page_content": content_md
        }

        # salva JSON com os metadados
        save_json(PROCESSED_FILE, documento)
        
        # salva o markdown
        save_markdown(MARKDOWN_FILE, content_md)
            
        logging.info(f"Documento {file_path.name} processado com sucesso.")

    except Exception as e:
        logging.error(f"Erro ao processar {file_path.name}: {e}")

def cloud_load_data(path: str):
    file_path = Path(path)
    
    # arquivo JSON para etapas futuras e .md para inspeção
    PROCESSED_FILE = PROCESSED_DIR / f"{file_path.stem}.json"
    MARKDOWN_FILE = MARKDOWN_DIR / f"{file_path.stem}.md"
    
    # retorna se já tiverem sido processados
    if PROCESSED_FILE.exists() and MARKDOWN_FILE.exists():
        logging.info(f"Documento {file_path.name} já processado. Ignorando.")
        return

    try:
        # llamaparser para processamento na nuvem
        # usa chave de api definida no env
        parser = LlamaParse(
            result_type=ResultType.MD,
            language='pt',
            hide_headers=True,
            hide_footers=True,
            split_by_page=False,
            premium_mode=True,
            show_progress=True
        )
        
        result = parser.load_data(path)
        # juntando texto e extraindo metadados
        content = ''.join(doc.text for doc in result)
        metadados = extract_metadata(content)
        
        dict_metadados = metadados.model_dump(exclude_none=True)
        
        documento = {
            "id": f"{file_path.stem}_0",
            "metadata": {
                "source": file_path.name,
                **dict_metadados,
            },
            "page_content": content
        }
        
        # salva JSON com os metadados
        save_json(PROCESSED_FILE, documento)
        
        # salva o markdown
        save_markdown(MARKDOWN_FILE, content)
        
        logging.info(f"Documento {file_path.name} processado com sucesso.")

    except Exception as e:
        logging.error(f"Erro ao processar {file_path.name}: {e}")
    
   
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

    # remove cabeçalhos/rodapés repetitivos (linha a linha), sem depender de um curso/campus específico
    padroes = [
        r"^\s*#*\s*[\·\-\*]?\s*MINIST[ÉE]RIO DA EDUCA[ÇC][ÃA]O.*$",
        r"^\s*#*\s*[\·\-\*]?\s*UNIVERSIDADE\s+FEDERAL\b.*$",
        r"^\s*#*\s*[\·\-\*]?\s*INSTITUTO\b.*$",
        r"^\s*#*\s*[\·\-\*]?\s*COLEGIADO\s+(DO|DE)\s+CURSO\b.*$",
        r"^\s*#*\s*[\·\-\*]?\s*REITORIA\s*$",
        r"^\s*#*\s*[\·\-\*]?\s*CAMPUS\b.*$",
        r"^\s*P([ÁA]G|[ÁA]GINA)\.?\s*\d+(\s*(de|/)\s*\d+)?\s*$",
        r"^\s*(Rua|Av\.?|Avenida|Rod\.?|Rodovia)\s+.*\bCEP\b.*$"
    ]
    padroes_compilados = [re.compile(p, flags=re.IGNORECASE) for p in padroes]

    palavras_institucionais = (
        "universidade", "instituto", "colegiado", "departamento", "campus",
        "reitoria", "pro-reitoria", "pro reitoria", "secretaria", "diretoria"
    )

    def _normalize_ascii_minusculo(valor: str) -> str:
        decomposto = unicodedata.normalize("NFKD", valor)
        sem_acentos = "".join(ch for ch in decomposto if not unicodedata.combining(ch))
        return sem_acentos.casefold()

    linhas_limpa = []
    for linha in texto.splitlines():
        l = linha.strip()
        if not l:
            linhas_limpa.append("")
            continue
        if any(p.match(l) for p in padroes_compilados):
            continue

        # heurística conservadora para cabeçalhos em caixa alta com termos institucionais.
        # evita acoplamento a nomes específicos de unidade/campus.
        l_sem_marcadores = re.sub(r"^[#\-\*\·\s]+", "", l)
        tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", l_sem_marcadores)
        if tokens:
            tokens_maiusculos = sum(1 for t in tokens if t == t.upper())
            razao_maiusculas = tokens_maiusculos / len(tokens)
            l_normalizada = _normalize_ascii_minusculo(l_sem_marcadores)
            tem_palavra_institucional = any(p in l_normalizada for p in palavras_institucionais)
            if razao_maiusculas >= 0.7 and tem_palavra_institucional and len(l_sem_marcadores) <= 120:
                continue

        l_normalizada = _normalize_ascii_minusculo(l_sem_marcadores)
        if " bairro " in f" {l_normalizada} " and " cep " in f" {l_normalizada} ":
            continue
        linhas_limpa.append(l)

    texto = "\n".join(linhas_limpa)

    # normaliza espaçamento
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()

if __name__ == "__main__":
    # usa o parser na cluod
    if USE_CLOUD:
        for doc in RAW_DIR.glob("*.pdf"):
            cloud_load_data(str(doc))
    
    # caso contrário, usa o parser local (docling)
    else:
        for doc in RAW_DIR.glob("*.pdf"):
            local_load_data(str(doc))

