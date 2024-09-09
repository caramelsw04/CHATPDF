from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_openai import OpenAIEmbeddings
import os

# PDF 파일 로드
loader = PyPDFLoader("luckyday.pdf")
# PDF 파일을 페이지별로 분리하여 로드
pages = loader.load_and_split()

# 텍스트 분리 설정 (텍스트를 1000자씩 분리, 50자 오버랩)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,  # 각 청크의 크기 (1000자)
    chunk_overlap  = 50,  # 청크 간의 오버랩(50자)
    length_function = len,  # 길이 측정 함수로 len 사용
    is_separator_regex = False,  # 구분자를 정규 표현식으로 사용하지 않음
)
# 문서 페이지를 설정한 기준으로 분리
texts = text_splitter.split_documents(pages)

# OpenAI 임베딩 모델 로드 (텍스트 임베딩을 위해 사용)
embeddings_model = OpenAIEmbeddings()

# Chroma 데이터베이스가 저장될 디렉토리 경로 설정
persist_directory = './db/chromadb'

# Chroma 데이터베이스가 없으면 새로 생성, 있으면 불러오기
if not os.path.exists(persist_directory):
    # Chroma에 텍스트와 임베딩을 저장
    chromadb = Chroma.from_documents(
        texts,  # 분리된 텍스트
        embeddings_model,  # 임베딩 모델
        collection_name = 'esg',  # 컬렉션 이름
        persist_directory = persist_directory,  # 저장할 디렉토리
    )
else:
    # 기존에 저장된 Chroma 데이터베이스 불러오기
    chromadb = Chroma(
        persist_directory=persist_directory,  # 저장된 디렉토리 경로
        embedding_function=embeddings_model,  # 임베딩 모델
        collection_name='esg'  # 컬렉션 이름
    )
