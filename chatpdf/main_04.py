# main_04 : pdf파일 읽기 + splitting + embedding + db 저장 + 챗봇 기능 추가

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
import os

# PDF 파일 로더 설정 (luckyday.pdf 파일을 로드)
loader = PyPDFLoader("luckyday.pdf")
pages = loader.load_and_split()

# 텍스트 분리 (텍스트를 1000자 단위로 분할, 50자의 오버랩 설정)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 50,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages)

# OpenAI Embeddings 모델 로드 (텍스트 임베딩을 위한 설정)
embeddings_model = OpenAIEmbeddings()

# Chroma 데이터베이스 경로 설정 (임베딩된 텍스트를 저장할 디렉토리)
persist_directory = './db/chromadb'

# Chroma 데이터베이스가 없으면 생성, 있으면 불러오기
if not os.path.exists(persist_directory):
    chromadb = Chroma.from_documents(
        texts, 
        embeddings_model,
        collection_name = 'esg',
        persist_directory = persist_directory,
    )
else:
    chromadb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings_model,
        collection_name='esg'
    )

# 질문 설정 (사용자가 묻고 싶은 질문)
question = "아내가 먹고 싶어하는 음식은 무엇이야?"

# OpenAI 모델 로드 (GPT-3.5-turbo 모델 사용)
llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

# MultiQueryRetriever 생성 (다중 쿼리 검색을 위해 retriever 생성)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=chromadb.as_retriever(), llm=llm
)

# 문서 검색 (질문을 사용해 관련 문서 검색)
docs = retriever_from_llm.invoke({"query": question})

# 검색된 문서 수 및 첫 300자 출력
print(f"검색된 문서 수: {len(docs)}")
for i, doc in enumerate(docs):
    print(f"문서 {i+1}: {doc.page_content[:300]}...")  
    print("---")

# RetrievalQA 체인 생성 (질문에 답변하기 위한 체인 구성)
qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=chromadb.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
            )

# 질문에 대한 답변 생성
result = qa_chain.invoke({"query": question})

# 답변 출력
print(result)
