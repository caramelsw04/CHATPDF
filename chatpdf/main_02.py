from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#Loader
loader = PyPDFLoader("luckyday.pdf")
pages = loader.load_and_split()

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 500,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)
texts = text_splitter.split_documents(pages)

print(texts[0].page_content)