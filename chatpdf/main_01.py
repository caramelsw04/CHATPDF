from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("luckyday.pdf")
pages = loader.load_and_split()

print(pages[0].page_content)