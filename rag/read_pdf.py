from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader('attention.pdf')
# loader=PyPDFLoader('Jayesh-Das.pdf')
doc=loader.load()
# print(doc)
# print(len(doc))

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(doc)
# print(documents[:5])


from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# Initialize Olama embeddings
embeddings = OllamaEmbeddings(model="paraphrase-multilingual")

db = Chroma.from_documents(documents,embeddings)

# print(db)
# query = "Who are the authors of attention is all you need?"
query = "what is Positional Encoding ?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)