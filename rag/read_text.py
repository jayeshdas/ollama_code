from langchain_community.document_loaders import TextLoader
loader=TextLoader("sample.txt")
doc=loader.load()