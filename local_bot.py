from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM  # Updated import
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question: {question}"),
    ]
)

# Streamlit framework
st.title('LangChain Demo With LLAMA2 API')
input_text = st.text_input("Search the topic you want")

# Updated OllamaLLM class
llm = OllamaLLM(model="llama2")
output_parser = StrOutputParser()

# Create chain
chain = prompt | llm | output_parser

# Process user input
if input_text:
    st.write(chain.invoke({"question": input_text}))
