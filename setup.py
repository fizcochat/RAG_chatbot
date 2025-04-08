from setuptools import setup, find_packages

setup(
    name="fiscozen-chatbot",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "streamlit-chat",
        "openai",
        "pinecone-client",
        "python-dotenv",
        "langchain",
        "langchain-openai",
        "langchain-pinecone",
        "fasttext",
        "tqdm",
        "PyPDF2",
        "psutil"
    ],
    python_requires=">=3.11",
) 