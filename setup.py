from setuptools import setup, find_packages

setup(
    name="psychmet-chatbot",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit>=1.29.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-openai>=0.0.5",
        "faiss-cpu>=1.7.4",
        "pypdf>=3.17.0",
        "python-dotenv>=1.1.1",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "sentence-transformers>=2.2.2",
        "tiktoken>=0.5.2",
        "openai>=1.0.0",
    ],
    python_requires=">=3.10",
)
