# Censusbot

A proof-of-concept using Langchain to flexibly implement RAG using a codebook.

## Installation Instructions

### Set up the virtual environment

```bash
python -m venv .venv

source .venv/bin/activate

pip -r requirements.txt
```

### Set OpenAI API Key environment variable

```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Run Flask

```bash
flask run
```

Navigate to [http://127.0.0.1:5000]. Ask a question.
