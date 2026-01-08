import ollama

EMBED_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "llama3.2:latest"


def get_embedding(text: str):
    response = ollama.embeddings(
        model=EMBED_MODEL,
        prompt=text
    )
    return response["embedding"]


def generate_answer(context: str, question: str):
    prompt = f"""
Use the context below to answer the question.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{question}
"""
    response = ollama.generate(
        model=LLM_MODEL,
        prompt=prompt
    )
    return response["response"]
