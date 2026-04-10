import argparse

from core.retriever import Retriever
from core.vector_store import VectorStore
from core.embeddings import EmbeddingModel


# -----------------------------
# SIMPLE LLM WRAPPER (OPENAI / LOCAL SWITCHABLE)
# -----------------------------
class Generator:
    def __init__(self, model_type="openai"):
        self.model_type = model_type

        if model_type == "openai":
            from openai import OpenAI
            self.client = OpenAI()
            self.model = "gpt-4o-mini"

        elif model_type == "local":
            from transformers import pipeline
            self.pipe = pipeline("text-generation", model="gpt2")

        else:
            raise ValueError("Unsupported model type")

    def generate(self, prompt: str) -> str:
        if self.model_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Answer based only on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content

        elif self.model_type == "local":
            output = self.pipe(prompt, max_length=300, num_return_sequences=1)
            return output[0]["generated_text"]


# -----------------------------
# BUILD PROMPT
# -----------------------------
def build_prompt(query: str, context: str) -> str:
    return f"""
Context:
{context}

Question:
{query}

Answer:
"""


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", type=str, default="text",
                        choices=["text", "hybrid", "text_to_image"])
    parser.add_argument("--model", type=str, default="openai",
                        choices=["openai", "local"])

    args = parser.parse_args()

    # Load stores
    text_store = VectorStore.load("stores/text_store")
    image_store = VectorStore.load("stores/image_store")

    retriever = Retriever(
        text_store=text_store,
        image_store=image_store,
        embedder=EmbeddingModel()
    )

    generator = Generator(model_type=args.model)

    # -----------------------------
    # RETRIEVAL
    # -----------------------------
    if args.mode == "text":
        results = retriever.retrieve_text(args.query)

    elif args.mode == "hybrid":
        results = retriever.retrieve_hybrid(args.query)

    elif args.mode == "text_to_image":
        results = retriever.retrieve_text_to_image(args.query)

    else:
        raise ValueError("Invalid mode")

    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------
    context = retriever.build_context(results)

    # -----------------------------
    # GENERATE ANSWER
    # -----------------------------
    prompt = build_prompt(args.query, context)
    answer = generator.generate(prompt)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n=== CONTEXT ===\n")
    print(context[:1000])  # truncate for sanity

    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()