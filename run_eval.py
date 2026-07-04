"""
run_eval.py

This is the "evaluation checkpoint" the assignment asks for: run 3 sample
questions through the full pipeline and print the retrieved chunks plus
the final answer for each one.

Run it with:
    python run_eval.py

Note on dependencies: 
- The embeddings now use ChromaDB's built-in ONNX runtime, which downloads 
  a tiny model (~80MB) on the first run. 
- The summary generation uses the Hugging Face Inference API, so no large 
  language models are downloaded to your disk. 
Make sure you have an internet connection for the first run and that your 
HUGGINGFACEHUB_API_TOKEN is set in your .env file.
"""

from app.orchestrator import run_pipeline
from app.rag_pipeline import build_or_load_vector_store

SAMPLE_QUESTIONS = [
    "What are the top complaints this month and how do they compare to last month?",
    "What is our current CSAT and does it meet the company target?",
    "How do we handle a customer complaint about long wait times?",
]


def main():
    vector_store = build_or_load_vector_store("data/faq.txt")

    for question in SAMPLE_QUESTIONS:
        print("=" * 80)
        print("QUESTION:", question)
        result = run_pipeline(
            question=question,
            survey_data_path="data/survey_data.json",
            vector_store=vector_store,
        )

        if result.rag_result:
            print("\nRETRIEVED FAQ CHUNKS:")
            for chunk in result.rag_result.chunks:
                print("-", chunk.replace("\n", " ")[:150])

        if result.data_result:
            print("\nDATA METRICS:", result.data_result.model_dump())

        if result.comparison_result:
            print("\nCOMPARISON:", result.comparison_result.notable_changes)

        print("\nFINAL ANSWER:")
        print(result.answer)
        print()


if __name__ == "__main__":
    main()