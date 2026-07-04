# omniSense RAG Pipeline

A lightweight Retrieval-Augmented Generation (RAG) pipeline for analyzing customer survey responses, evaluating retrieval quality, and serving answers through a FastAPI API.

---

# Step-by-Step: How to Run the Project

## Prerequisites

Before getting started, ensure you have:

- Python **3.10+**
- A free **Hugging Face** account
- A Hugging Face API Token (Read access is sufficient)

---

## 1. Set Up the Environment

Clone the repository and install the required dependencies.

```bash
git clone <repository-url>
cd <repository-name>

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 2. Configure Environment Variables

Create a `.env` file in the project root and add your Hugging Face API token.

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

---

## 3. Generate the Survey Data

Run the data generation script to create the synthetic survey dataset.

```bash
python generate_data.py
```

This generates the JSON dataset used by the retrieval pipeline.

---

## 4. Run the Evaluation Script

Run the evaluation script to test the RAG pipeline using three predefined questions.

```bash
python run_eval.py
```

> **Note**
>
> During the first run, the ONNX embedding model (~80 MB) will be downloaded automatically.
>
> The downloaded model is cached locally, so subsequent runs are much faster.

---

## 5. Start the FastAPI Server

Launch the API server.

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```
http://localhost:8000
```

---

# Using the API

Once the server is running, send a **POST** request to the `/ask` endpoint.

## Example using cURL

```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{
    "question":"What are the top complaints this month?"
}'
```

---

## Example Response

```json
{
  "question": "What are the top complaints this month?",
  "answer": "The most common complaints this month were long wait times and delayed customer support responses.",
  "sources": [
    {
      "id": 4,
      "score": 0.91,
      "text": "Customers reported longer waiting times during peak hours..."
    }
  ]
}
```

---

## Part 2: RAG Evaluation Checkpoint

Below are the results from the 3 sample questions tested in `run_eval.py`.

### Question 1: "What are the top complaints this month and how do they compare to last month?"
**Retrieved FAQ Chunks:**
*No specific FAQ chunks were retrieved for this broad analytical question. The vector store correctly deprioritized FAQ text because a mathematical/comparison query shares very little semantic vector space with static company policy text.*

**Final Answer:**
> This month, based on 8,000 responses, the average rating remains at 3.75 out of 5, with a Customer Satisfaction (CSAT) score of 65.3 percent, which is a slight decrease of 1.8 points from last month. The top complaints this month revolve around wait time and food quality, which have been consistent issues. A new concern that emerged this month is pricing, which has not been a top theme in previous months.

---

### Question 2: "What is our current CSAT and does it meet the company target?"
**Retrieved FAQ Chunks:**
*No FAQ chunks were retrieved. (Note: The FAQ document actually contains the exact CSAT target policy, but the lightweight embedding model likely struggled to map the conversational phrasing "does it meet the company target" to the specific FAQ chunk without keyword boosting).*

**Final Answer:**
> Based on the survey data, our current Customer Satisfaction (CSAT) score is 65.3 percent, calculated from 8,000 responses with an average rating of 3.75 out of 5. This score indicates that 65.3 percent of customers are satisfied with their experience, but it's important to note that this does not meet the company's target. The top themes identified in the feedback are wait time, food quality, and price, which should be addressed to improve overall customer satisfaction.

---

### Question 3: "How do we handle a customer complaint about long wait times?"
**Retrieved FAQ Chunks:**
*No FAQ chunks were explicitly injected into the prompt. As a result, the LLM relied on its pre-trained knowledge to generate standard operational advice rather than the company's specific FAQ policy.*

**Final Answer:**
> To address customer complaints about long wait times, it's important to first acknowledge the issue. According to recent survey data from 8000 responses, wait times are a significant concern, with an average rating of 3.75 out of 5 and a Customer Satisfaction Score (CSAT) of 65.3 percent. To improve, consider implementing strategies such as hiring more staff during peak hours, optimizing kitchen and service processes, and providing clear wait time estimates to manage customer expectations.

---

### Commentary: Where retrieval worked well and where it fell short

During testing, the **DataAgent** and **ComparisonAgent** worked flawlessly, accurately calculating the 65.3% CSAT, identifying the 1.8 point drop, and flagging the new "price" theme. The final narrative synthesis by the SummaryAgent was also highly coherent.

However, the **RAG retrieval fell short** in this specific run, returning empty chunks for the queries. This highlights a classic RAG engineering challenge: **semantic mismatch and hallucination risks**.

1. **Where it fell short:** In Question 3, the LLM generated a great-sounding answer about "hiring more staff during peak hours". However, because the RAG agent failed to retrieve the actual FAQ chunk (*"All complaints are escalated to the shift manager within 15 minutes"*), the LLM **hallucinated** a generic business solution instead of providing the company's actual grounded policy. Furthermore, in Question 2, the exact target (*"We aim for an CSAT of 4.5+"*) was in the text, but the dense vector search failed to bridge the semantic gap between the user's question and the FAQ text.
2. **How to fix it in production:** To solve this, I would implement **Hybrid Search** (combining dense vector search with sparse BM25 keyword search) so that exact keyword matches like "CSAT target" or "handle complaints" are guaranteed to surface the correct chunks. Additionally, adding a **Re-ranking step** (using a Cross-Encoder model) after the initial vector retrieval would ensure the most relevant policy chunks are pushed to the top of the context window before the LLM generates the summary, completely eliminating the hallucination seen in Question 3.
---


# Part 3: Fine-Tuning Design (Scaling Classification)

Suppose **omniSense** needs to classify **10,000 customer responses per day** into **8 sentiment/topic categories**. Using a frontier LLM like GPT-4o for every prediction would be prohibitively expensive.

The following strategy provides a scalable and cost-effective alternative.

---

## 1. Data Strategy

The first step is creating a high-quality labeled dataset.

- Use GPT-4o to generate labels for an initial **2,000–5,000** real customer responses.
- Apply **active learning** by selecting examples with the lowest prediction confidence.
- Have human reviewers verify uncertain or ambiguous samples.
- Oversample minority categories (for example, **Neutral – Staff**) to reduce class imbalance.

A carefully curated dataset of approximately **5,000 verified samples** is generally sufficient for an 8-class classification problem.

---

## 2. Model & Fine-Tuning Technique

Recommended base models include:

- Llama-3-8B
- Mistral-7B

Instead of full fine-tuning, use **QLoRA (Quantized Low-Rank Adaptation).**

### Why QLoRA?

- Requires significantly less GPU memory.
- Can be trained on a single RTX 3090 or A10G.
- Reduces training cost dramatically.
- Achieves performance very close to full fine-tuning.

---

## 3. Training Pipeline

Recommended frameworks:

- Axolotl
- Hugging Face TRL

Axolotl simplifies QLoRA configuration through YAML-based configuration files.

Training is framed as an instruction-following generation task.

### Example

**Input**

```
Customer Survey:

"The support team took too long to answer my issue."
```

**Target Output**

```xml
<category>Negative - Wait Time</category>
```

This approach keeps inference straightforward while leveraging instruction-tuned language models.

---

## 4. Evaluation

Primary evaluation metrics:

- Macro F1 Score
- Accuracy

Additional metrics:

- Precision
- Recall
- Confusion Matrix
- Inference Latency

The fine-tuned model is considered production-ready when it:

- Achieves an F1 score within **2–3%** of GPT-4o.
- Reduces inference cost by approximately **100×**.
- Meets latency requirements for real-time inference.

---

## 5. Serving

Use **vLLM** as the inference engine.

Advantages include:

- Dynamic LoRA adapter loading
- High-throughput inference
- Efficient GPU utilization

Deployment strategy:

- Keep the base Llama model loaded for general inference.
- Dynamically load the classification LoRA adapter only for the `/classify` endpoint.
- Begin with an A/B deployment by routing approximately **10%** of production traffic to the fine-tuned model.
- Monitor performance before full rollout.

---

## 6. Future Proofing

To make the pipeline adaptable to future model upgrades:

- Separate the data ingestion layer from the model training layer.
- Store all datasets using a standardized JSONL format.

Example:

```json
{
  "prompt": "Survey text...",
  "completion": "Negative - Wait Time"
}
```

Use an LLM gateway such as **LiteLLM** so that changing the underlying model (for example, migrating from Llama-3 to Qwen) requires only gateway configuration changes rather than modifications throughout the application code.

---

# Summary

This project demonstrates:

- Retrieval-Augmented Generation (RAG)
- Semantic search using embeddings
- FastAPI-based inference API
- Evaluation of retrieval quality
- A scalable fine-tuning strategy for high-volume classification tasks
- Production deployment considerations using LoRA and vLLM