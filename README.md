#  Concept Extraction from Competitive Exam Questions

**Author:** Vishal Kumar  
**Roll No.:** 22b0687  
**Submission for:**  AI/ML Research & Data Engineer  
**GitHub Repository:** [https://github.com/VishalIITB/interns-test-ai-data](https://github.com/VishalIITB/interns-test-ai-data)

---

##  Objective

The objective of this assignment is to simulate an **LLM-driven concept extraction pipeline** that can analyze competitive exam questions (e.g., UPSC, NEET, JEE) and identify the **underlying conceptual tags** they test.

---

##  How It Works

The pipeline uses a hybrid approach combining:
- **Keyword Matching**
- **Named Entity Recognition (NER)**
- **TF-IDF Vector Similarity**
- **(Optional) LLM Simulation for fallback**

Each question is mapped to one or more conceptual tags based on subject-specific dictionaries.

---

## Folder Structure

```
.
├── main.py                 # Entry point, handles CLI and user code
├── llm_api.py              # Handles Anthropic API calls, loads API key from .env
├── csv_reader.py           # Reads CSV from resources/ and returns data
├── resources/              # Folder containing subject CSVs (ancient_history.csv, math.csv, etc.)
├── outputs                 # Stores results
├── requirements.txt        # Python dependencies
├── Makefile                # Run commands
└── README.md               # Instructions
```


# LLM Prompt Format Used
The 5-shot prompting is in .txt file on Qwen model.

I used 5-shot prompting to get the answer as shown below-

# Example-

{
"Question Number": 6,

"Question": "Consider the following pairs: Historical place Well - known for 1. Burzahom : Rock-cut shrines 2. Chandra-ketugarh : Terracotta art 3. Ganeshwar : Copper artefacts Which of the pairs given above is/are correctly matched?",

"Option A": "1 only",

"Option B": "1 and 2",

"Option C": "3 only",

"Option D": "2 and 3",

"Ans": "D",

"Concept": "Indian Archaeology"
}

