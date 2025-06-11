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
