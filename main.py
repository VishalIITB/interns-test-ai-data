import argparse
from csv_reader import read_subject_csv
# from llm_api import call_anthropic  # Uncomment if using Anthropic in your solution

import os
import csv
# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_concepts_for_subject(subject):
    if subject == "ancient_history":
        concept_dict = {
            "harappa": ["Harappan Civilization"],
            "mohenjo": ["Harappan Civilization"],
            "ashoka": ["Ashokan Edicts", "Mauryan Empire"],
            "dhamma": ["Ashokan Edicts"],
            "gupta": ["Gupta Empire", "Scientific Contributions"],
            "tank": ["Irrigation", "Water Management"],
            "taniyurs": ["Brahmadeya Grants"],
            "ghatikas": ["Temple-based Education"],
            "arthashastra": ["Kautilya's Arthashastra"],
            "vedas": ["Vedic Literature"],
            "buddha": ["Buddhism"],
            "buddhism": ["Buddhism"],
            "vedic": ["Vedic Civilization"],
            "sine": ["History of Indian Science"],
            "quadrilateral": ["Geometry"],
        }

        concept_descriptions = {
            "Harappan Civilization": "Ancient civilization with advanced urban planning, trade, drainage.",
            "Ashokan Edicts": "Inscriptions by Ashoka about governance and dhamma.",
            "Mauryan Empire": "Empire known for Ashoka and administration by Kautilya.",
            "Gupta Empire": "Known for contributions to science, art, literature, mathematics.",
            "Scientific Contributions": "Includes zero, pi, sine, astronomy, and surgery.",
            "Temple-based Education": "Education in temples using ghatikas and Brahmin scholars.",
            "Brahmadeya Grants": "Land grants to Brahmins during Chola period.",
            "Vedic Literature": "Text from Rigveda, Yajurveda, and early religious traditions.",
            "Buddhism": "Teachings of Buddha, emergence of Buddhist philosophy.",
            "Vedic Civilization": "Early society in north India with ritualistic practices.",
            "History of Indian Science": "Includes advances in astronomy, math, metallurgy.",
            "Geometry": "Understanding of shapes like triangles, quadrilaterals.",
            "Irrigation": "Tank-based systems for farming and village water supply.",
            "Water Management": "Ancient practices of storing, distributing water.",
            "Kautilya's Arthashastra": "Text on Mauryan economy, diplomacy and governance.",
        }

    elif subject == "math":
        concept_dict = {
            "calculus": ["Calculus"],
            "derivative": ["Calculus"],
            "integral": ["Calculus"],
            "probability": ["Probability"],
            "mean": ["Statistics"],
            "median": ["Statistics"],
            "mode": ["Statistics"],
            "matrix": ["Linear Algebra"],
            "determinant": ["Linear Algebra"],
            "vector": ["Vectors"],
            "complex": ["Complex Numbers"],
            "logarithm": ["Logarithms"],
            "circle": ["Geometry"],
            "quadrilateral": ["Geometry"],
            "trigonometry": ["Trigonometry"],
            "sine": ["Trigonometry"],
            "cosine": ["Trigonometry"]
        }

        concept_descriptions = {
            "Calculus": "Mathematical study of change using derivatives and integrals.",
            "Probability": "Quantification of uncertainty and prediction of outcomes.",
            "Statistics": "Analyzing data using measures like mean, median, and mode.",
            "Linear Algebra": "Study of matrices, determinants, and linear transformations.",
            "Vectors": "Quantities with both magnitude and direction.",
            "Complex Numbers": "Numbers in the form a + bi, used in various equations.",
            "Logarithms": "Inverse of exponentiation, simplifies calculations.",
            "Geometry": "Study of shapes, sizes, and properties of space.",
            "Trigonometry": "Relations in triangles involving sine, cosine, and tangent."
        }

    elif subject == "physics":
        concept_dict = {
            "velocity": ["Kinematics"],
            "acceleration": ["Kinematics"],
            "motion": ["Kinematics"],
            "force": ["Newton's Laws"],
            "newton": ["Newton's Laws"],
            "gravity": ["Gravitation"],
            "work": ["Work and Energy"],
            "energy": ["Work and Energy"],
            "power": ["Work and Energy"],
            "wave": ["Waves and Oscillations"],
            "frequency": ["Waves and Oscillations"],
            "voltage": ["Electricity"],
            "current": ["Electricity"],
            "resistance": ["Electricity"],
            "light": ["Optics"],
            "lens": ["Optics"],
            "mirror": ["Optics"]
        }

        concept_descriptions = {
            "Kinematics": "Study of motion without considering the forces.",
            "Newton's Laws": "Laws governing force, mass, and acceleration.",
            "Gravitation": "Force of attraction between masses.",
            "Work and Energy": "Concepts of work, energy transformation, and power.",
            "Waves and Oscillations": "Motion patterns like sound, light, and spring systems.",
            "Electricity": "Concepts of electric current, voltage, and resistance.",
            "Optics": "Study of light, reflection, refraction, and optical instruments."
        }

    elif subject == "economics":
        concept_dict = {
            "inflation": ["Inflation"],
            "cpi": ["Inflation"],
            "gdp": ["Macroeconomics"],
            "growth rate": ["Macroeconomics"],
            "demand": ["Demand and Supply"],
            "supply": ["Demand and Supply"],
            "market": ["Market Structures"],
            "monopoly": ["Market Structures"],
            "tax": ["Public Finance"],
            "budget": ["Public Finance"],
            "interest": ["Banking"],
            "bank": ["Banking"],
            "monetary": ["Monetary Policy"],
            "fiscal": ["Fiscal Policy"]
        }

        concept_descriptions = {
            "Inflation": "Rise in general level of prices of goods and services.",
            "Macroeconomics": "Study of the economy as a whole, including GDP and growth.",
            "Demand and Supply": "Basic economic model of price and quantity.",
            "Market Structures": "Types of markets like monopoly, oligopoly, perfect competition.",
            "Public Finance": "Government spending, taxation, and budgeting.",
            "Banking": "Financial institutions dealing with money, loans, and interest.",
            "Monetary Policy": "Central bank policies controlling money supply and interest rates.",
            "Fiscal Policy": "Government revenue and spending to influence economy."
        }

    else:
        raise ValueError("Invalid subject name.")

    return concept_dict, concept_descriptions


# --- Concept Extraction Pipeline ---

def extract_named_entities(text):
    doc = nlp(text)
    return [ent.text.lower() for ent in doc.ents]

def get_tfidf_model(concept_descriptions):
    concept_names = list(concept_descriptions.keys())
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(concept_descriptions.values())
    return tfidf_vectorizer, tfidf_matrix, concept_names

def tfidf_match(question, tfidf_vectorizer, tfidf_matrix, concept_names):
    question_vec = tfidf_vectorizer.transform([question])
    similarity = cosine_similarity(question_vec, tfidf_matrix)
    best_idx = similarity.argmax()
    return [concept_names[best_idx]]

def extract_concepts(question, concept_dict, tfidf_vectorizer, tfidf_matrix, concept_names):
    question_lower = question.lower()
    tokens = question_lower.split()
    concepts = set()

    # Keyword Matching
    for word in tokens:
        if word in concept_dict:
            concepts.update(concept_dict[word])

    # Named Entity Matching
    named_ents = extract_named_entities(question)
    for ne in named_ents:
        if ne in concept_dict:
            concepts.update(concept_dict[ne])

    # TF-IDF Fallback
    if not concepts:
        concepts.update(tfidf_match(question, tfidf_vectorizer, tfidf_matrix, concept_names))

    return list(concepts)

def write_output(file_path, result):
    os.makedirs("outputs", exist_ok=True)
    with open(file_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["question_number", "question", "concepts"])
        for item in result:
            writer.writerow([
                item["question_number"],
                item["question"],
                "; ".join(item["concepts"])
            ])

def main():
    parser = argparse.ArgumentParser(description="Intern Test Task: Question to Concept Mapping")
    parser.add_argument('--subject', required=True, choices=['ancient_history', 'math', 'physics', 'economics'], help='Subject to process')
    args = parser.parse_args()

    concept_dict, concept_descriptions = get_concepts_for_subject(args.subject)

    tfidf_vectorizer, tfidf_matrix, concept_names = get_tfidf_model(concept_descriptions)


    data = read_subject_csv(args.subject)
    print(f"Loaded {len(data)} questions for subject: {args.subject}")

    # --- PLACEHOLDER FOR USER CODE ---
    # TODO: Implement your question-to-concept mapping logic here.
    # For example, iterate over data and map questions to concepts.
    # You can use the call_anthropic function from llm_api.py if needed.
    # Example:
    # for row in data:
    #     question = row['question']
    #     # concept = call_anthropic(f"Map this question to a concept: {question}")
    #     # print({"question": question, "concept": concept})
    # ----------------------------------

    results = []
    for i, row in enumerate(data):
        q_number = row.get("Question Number", f"{i+1}")
        question = row["Question"]
        concepts = extract_concepts(question, concept_dict, tfidf_vectorizer, tfidf_matrix, concept_names)

        results.append({
            "question_number": q_number,
            "question": question,
            "concepts": concepts
        })

        print(f"Q{q_number}: {question}")
        print(f"  → Concepts: {', '.join(concepts)}")

    output_path = f"outputs/output_concepts_{args.subject}.csv"
    write_output(output_path, results)
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
