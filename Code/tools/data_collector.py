import pandas as pd
import similarity_toolkit_LLM as stk
import pyreadstat as prs
import os

RELEVANT_QUESTIONS = {
    "age": ["How old are you?", "What is your age?", "When were you born?", "Age?"],
    "gender": ["What is your gender?", "Are you male or female?", "Gender?"],
    "education": [
        "What is your highest level of education?",
        "What is your educational attainment?"
    ],
    "household_size": [
        "How many people live in your household?",
        "What is the size of your household?"
    ],
    "income_household_per_person": [
        "What is your household income per person?",
        "What is the income per person in your household?",
        "What is on average the income per household member?",
        "What is the monthly income per person in your household?"
    ],
    "income_personal": [
        "What is your personal income?",
        "What is your individual income?",
        "What is your monthly income?"
    ],
    "employment_status": [
        "What is your current employment status?",
        "Are you employed, unemployed, or retired?",
        "What is your job status?"
    ]
}

SPSS_PATH = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS SPSS"

def find_relevant_questions(spss_path: str, relevant_questions: dict) -> dict:
    toolkit = stk.SimilarityToolkit()
    results = {key: [] for key in relevant_questions.keys()}

    list_dir = [file for file in os.listdir(spss_path) if file.endswith(".sav")]

    for file in list_dir:
        df, meta = prs.read_sav(os.path.join(spss_path, file))
        column_labels = meta.column_labels

        for question_key, question_variants in relevant_questions.items():
            for column, label in column_labels.items():
                for variant in question_variants:
                    similarity_score = toolkit.similarity(label, variant)
                    if similarity_score > 0.8:  # Threshold for relevance
                        results[question_key].append((file, column, label, similarity_score))

    return results
