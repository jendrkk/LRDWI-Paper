import pandas as pd
import similarity_toolkit_LLM as stk
import pyreadstat as prs
import numpy as np
import os

RELEVANT_QUESTIONS = {
    "age": ["W którym roku się Pan(i) urodził(a)?", "Rok urodzenia respondenta"],
    "gender": ["płeć", "Płeć respondenta"],
    "location": ["Województwo", "WOJEWÓDZTWO"],
    "city_size": [
        "Czy miejscowość, w której Pan(i) mieszka na stałe jest:"
    ],
    "education": [
        "Jakie ma Pan(i) wykształcenie? Proszę podać najwyższy osiągnięty przez Pana(ią) poziom wykształcenia.", "Jakie ma Pan(i) wykształcenie?"
    ],
    "if_retired": [
        "Czy otrzymuje Pan(i) rentę lub emeryturę, niezależnie od tego, czy jest ona Pana(i) głównym źródłem utrzymaniaczy też nie?",
    ],
    "if_party_member": [
        "Czy w przeszłości należał(a) Pan(i) do PZPR?"
    ],
    "marital": [
        "Jaki jest Pana(i) stan cywilny?"  
    ],
    "household_size": [
        "Z ilu osób, łącznie z Panem(ią), składa się Pana(i) gospodarstwo domowe?"
    ],
    "income_household_per_person": [
        "Ile wynoszą PRZECIĘTNE MIESIĘCZNE DOCHODY NETTO (NA RĘKĘ) PRZYPADAJĄCE NA JEDNĄ OSOBĘ W PANA(I) GOSPODARSTWIE DOMOWYM? Proszę wziąć pod uwagę dochody z pracy głównej wraz z premiami, nagrodami, dodatkami, dochody z prac dodatkowych, także dorywczy",
        "Kwota przeciętnych, miesięcznych dochodów NETTO na jedną osobę w gospodarstwie domowym respondenta - średnia z ostatnich 3 ms",
        "A czy mógłby(mogłaby) Pan(i) podać przybliżone dochody? Proszę spojrzeć na ekran. Czy mógłby(mogłaby) Pan(i) wskazać, w którym przedziale mieszczą się PRZECIĘTNE MIESIĘCZNE DOCHODY NETTO (NA RĘKĘ) PRZYPADAJĄCE NA JEDNĄ OSOBĘ W PANA(I) GOSPODARST",
        "Ile wynoszą W STARYCH ZŁOTYCH miesięczne dochody na 1 osobę w Pana(i) gospodarstwie domowym? [tysiące starych złotych]",
        "Ile wynoszą W NOWYCH ZŁOTYCH miesięczne dochody na 1 osobę w Pana(i) gospodar- stwie domowym? Proszę wziąć pod uwagę dochody z pracy głównej wraz z premiami, nagrodami, dochody z prac dodatkowych."
    ],
    "income_personal": [
        "Ile wynoszą PANA(I) MIESIĘCZNE DOCHODY NETTO (NA RĘKĘ)? Proszę wziąć pod uwagę Pana(i) dochody z pracy głównej wraz z premiami, nagrodami, dodatkami, dochody z prac dodatkowych, także dorywczych, renty i emerytury, stypendia oraz inne dodatkowe",
        "Kwota przeciętnych, miesięcznych dochodów NETTO respondenta",
        "A czy mógłby(mogłaby) Pan(i) podać przybliżone dochody? Proszę spojrzeć na ekran. Czy mógłby(mogłaby) Pan(i) wskazać, w którym przedziale mieszczą się PANA(I) MIESIĘCZNE DOCHODY NETTO (NA RĘKĘ)?"
    ],
    "employment_status": [
        "What is your current employment status?",
        "Are you employed, unemployed, or retired?",
        "What is your job status?"
    ],
    "savings": [
        "Czy Pana(i) gospodarstwo domowe posiada oszczędności pieniężne?"
    ],
    "debt": [
        "Czy Pana(i) gospodarstwo domowe ma obecnie do spłacenia jakieś raty, pożyczki, długi lub kredyty?"
    ],
    "wealth": [
        "Gdyby sprzedał(a) Pan(i) wszystko, co posiada Pana(i) gospodarstwo domowe to mniej więcej, ile pieniędzy mógł(a)by Pan(i) w ten sposób uzyskać?",
    ],
    "house": [
        "Proszę powiedzieć, które z wymienionych dóbr znajdowały się Pana(i) gospodarstwie domowym przed 1990 rokiem, są obecnie w Pan(i) gospodarstwie domowym, a które spośród nich chciał(a)by Pan(i) mieć: - Dom mieszkalny"
    ],
    "flat": [
        "Proszę powiedzieć, które z wymienionych dóbr znajdowały się Pana(i) gospodarstwie domowym przed 1990 rokiem, są obecnie w Pan(i) gospodarstwie domowym, a które spośród nich chciał(a)by Pan(i) mieć: - Mieszkanie własnościowe"
    ],
    "immobility": [
        "Czy Pan(i) lub ktoś z Pana(i) gospodarstwa domowego w ciągu ostatnich 12 miesięcy (w 1995 roku): - budował, kupił dom/mieszkanie"
    ],
    "car": [
        "Czy posiada/posiadają Pan(i)/Państwo w swym gospodarstwie domowym samochód osobowy/samochody osobowe?",
        "Czy Pan(i) lub ktoś z Pana(i) gospodarstwa domowego w ciągu ostatnich 12 miesięcy (w 1995 roku): - kupił samochód"
    ],
    "if_investing": [
      "Czy posiada Pan(i): - Akcje firm notowanych na giełdzie", "Czy posiada Pan(i): - Obligacje skarbu państwa", "Czy posiada Pan(i): - Jednostki uczestnictwa w funduszu powierniczym"
    ],
    "weight": [
        "WAGA"
    ]
}

TRIGGER_WORDS = [
    "mieszkanie", "dom", "samochód", "oszczędności", "kredyt", "dług", "dochody", "inwestuje", "akcje"
]

SPSS_PATH = "/Users/jedrek/Documents/Studium Volkswirschaftslehre/3. Semester/Long-run dynamics of wealth inequalities/Paper/Data/CBOS SPSS"

def remove_code_from_label(label: str) -> str:
    # Remove any leading codes from the label (e.g., "Q1. ", "M23. ", etc.).
    # A code is a string that starts with a letter(s) followed by digits and a dot.
    import re
    cleaned_label = re.sub(r'^[A-Za-z]+\d*\.\s*', '', label)
    return cleaned_label.strip()

def v_split(xs, sep="/"):
    if isinstance(xs, str):
        xs = [xs]
    res = []
    for x in xs:
        if isinstance(x, str):
            res.extend(x.split(sep))
    return res


def check_trigger_words(label: str) -> bool:
    label_lower = label.lower()
    label_lower = v_split(label_lower.split(), "/")
    label_lower = v_split(label_lower, "(")
    label_lower = v_split(label_lower, ")")
    
    for word in TRIGGER_WORDS:
        if word in label_lower:
            return True
    return False

def find_relevant_questions(st: stk.SimilarityToolkit, spss_file_path: str, threshold: float = 0.95) -> pd.DataFrame:
    # Load SPSS file
    df, meta = prs.read_sav(spss_file_path)
    
    results = []
    ambigous_labels = []
    
    i = 0
    for column in df.columns:
        column_label = meta.column_labels[i]
        i += 1
        
        if column_label is None:
            continue
        
        column_label = remove_code_from_label(column_label)
        
        label_added = False
        
        for key, questions in RELEVANT_QUESTIONS.items():
            scores = []
            if column_label in questions:
                results.append({
                    "column_name": column,
                    "column_label": column_label,
                    "matched_question": questions[np.where(np.array(questions) == column_label)[0][0]],
                    "similarity_score": 1,
                    "relevant_key": key
                })
                label_added = True
                print(f"Exactly matched column '{column}' with label '{column_label}' to question '{questions[np.where(np.array(questions) == column_label)[0][0]]}'")
                print("\n")
                continue
            
            for question in questions:
                #print(f"Comparing column label: '{column_label}' with question: '{question}'")
                score = st.similarity(column_label, question)
                scores.append(score['score'])
            
            #print(f"Scores for column '{column}' and key '{key}': {scores}")
            max_score = max(scores)
            if max_score >= threshold:
                results.append({
                    "column_name": column,
                    "column_label": column_label,
                    "matched_question": questions[scores.index(max_score)],
                    "similarity_score": max_score,
                    "relevant_key": key
                })
                label_added = True
                print(f"Matched column '{column}' with label '{column_label}'.")
                print(f"to question '{questions[scores.index(max_score)]}' (score: {max_score})")
                print("\n")

        if check_trigger_words(column_label) and not label_added:
            ambigous_labels.append({
                "column_name": column,
                "column_label": column_label
            })
            print(f"Ambiguous column '{column}' with label '{column_label}' (max score: {max_score})")
            print("\n")

    return pd.DataFrame(results), pd.DataFrame(ambigous_labels)

def main():
    spss_files = [file for file in os.listdir(SPSS_PATH) if file.endswith(".sav")]
    
    all_results = []
    all_ambiguous = []
    
    # Initialize similarity toolkit
    st = stk.SimilarityToolkit()
    
    for file in spss_files:
        file_path = os.path.join(SPSS_PATH, file)
        print("-"*50)
        print(f"Processing file: {file}")
        result_df, ambiguous_df = find_relevant_questions(st, file_path, threshold=0.925)
        result_df["spss_file"] = file
        ambiguous_df["spss_file"] = file
        all_results.append(result_df)
        all_ambiguous.append(ambiguous_df)
        print("="*50)
        
    final_df = pd.concat(all_results, ignore_index=True)
    ambiguous_final_df = pd.concat(all_ambiguous, ignore_index=True)
    final_df.to_csv("relevant_questions_summary.csv", index=False)
    ambiguous_final_df.to_csv("ambiguous_questions_summary.csv", index=False)
    print("Summary saved to relevant_questions_summary.csv")

if __name__ == "__main__":
    main()