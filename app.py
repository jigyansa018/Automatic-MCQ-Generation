# pip install flask spacy PyPDF2
# python -m spacy download en_core_web_sm

from flask import Flask, render_template, request
import spacy
import random
import string
import sqlite3
import PyPDF2
import io
from datetime import datetime

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

DATABASE = "quiz_history.db"

# ── DATABASE ───────────────────────────────────────────────────────────────────
def init_db():
    with sqlite3.connect(DATABASE) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                num_q      INTEGER,
                difficulty TEXT,
                engine     TEXT
            )
        """)

def save_session(num_q, difficulty, engine="spacy"):
    with sqlite3.connect(DATABASE) as conn:
        conn.execute(
            "INSERT INTO history (created_at, num_q, difficulty, engine) VALUES (?,?,?,?)",
            (datetime.utcnow().isoformat(), num_q, difficulty, engine)
        )

def get_history():
    with sqlite3.connect(DATABASE) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM history ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


# ── STOPWORDS ──────────────────────────────────────────────────────────────────
STOPWORDS = set([
    "the","a","an","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","dare","ought","used","to","of","in","on","at","by",
    "for","with","about","against","between","into","through","during","before",
    "after","above","below","from","up","down","out","off","over","under",
    "then","once","here","there","when","where","why","how","all","both","each",
    "few","more","most","other","some","such","no","nor","not","only","own",
    "same","so","than","too","very","just","because","as","until","while",
    "although","though","since","unless","if","or","and","but","yet","either",
    "neither","also","however","therefore","thus","hence","that","this","these",
    "those","it","its","itself","they","them","their","theirs","themselves",
    "he","him","his","himself","she","her","hers","herself","we","us","our",
    "ours","ourselves","you","your","yours","yourself","i","me","my","mine",
    "myself","who","whom","whose","which","what","one","two","three","first",
    "second","third","new","old","many","much","any","every","another","certain",
    "several","already","often","always","never","sometimes","usually","generally",
    "mainly","mostly","particularly","especially","including","example","like",
    "well","even","still","back","way","according","regarding","various",
    "different","important","said","say","says","per","etc","ie","eg",
])

EXCLUDED_POS = {"ADP","VERB","AUX","CONJ","CCONJ","SCONJ",
                "DET","PUNCT","SPACE","PART","INTJ"}


# ── TEXT CLEANING ──────────────────────────────────────────────────────────────
def clean_text(text):
    cleaned = ""
    for ch in text:
        if ch in string.punctuation and ch not in ".,'":
            cleaned += " "
        else:
            cleaned += ch
    return " ".join(cleaned.split())


# ── FILE EXTRACTION ────────────────────────────────────────────────────────────
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + " "
        return clean_text(text)
    elif filename.endswith(".txt"):
        return clean_text(file.read().decode("utf-8", errors="ignore"))
    return ""


# ── GOOD SENTENCES ─────────────────────────────────────────────────────────────
def get_good_sentences(doc):
    good = []
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if len(tokens) < 8:
            continue
        has_verb = any(t.pos_ == "VERB" for t in tokens)
        has_noun = any(t.pos_ in ("NOUN", "PROPN") for t in tokens)
        if has_verb and has_noun:
            good.append(sent)
    return good


# ── ANSWER CANDIDATES ──────────────────────────────────────────────────────────
def get_answer_candidates(span):
    candidates = []
    seen = set()

    for ent in span.ents:
        word = ent.text.strip()
        if len(word) > 1 and word.lower() not in STOPWORDS and word.lower() not in seen:
            seen.add(word.lower())
            candidates.append((word, ent.label_))

    for chunk in span.noun_chunks:
        word = chunk.text.strip()
        if len(word) > 1 and word.lower() not in STOPWORDS and word.lower() not in seen:
            seen.add(word.lower())
            candidates.append((word, "NOUN_CHUNK"))

    for token in span:
        if token.pos_ not in EXCLUDED_POS:
            word = token.text.strip()
            if (len(word) > 1 and word.lower() not in STOPWORDS
                    and not token.is_punct and not token.is_space
                    and word.lower() not in seen):
                seen.add(word.lower())
                candidates.append((word, token.pos_))

    return candidates


# ── WH-QUESTION ────────────────────────────────────────────────────────────────
WH_MAP = {
    "PERSON":"Who","ORG":"Which organization","GPE":"Where","LOC":"Where",
    "DATE":"When","TIME":"When","MONEY":"How much","CARDINAL":"How many",
    "ORDINAL":"Which","PERCENT":"What percentage","PRODUCT":"Which product",
    "EVENT":"What event","WORK_OF_ART":"What","LAW":"Which law",
    "LANGUAGE":"Which language","QUANTITY":"How much",
}
PROCESS_WORDS = {"process","cause","affect","work","function","method",
                 "technique","approach","mechanism","system","way","means","manner"}

def make_wh_question(sentence_text, answer, label):
    if label in WH_MAP:
        q_word = WH_MAP[label]
    elif label == "ADJ":
        q_word = "How"
    elif label in ("CARDINAL","NUM"):
        q_word = "How many"
    elif any(pw in answer.lower() for pw in PROCESS_WORDS):
        q_word = "How"
    else:
        q_word = "What"
    q_sentence = sentence_text.replace(answer, "________", 1)
    return f"{q_word}: {q_sentence.strip()}", "WH"


# ── FILL-IN-THE-BLANK ─────────────────────────────────────────────────────────
def make_fill_question(sentence_text, answer):
    q_sentence = sentence_text.replace(answer, "________", 1)
    return f"Fill in the blank: {q_sentence.strip()}", "FILL"


# ── DISTRACTORS ────────────────────────────────────────────────────────────────
def get_distractors(correct, all_candidates, difficulty, num=3):
    pool = list({c for c, _ in all_candidates if c.lower() != correct.lower()})
    if difficulty == "easy":
        pool.sort(key=len)
    elif difficulty == "hard":
        pool.sort(key=len, reverse=True)
    else:
        random.shuffle(pool)
    distractors = pool[:num]
    generic = ["None of the above","All of the above",
               "Cannot be determined","Not mentioned in the text"]
    i = 0
    while len(distractors) < num:
        g = generic[i % len(generic)]
        if g not in distractors:
            distractors.append(g)
        i += 1
    return distractors[:num]


# ── MCQ GENERATION ────────────────────────────────────────────────────────────
def generate_mcqs(text, num_questions=10, difficulty="medium"):
    doc = nlp(text)
    sentences = get_good_sentences(doc)
    if not sentences:
        return []

    full_doc_candidates = get_answer_candidates(doc)
    mcqs = []
    used_answers = set()
    random.shuffle(sentences)

    for sent in sentences:
        if len(mcqs) >= num_questions:
            break

        sent_candidates = get_answer_candidates(sent)
        if not sent_candidates:
            continue

        answer, label = sent_candidates[0]

        if answer.lower() in used_answers:
            found = False
            for ans, lbl in sent_candidates[1:]:
                if ans.lower() not in used_answers:
                    answer, label = ans, lbl
                    found = True
                    break
            if not found:
                continue

        used_answers.add(answer.lower())
        sentence_text = sent.text.strip()

        if len(mcqs) % 2 == 0:
            question, q_type = make_wh_question(sentence_text, answer, label)
        else:
            question, q_type = make_fill_question(sentence_text, answer)

        distractors = get_distractors(answer, full_doc_candidates, difficulty)
        options = [answer] + distractors
        random.shuffle(options)
        correct_index = options.index(answer)

        mcqs.append({
            "question":      question,
            "options":       options,
            "answer":        answer,
            "correct_index": correct_index,
            "difficulty":    difficulty,
            "type":          q_type,
            "label":         label,
        })

    return mcqs


# ── ROUTES ─────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        files         = request.files.getlist("files[]")
        num_questions = int(request.form.get("num_questions", 10))
        difficulty    = request.form.get("difficulty", "medium")

        combined_text = ""
        for f in files:
            if f and f.filename:
                combined_text += " " + extract_text(f)

        combined_text = combined_text.strip()
        if not combined_text:
            return render_template("index.html",
                error="Could not extract text from the uploaded file(s).")

        mcqs = generate_mcqs(combined_text, num_questions, difficulty)
        if not mcqs:
            return render_template("index.html",
                error="Not enough content to generate questions. Try a longer document.")

        # Save session to history
        save_session(num_questions, difficulty, engine="spacy")

        return render_template("mcqs.html", mcqs=mcqs,
                               num_questions=num_questions, engine="spacy")

    return render_template("index.html")


@app.route("/stats")
def stats():
    history = get_history()
    return render_template("stats.html", history=history)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
