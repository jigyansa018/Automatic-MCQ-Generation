# pip install flask spacy PyPDF2
# python -m spacy download en_core_web_sm

from flask import Flask, render_template, request
import spacy
import random
import string
import sqlite3
import re
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


# ── PDF / TEXT CLEANING ────────────────────────────────────────────────────────
_NOISE_PATTERNS = re.compile(
    r"""
    ^\s*\d+\s*$                        # lone page numbers
    | ^\s*page\s+\d+                   # "Page 3"
    | ^\s*chapter\s+\d+                # "Chapter 1"
    | ^\s*figure\s+\d+                 # "Figure 2"
    | ^\s*table\s+\d+                  # "Table 4"
    | ^\s*\d+\.\d+(\.\d+)*\s          # "1.2.3 Section heading"
    | @                                # email addresses
    | https?://                        # URLs
    | copyright|\(c\)                  # copyright lines
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _is_mostly_upper(line: str) -> bool:
    """Lines that are mostly uppercase are headings, names, or institution labels."""
    letters = [c for c in line if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isupper()) / len(letters) > 0.6

def clean_text(raw: str) -> str:
    """
    Strip noise lines from raw extracted text, then re-join into clean prose.
    Handles:
      - Page numbers, figure/table captions, section numbers
      - Author names / institution lines (mostly-uppercase)
      - URLs, emails, copyright lines
      - Lines with fewer than 5 words (headers, labels)
    """
    lines = raw.splitlines()
    good_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _NOISE_PATTERNS.search(stripped):
            continue
        if _is_mostly_upper(stripped):
            continue
        if len(stripped.split()) < 5:          # too short to be real content
            continue
        good_lines.append(stripped)

    text = " ".join(good_lines)
    text = re.sub(r'\s+', ' ', text)
    # Remove isolated single capital letters (OCR noise, e.g. "S .")
    text = re.sub(r'\b[A-Z]\b\.?\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── FILE EXTRACTION ────────────────────────────────────────────────────────────
def extract_text(file):
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
        raw = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                raw += t + "\n"
        return clean_text(raw)
    elif filename.endswith(".txt"):
        return clean_text(file.read().decode("utf-8", errors="ignore"))
    return ""


# ── SENTENCE QUALITY SCORING ───────────────────────────────────────────────────
_CONCEPT_VERBS = {
    "is","are","was","were","refers","defines","means","represents",
    "consists","contains","includes","involves","requires","provides",
    "enables","allows","supports","uses","performs","processes",
    "stores","manages","controls","creates","generates","converts",
    "transmits","receives","connects","operates","executes","runs",
    "handles","implements","describes","indicates","shows","demonstrates",
    "called","known","defined","used","applied","classified","divided",
}

def _sentence_quality_score(sent) -> float:
    """
    Score a sentence 0–1 for suitability as a quiz question source.

    Penalise:
      - High proportion of PROPN (names, places) → likely metadata
      - No common NOUN (not a conceptual statement)
      - No VERB (incomplete sentence)

    Reward:
      - Has subject + verb + object structure
      - Contains concept-level vocabulary
      - Reasonable length (10–50 words)
    """
    tokens = [t for t in sent if not t.is_space and not t.is_punct]
    if not tokens:
        return 0.0

    total        = len(tokens)
    propn_count  = sum(1 for t in tokens if t.pos_ == "PROPN")
    verb_count   = sum(1 for t in tokens if t.pos_ == "VERB")
    noun_count   = sum(1 for t in tokens if t.pos_ == "NOUN")   # common nouns only

    # Hard disqualifiers
    if propn_count / total > 0.40:   # more than 40 % proper nouns → noise
        return 0.0
    if verb_count == 0:
        return 0.0
    if noun_count == 0:              # must have at least one common noun
        return 0.0

    has_concept_verb = any(t.lemma_.lower() in _CONCEPT_VERBS for t in tokens)

    score  = 0.5
    score -= (propn_count / total) * 0.5
    score += min(verb_count / total, 0.15)
    score += min(noun_count / total, 0.20)
    if has_concept_verb:
        score += 0.15

    return min(max(score, 0.0), 1.0)


def get_good_sentences(doc):
    """
    Return sentences sorted best-first.
    Filters:
      - length 10–55 tokens
      - starts with uppercase letter
      - ends with sentence-final punctuation
      - quality score ≥ 0.4
    """
    good = []
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_space]
        if not (10 <= len(tokens) <= 55):
            continue
        text = sent.text.strip()
        if not text or not text[0].isupper():
            continue
        if text[-1] not in ".!?":
            continue
        score = _sentence_quality_score(sent)
        if score >= 0.4:
            good.append((score, sent))

    good.sort(key=lambda x: x[0], reverse=True)
    return [sent for _, sent in good]


# ── ANSWER CANDIDATES ──────────────────────────────────────────────────────────
# Only entity types that make sense as quiz answers
_ALLOWED_ENT_TYPES = {
    "ORG", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    "DATE", "TIME", "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL",
}

def get_answer_candidates(span):
    candidates = []
    seen = set()

    # Named entities — only allowed types (skip PERSON, GPE, LOC — too noisy)
    for ent in span.ents:
        if ent.label_ not in _ALLOWED_ENT_TYPES:
            continue
        word = ent.text.strip()
        if len(word) > 1 and word.lower() not in STOPWORDS and word.lower() not in seen:
            seen.add(word.lower())
            candidates.append((word, ent.label_))

    # Noun chunks — only if they contain at least one common noun (not all PROPN)
    for chunk in span.noun_chunks:
        if not any(t.pos_ == "NOUN" for t in chunk):
            continue
        word = chunk.text.strip()
        if len(word) > 1 and word.lower() not in STOPWORDS and word.lower() not in seen:
            seen.add(word.lower())
            candidates.append((word, "NOUN_CHUNK"))

    # Individual common nouns and adjectives only
    for token in span:
        if token.pos_ not in ("NOUN", "ADJ"):
            continue
        word = token.text.strip()
        if (len(word) > 2
                and word.lower() not in STOPWORDS
                and not token.is_punct
                and not token.is_space
                and word.lower() not in seen):
            seen.add(word.lower())
            candidates.append((word, token.pos_))

    return candidates


# ── WH-QUESTION — natural, NO "Who:" / "What:" prefix ────────────────────────
WH_MAP = {
    "ORG":         "Which organization",
    "DATE":        "When",
    "TIME":        "When",
    "MONEY":       "How much",
    "CARDINAL":    "How many",
    "ORDINAL":     "Which",
    "PERCENT":     "What percentage",
    "PRODUCT":     "Which product",
    "EVENT":       "What event",
    "WORK_OF_ART": "What",
    "LAW":         "Which law",
    "LANGUAGE":    "Which language",
    "QUANTITY":    "How much",
    "NOUN_CHUNK":  "What",
    "NOUN":        "What",
    "ADJ":         "How",
}
PROCESS_WORDS = {"process","cause","affect","work","function","method",
                 "technique","approach","mechanism","system","way","means","manner"}

def make_wh_question(sentence_text, answer, label):
    """
    Replace the answer inside the sentence with the WH question word.

    Example:
      "Embedded software resides in read-only memory."
      answer = "memory"  → "What does embedded software reside in?"
      (simple replacement approach)
    """
    q_word = WH_MAP.get(label, "What")
    if any(pw in answer.lower() for pw in PROCESS_WORDS):
        q_word = "How"

    q_sentence = sentence_text.replace(answer, q_word, 1).strip().rstrip(".")
    if q_sentence:
        q_sentence = q_sentence[0].upper() + q_sentence[1:]
    if not q_sentence.endswith("?"):
        q_sentence += "?"

    return q_sentence, "WH"


# ── FILL-IN-THE-BLANK — NO prefix ────────────────────────────────────────────
def make_fill_question(sentence_text, answer):
    """
    Replace the answer with a blank.  No 'Fill in the blank:' prefix.

    Example:
      "The CPU executes instructions stored in memory."
      answer = "instructions"
      result = "The CPU executes ________ stored in memory."
    """
    q_sentence = sentence_text.replace(answer, "________", 1).strip()
    return q_sentence, "FILL"


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
    generic = ["None of the above", "All of the above",
               "Cannot be determined", "Not mentioned in the text"]
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
    sentences = get_good_sentences(doc)   # already sorted best-first
    if not sentences:
        return []

    full_doc_candidates = get_answer_candidates(doc)
    mcqs        = []
    used_answers = set()

    for sent in sentences:
        if len(mcqs) >= num_questions:
            break

        sent_candidates = get_answer_candidates(sent)
        if not sent_candidates:
            continue

        # Pick the first unused, meaningful answer from this sentence
        answer, label = None, None
        for ans, lbl in sent_candidates:
            if ans.lower() not in used_answers and ans in sent.text:
                answer, label = ans, lbl
                break
        if answer is None:
            continue

        used_answers.add(answer.lower())
        sentence_text = sent.text.strip()

        # Alternate: even → WH question, odd → fill-in-the-blank
        if len(mcqs) % 2 == 0:
            question, q_type = make_wh_question(sentence_text, answer, label)
        else:
            question, q_type = make_fill_question(sentence_text, answer)

        distractors   = get_distractors(answer, full_doc_candidates, difficulty)
        options       = [answer] + distractors
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
