# Install in terminal one by one
# pip install flask-bootstrap
# pip install flask
# pip install spacy
# pip install PyPDF2
# python -m spacy download en_core_web_sm
 
from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import spacy
from collections import Counter
import random
import re
import PyPDF2
from PyPDF2 import PdfReader
 
app = Flask(__name__)
Bootstrap(app)
 
# Load English NLP model
nlp = spacy.load("en_core_web_sm")
 
# ─── Stopwords ────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "and", "but", "or", "nor", "so",
    "yet", "both", "either", "neither", "not", "only", "own", "same",
    "than", "too", "very", "just", "that", "this", "these", "those",
    "it", "its", "they", "them", "their", "he", "she", "we", "you",
    "i", "me", "my", "your", "our", "his", "her", "which", "who", "what",
    "also", "however", "therefore", "thus", "hence", "although", "since",
    "such", "each", "every", "any", "all", "most", "other", "more", "some"
}
 
# ─── Question Templates ────────────────────────────────────────────────────────
 
def make_wh_question(sentence, subject, answer, doc):
    """Generate Wh-style or How-style questions based on NER / POS tags."""
    sent_doc = nlp(sentence)
 
    # Detect entity label of the answer for smarter question word
    ent_labels = {ent.text: ent.label_ for ent in sent_doc.ents}
    label = ent_labels.get(answer, "")
 
    # Map entity label → question word
    label_to_qword = {
        "PERSON":   "Who",
        "ORG":      "Which organization",
        "GPE":      "Where",        # Geopolitical entity (city/country)
        "LOC":      "Where",
        "DATE":     "When",
        "TIME":     "When",
        "MONEY":    "How much",
        "QUANTITY": "How many",
        "PERCENT":  "What percentage",
        "PRODUCT":  "What",
        "EVENT":    "What",
        "LAW":      "What",
        "LANGUAGE": "What language",
        "NORP":     "Which group",   # Nationalities, religions, political groups
        "FAC":      "Where",         # Facilities
        "WORK_OF_ART": "What",
        "CARDINAL": "How many",
        "ORDINAL":  "Which",
    }
 
    question_word = label_to_qword.get(label, "What")
 
    # Check for HOW patterns via verb context
    verb_tokens = [t.text.lower() for t in sent_doc if t.pos_ == "VERB"]
    how_verbs = {"process", "work", "function", "operate", "occur", "happen",
                 "develop", "form", "create", "produce", "generate", "achieve",
                 "improve", "increase", "decrease", "change", "affect", "cause"}
    if any(v in how_verbs for v in verb_tokens):
        question_word = "How"
 
    # Build clean question stem: replace answer with blank + question word prefix
    blank_sentence = re.sub(r'\b' + re.escape(answer) + r'\b', "______", sentence, count=1)
    blank_sentence = blank_sentence.strip().rstrip(".")
 
    question_stem = f"{question_word} is referred to as '______' in: \"{blank_sentence}\"?"
    return question_stem, question_word
 
 
def clean_text(text):
    """Remove excessive whitespace and non-printable characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E]', ' ', text)
    return text.strip()
 
 
def is_meaningful_token(token_text):
    """Return True if token is a meaningful keyword (not stopword, not too short)."""
    t = token_text.lower().strip()
    return (
        len(t) > 2 and
        t not in STOPWORDS and
        t.isalpha()
    )
 
 
def get_distractors(answer, doc, difficulty, num=3):
    """
    Pick contextually relevant distractors:
    - Easy:   other nouns from the same sentence
    - Medium: nouns from the whole text
    - Hard:   named entities or nouns from the whole text (semantically challenging)
    """
    answer_lower = answer.lower()
 
    if difficulty == 'hard':
        candidates = list({
            ent.text for ent in doc.ents
            if ent.text.lower() != answer_lower and is_meaningful_token(ent.text)
        })
    else:
        candidates = list({
            token.text for token in doc
            if token.pos_ in ("NOUN", "PROPN")
            and token.text.lower() != answer_lower
            and is_meaningful_token(token.text)
        })
 
    random.shuffle(candidates)
    distractors = candidates[:num]
 
    # Pad with generic placeholders if not enough
    placeholders = ["[Option A]", "[Option B]", "[Option C]"]
    while len(distractors) < num:
        distractors.append(placeholders[len(distractors) % 3])
 
    return distractors
 
 
def generate_mcqs(text, num_questions=5, difficulty='medium'):
    if not text or not text.strip():
        return []
 
    text = clean_text(text)
    doc = nlp(text[:100000])  # Cap at 100k chars for performance
 
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
 
    # Filter sentences by difficulty (word count)
    if difficulty == 'easy':
        filtered = [s for s in sentences if 8 <= len(s.split()) <= 18]
    elif difficulty == 'hard':
        filtered = [s for s in sentences if len(s.split()) >= 22]
    else:
        filtered = [s for s in sentences if 12 <= len(s.split()) <= 30]
 
    if len(filtered) < num_questions:
        filtered = sentences  # fallback to all
 
    random.shuffle(filtered)
 
    mcqs = []
    used_sentences = set()
 
    for sentence in filtered:
        if len(mcqs) >= num_questions:
            break
        if sentence in used_sentences:
            continue
 
        sent_doc = nlp(sentence)
 
        # Prefer named entities as answer; fall back to important nouns
        candidates = [
            ent.text for ent in sent_doc.ents
            if is_meaningful_token(ent.text)
        ]
        if not candidates:
            candidates = [
                token.text for token in sent_doc
                if token.pos_ in ("NOUN", "PROPN") and is_meaningful_token(token.text)
            ]
 
        if not candidates:
            continue
 
        # Pick the most frequent meaningful candidate as the answer
        freq = Counter(candidates)
        answer = freq.most_common(1)[0][0]
 
        # Generate Wh / How question
        question_stem, q_type = make_wh_question(sentence, answer, answer, doc)
 
        # Build answer choices
        distractors = get_distractors(answer, doc, difficulty, num=3)
        choices = [answer] + distractors
        random.shuffle(choices)
 
        correct_letter = chr(65 + choices.index(answer))  # A, B, C, D
 
        mcqs.append((question_stem, choices, correct_letter, difficulty, q_type))
        used_sentences.add(sentence)
 
    return mcqs
 
 
# ─── PDF Helper ───────────────────────────────────────────────────────────────
 
def process_pdf(file):
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    except Exception as e:
        print(f"PDF read error: {e}")
    return text
 
 
# ─── Routes ───────────────────────────────────────────────────────────────────
 
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = ""
 
        if 'files[]' in request.files:
            files = request.files.getlist('files[]')
            for file in files:
                if file.filename.endswith('.pdf'):
                    text += process_pdf(file)
                elif file.filename.endswith('.txt'):
                    text += file.read().decode('utf-8', errors='ignore')
        else:
            text = request.form.get('text', '')
 
        num_questions = int(request.form.get('num_questions', 5))
        difficulty = request.form.get('difficulty', 'medium')
 
        mcqs = generate_mcqs(text, num_questions=num_questions, difficulty=difficulty)
        mcqs_with_index = [(i + 1, mcq) for i, mcq in enumerate(mcqs)]
 
        return render_template('mcqs.html', mcqs=mcqs_with_index)
 
    return render_template('index.html')
 
 
if __name__ == '__main__':
    app.run(debug=True)