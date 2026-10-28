"""
Intelligent Patient Feedback Analytics System for Blood Tests
=============================================================
Somaiya Vidyavihar University | KJSSE / AIDS / TY B.Tech / Sem-V | ITA Elective

Data files (next to this script):
  patient_feedback_dataset.csv   600-ish synthetic feedback records used by every page
  ir_documents.csv               12 branch reports used by Expt 6
If a file is missing the app regenerates it and saves it, so you can also edit the CSVs by hand
or upload your own CSV (same columns) from the sidebar.

ONE file, ALL 8 experiments, applied to patient feedback for blood tests:

  Expt 1  Text preprocessing pipeline (lowercase -> POS tagging)
  Expt 2  BoW vs TF-IDF vs Word2Vec
  Expt 3  Text classification: Naive Bayes vs Linear SVM vs Decision Tree
  Expt 4  Text clustering: Hierarchical (dendrogram) vs DBSCAN
  Expt 5  Topic modelling: LDA vs NMF
  Expt 6  Information retrieval: TF-IDF + NER + cosine similarity
  Expt 7  Aspect-Based Sentiment Analysis (+ document sentiment + department routing)
  Expt 8  Summarization: TextRank vs Transformer (BART) + ROUGE
  + Dashboard and Live Feedback Analyzer that tie everything together

Run:
    pip install streamlit pandas numpy scikit-learn scipy matplotlib gensim nltk networkx
    # optional (the app degrades gracefully without them):
    pip install spacy rouge-score transformers torch
    python -m spacy download en_core_web_sm
    streamlit run patient_feedback_app.py

The dataset is generated inside this file (synthetic, seeded, no real patient data).
"""

import re
import string
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, silhouette_score)
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
SEED = 42

st.set_page_config(page_title="Patient Feedback Analytics - Blood Tests", page_icon="🩸", layout="wide")

# =====================================================================================
# 0. OPTIONAL LIBRARIES / NLTK DATA (everything falls back gracefully)
# =====================================================================================
try:
    import nltk
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except Exception:  # pragma: no cover
    nltk = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import gensim
    from gensim.models import Word2Vec
except Exception:  # pragma: no cover
    gensim = None


@st.cache_resource(show_spinner="Checking NLTK data (first run downloads a few small packages)...")
def nltk_status():
    """Try to make NLTK resources available. Returns dict of flags."""
    flags = {"punkt": False, "stopwords": False, "wordnet": False, "tagger": False}
    if nltk is None:
        return flags

    def have(path):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            return False

    wanted = {
        "punkt": [("tokenizers/punkt_tab", "punkt_tab"), ("tokenizers/punkt", "punkt")],
        "stopwords": [("corpora/stopwords", "stopwords")],
        "wordnet": [("corpora/wordnet", "wordnet")],
        "tagger": [("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
                   ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")],
    }
    for key, options in wanted.items():
        for path, pkg in options:
            if have(path):
                flags[key] = True
                break
        if not flags[key]:
            for path, pkg in options:
                try:
                    nltk.download(pkg, quiet=True)
                except Exception:
                    pass
                if have(path):
                    flags[key] = True
                    break
    # a real tokenizer smoke test (nltk can find a path yet fail to load)
    if flags["punkt"]:
        try:
            nltk.word_tokenize("test sentence.")
            nltk.sent_tokenize("One. Two.")
        except Exception:
            flags["punkt"] = False
    if flags["tagger"]:
        try:
            nltk.pos_tag(["test"])
        except Exception:
            flags["tagger"] = False
    if flags["wordnet"]:
        try:
            WordNetLemmatizer().lemmatize("tests")
        except Exception:
            flags["wordnet"] = False
    return flags


@st.cache_resource(show_spinner=False)
def load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


# =====================================================================================
# 1. DATASET GENERATOR - "Intelligent Patient Feedback Analytics System for Blood Tests"
# =====================================================================================
# aspect -> (department, keywords, positive phrases, negative phrases)
ASPECTS = {
    "Staff behaviour": ("Sample Collection",
        ["staff", "phlebotomist", "technician", "nurse", "receptionist"],
        ["the phlebotomist was polite and professional", "the staff were friendly and helpful",
         "the nurse was caring and gentle", "the staff were not rude at all"],
        ["the phlebotomist was rude and careless", "the staff were unhelpful",
         "the receptionist was dismissive", "the staff were not helpful"]),
    "Waiting time": ("Front Desk",
        ["wait", "waiting", "queue", "crowd", "crowded"],
        ["the waiting time was short", "the queue moved quickly",
         "the waiting time was minimal", "the wait was not long"],
        ["the waiting time was very long", "the queue was endless",
         "the centre was crowded and chaotic", "the wait was too long"]),
    "Pain & needle": ("Sample Collection",
        ["needle", "prick", "vein", "bruise", "draw", "painless"],
        ["the needle prick was painless", "the blood draw was quick and painless",
         "the needle was gentle", "the needle prick was not painful"],
        ["the needle prick was painful", "finding the vein took multiple painful attempts",
         "it left a bad bruise", "the blood draw was not painless"]),
    "Report turnaround": ("Reporting",
        ["report", "reports", "results", "turnaround"],
        ["the report was delivered quickly", "the report was timely",
         "the results were ready fast", "the report was not late"],
        ["the report was delayed", "the results were late",
         "the report took too long", "the report was not timely"]),
    "Report accuracy": ("Reporting",
        ["accurate", "accuracy", "inaccurate", "error", "mistake", "retest", "readings", "values"],
        ["the values were accurate and reliable", "my doctor confirmed the readings were accurate",
         "the accuracy was excellent", "there was no error in the values"],
        ["the readings were inaccurate", "there was an error in my name and values",
         "i needed a retest because of a mistake", "the values were not accurate"]),
    "Price & billing": ("Billing",
        ["price", "prices", "cost", "charges", "billing", "bill", "fees", "package"],
        ["the price was affordable", "the charges were reasonable",
         "billing was transparent", "the package was good value"],
        ["the price was expensive", "billing had hidden charges",
         "the package was overpriced", "the charges were not reasonable"]),
    "Home collection": ("Home Collection",
        ["home", "pickup", "slot"],
        ["the home pickup was punctual", "the home visit was smooth",
         "the slot was flexible and convenient", "the home pickup was not delayed"],
        ["the home pickup was delayed", "the home visit was cancelled",
         "the slot was missed", "the home pickup was not punctual"]),
    "Instructions": ("Digital & Support",
        ["fasting", "instructions", "guidance", "preparation", "advice"],
        ["the fasting instructions were clear", "the preparation guidance was helpful",
         "the advice was useful", "the fasting instructions were not confusing"],
        ["the fasting instructions were confusing", "the preparation guidance was unclear",
         "the advice was contradictory", "the fasting instructions were not clear"]),
    "App & portal": ("Digital & Support",
        ["app", "portal", "website", "booking", "online", "login"],
        ["the booking app was easy to use", "the online portal was smooth",
         "the website was fast", "the app was not slow"],
        ["the app kept crashing", "the booking portal was confusing",
         "the website was slow", "the login failed repeatedly"]),
    "Hygiene": ("Sample Collection",
        ["hygiene", "clean", "gloves", "sterile", "dirty", "spotless", "unhygienic", "room"],
        ["hygiene was excellent", "the collection room was spotless",
         "the gloves were fresh and the equipment sterile", "the room was not dirty"],
        ["hygiene was poor", "the collection room was dirty and unhygienic",
         "the room was unclean", "the room was not clean"]),
}
ASPECT_KEYWORDS = {a: set(v[1]) for a, v in ASPECTS.items()}
ASPECT_DEPT = {a: v[0] for a, v in ASPECTS.items()}
DEPARTMENTS = sorted(set(ASPECT_DEPT.values()))

POS_WORDS = {"excellent", "great", "good", "quick", "quickly", "fast", "painless", "gentle", "polite", "friendly",
             "clean", "spotless", "sterile", "accurate", "reliable", "helpful", "affordable", "reasonable",
             "smooth", "punctual", "professional", "easy", "clear", "timely", "short", "minimal", "caring",
             "transparent", "flexible", "convenient", "useful", "fresh", "amazing", "best", "loved", "happy",
             "satisfied", "recommend", "enjoyable", "ready", "moved"}
NEG_WORDS = {"long", "late", "delayed", "slow", "painful", "rude", "dirty", "unhygienic", "unclean", "inaccurate",
             "wrong", "error", "mistake", "expensive", "overpriced", "hidden", "confusing", "unclear",
             "contradictory", "crashing", "crashed", "failed", "unhelpful", "careless", "dismissive", "bad",
             "poor", "terrible", "worst", "crowded", "chaotic", "endless", "cancelled", "missed", "disappointed",
             "unsatisfied", "unpleasant", "retest", "impatient", "frustrating"}
NEGATORS = {"not", "no", "never", "hardly", "without"}

TESTS = ["CBC", "Lipid profile", "HbA1c", "Thyroid profile", "Vitamin D", "Liver function", "Kidney function",
         "Fasting blood sugar"]
BRANCHES = ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Navi Mumbai"]

POS_OPEN = ["Really happy with my blood test experience.", "Great experience getting my {t} test done.",
            "Would recommend this diagnostic lab.", "Very satisfied with this lab."]
NEG_OPEN = ["Very disappointed with my blood test experience.", "Unpleasant experience getting my {t} test done.",
            "Would not recommend this diagnostic lab.", "Totally unsatisfied with this lab."]
NEUTRAL_OPEN = ["Went for a {t} test last week.", "Got my {t} done at this lab.",
                "Booked a {t} test on my doctor's suggestion.", "Took a {t} test for my annual checkup."]
POS_CLOSE = ["Will visit again.", "Five stars.", ""]
NEG_CLOSE = ["Will not visit again.", "One star.", ""]
NEUTRAL_CLOSE = ["Sharing my feedback.", "That was my experience.", ""]


@st.cache_data(show_spinner=False)
def make_dataset(n_per_class: int = 300, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []

    def one(label, noisy):
        test = rng.choice(TESTS)
        n_asp = rng.choice([1, 2, 2, 3])
        aspects = rng.sample(list(ASPECTS), n_asp)
        parts, truth = [], []
        for i, a in enumerate(aspects):
            flip = noisy and n_asp == 3 and i == n_asp - 1  # only 3-aspect reviews: majority still defines the label
            want_pos = (label == "Positive") != flip
            phrases = ASPECTS[a][2] if want_pos else ASPECTS[a][3]
            parts.append(rng.choice(phrases))
            truth.append(f"{a}:{'Positive' if want_pos else 'Negative'}")
        opens = POS_OPEN if label == "Positive" else NEG_OPEN
        closes = POS_CLOSE if label == "Positive" else NEG_CLOSE
        if rng.random() < 0.55:
            opens = NEUTRAL_OPEN
        if rng.random() < 0.65:
            closes = NEUTRAL_CLOSE
        body = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
        text = f"{rng.choice(opens).format(t=test)} {body} {rng.choice(closes)}".strip()
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=rng.randint(0, 270))
        rating = rng.choice([4, 5]) if label == "Positive" else rng.choice([1, 2])
        return dict(branch=rng.choice(BRANCHES), test=test, date=date, rating=rating,
                    main_aspect=aspects[0], department=ASPECT_DEPT[aspects[0]],
                    feedback=text, sentiment=label, truth_pairs=";".join(truth))

    for _ in range(n_per_class):
        rows.append(one("Positive", rng.random() < 0.40))
    for _ in range(n_per_class):
        rows.append(one("Negative", rng.random() < 0.40))
    df = pd.DataFrame(rows).drop_duplicates(subset="feedback").sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "feedback_id", [f"F{i + 1}" for i in range(len(df))])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    return df


DATA_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATASET_CSV = DATA_DIR / "patient_feedback_dataset.csv"
IR_CSV = DATA_DIR / "ir_documents.csv"
REQUIRED_COLS = ["feedback_id", "branch", "test", "date", "rating", "main_aspect", "department", "feedback",
                 "sentiment", "truth_pairs"]


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["rating"] = df["rating"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def read_dataset_csv(path: str, mtime: float) -> pd.DataFrame:
    return _prepare(pd.read_csv(path))


def load_dataset(uploaded=None) -> tuple:
    """Return (dataframe, source description). Order: uploaded CSV > CSV beside the script > generate + save."""
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.sidebar.error(f"Uploaded CSV is missing columns: {missing}. Using the default dataset instead.")
        else:
            return _prepare(df), f"uploaded file ({len(df)} rows)"
    if not DATASET_CSV.exists():
        make_dataset().drop(columns=["month"]).to_csv(DATASET_CSV, index=False)
    df = read_dataset_csv(str(DATASET_CSV), DATASET_CSV.stat().st_mtime)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"{DATASET_CSV.name} is missing columns {missing}. Delete it to regenerate the default file.")
        st.stop()
    return df, f"{DATASET_CSV.name} ({len(df)} rows)"


def load_ir_docs() -> pd.DataFrame:
    cols = ["Document_ID", "Location", "Date", "Test", "Paragraph"]
    if not IR_CSV.exists():
        pd.DataFrame(IR_DOCS, columns=cols).to_csv(IR_CSV, index=False)
    ir = pd.read_csv(IR_CSV)
    if not set(cols) <= set(ir.columns):
        st.error(f"{IR_CSV.name} must have columns {cols}. Delete it to regenerate the default file.")
        st.stop()
    return ir[cols]


# =====================================================================================
# 2. TEXT UTILITIES (Expt 1 pipeline, with graceful fallbacks)
# =====================================================================================
def sent_split(text: str):
    flags = nltk_status()
    text = text.strip()
    if nltk is not None and flags["punkt"]:
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except Exception:
            pass
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def word_tok(text: str):
    flags = nltk_status()
    if nltk is not None and flags["punkt"]:
        try:
            return nltk.word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"\w+|[^\w\s]", text)


def get_stopwords():
    flags = nltk_status()
    if nltk is not None and flags["stopwords"]:
        try:
            from nltk.corpus import stopwords
            return set(stopwords.words("english"))
        except Exception:
            pass
    return set(ENGLISH_STOP_WORDS)


def preprocess_steps(text: str):
    """Expt 1 pipeline. Returns an ordered dict of every intermediate output."""
    flags = nltk_status()
    sw = get_stopwords()
    lower = text.lower()
    cleaned = lower.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tok(cleaned)
    filtered = [w for w in tokens if w not in sw]
    stemmer = PorterStemmer() if nltk is not None else None
    stemmed = [stemmer.stem(w) for w in filtered] if stemmer else filtered
    if nltk is not None and flags["wordnet"]:
        lem = WordNetLemmatizer()
        lemmas = [lem.lemmatize(w) for w in filtered]
    else:
        lemmas = list(filtered)
    if nltk is not None and flags["tagger"]:
        try:
            pos = nltk.pos_tag(lemmas)
        except Exception:
            pos = [(w, "n/a") for w in lemmas]
    else:
        pos = [(w, "n/a") for w in lemmas]
    return {"Original Text": text, "Lowercase Text": lower, "Cleaned Text (no punctuation)": cleaned,
            "Tokens": tokens, "Filtered Tokens (stopwords removed)": filtered, "Stemmed Words": stemmed,
            "Lemmatized Words": lemmas, "POS Tagged Output": pos}


def simple_clean(t: str) -> str:
    t = re.sub(r"<.*?>", " ", str(t))
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    return re.sub(r"\s+", " ", t).lower().strip()


def fig_show(fig):
    st.pyplot(fig)
    plt.close(fig)


# =====================================================================================
# 3. ABSA ENGINE (Expt 7) - shared by Dashboard and Live Analyzer
# =====================================================================================
CLAUSE_SPLIT = re.compile(r"[.;,!?]|\bbut\b|\band\b|\bhowever\b|\balthough\b|\bwhile\b|\bthough\b")


def absa_extract(text: str):
    """Aspect -> sentiment pairs. Clause-level window, simple negation handling."""
    found = {}
    for clause in CLAUSE_SPLIT.split(text.lower()):
        toks = re.findall(r"[a-z']+", clause)
        if not toks:
            continue
        aspects = [a for a, kws in ASPECT_KEYWORDS.items() if any(t in kws for t in toks)]
        if not aspects:
            continue
        score = 0
        for i, t in enumerate(toks):
            s = 1 if t in POS_WORDS else -1 if t in NEG_WORDS else 0
            if s and any(x in NEGATORS for x in toks[max(0, i - 3):i]):
                s = -s
            score += s
        sent = "Positive" if score > 0 else "Negative" if score < 0 else "Neutral"
        for a in aspects:
            if a not in found or found[a] == "Neutral":
                found[a] = sent
    return list(found.items())


def doc_sentiment_lexicon(text: str) -> str:
    """Spec-style document classifier: count lexicon hits, no negation, no aspects."""
    words = re.findall(r"[a-z']+", text.lower())
    p = sum(w in POS_WORDS for w in words)
    n = sum(w in NEG_WORDS for w in words)
    return "Positive" if p > n else "Negative" if n > p else "Neutral"


DEPT_KEYWORDS = {}
for _a, _kw in ASPECT_KEYWORDS.items():
    DEPT_KEYWORDS.setdefault(ASPECT_DEPT[_a], set()).update(_kw)


def route_department(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    scores = {d: sum(w in kws for w in words) for d, kws in DEPT_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return "Other" if scores[best] == 0 else best


@st.cache_data(show_spinner=False)
def absa_corpus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid, text in zip(df["feedback_id"], df["feedback"]):
        for a, s in absa_extract(text):
            rows.append((fid, a, s))
    return pd.DataFrame(rows, columns=["feedback_id", "Aspect", "Sentiment"])


# =====================================================================================
# 4. SHARED MODEL TRAINING (Expt 3) - also used by Live Analyzer
# =====================================================================================
def make_models():
    return {"Naive Bayes": MultinomialNB(), "Linear SVM": LinearSVC(random_state=SEED),
            "Decision Tree": DecisionTreeClassifier(random_state=SEED)}


@st.cache_resource(show_spinner="Training classifiers...")
def train_full_models(df: pd.DataFrame):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(df["feedback"])
    models = make_models()
    for m in models.values():
        m.fit(X, df["sentiment"])
    return vec, models


# =====================================================================================
# PAGE: DASHBOARD
# =====================================================================================
def page_dashboard(df):
    st.title("🩸 Patient Feedback Analytics - Blood Tests")
    st.caption("Dashboard over the generated feedback corpus. Every number below is computed from the data on this page.")
    with st.expander("Filters", expanded=False):
        c1, c2 = st.columns(2)
        branches = c1.multiselect("Branch", BRANCHES, default=BRANCHES)
        tests = c2.multiselect("Blood test", TESTS, default=TESTS)
    f = df[df["branch"].isin(branches) & df["test"].isin(tests)]
    if f.empty:
        st.warning("No feedback matches the filters.")
        return
    neg = (f["sentiment"] == "Negative").mean()
    k = st.columns(4)
    k[0].metric("Feedback records", f"{len(f):,}")
    k[1].metric("Average rating", f"{f['rating'].mean():.2f} / 5")
    k[2].metric("Negative share", f"{neg:.1%}")
    ab = absa_corpus(df)
    ab_f = ab[ab["feedback_id"].isin(f["feedback_id"])]
    worst = (ab_f[ab_f.Sentiment != "Neutral"].groupby("Aspect").Sentiment
             .apply(lambda s: (s == "Negative").mean()).sort_values(ascending=False))
    k[3].metric("Most criticised aspect", worst.index[0] if len(worst) else "-",
                f"{worst.iloc[0]:.0%} negative" if len(worst) else None, delta_color="off")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Aspect-wise sentiment (ABSA)")
        ct = pd.crosstab(ab_f["Aspect"], ab_f["Sentiment"])
        st.bar_chart(ct)
    with c2:
        st.subheader("Negative share by branch")
        st.bar_chart(f.groupby("branch")["sentiment"].apply(lambda s: (s == "Negative").mean()).sort_values())
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Negative share by blood test")
        st.bar_chart(f.groupby("test")["sentiment"].apply(lambda s: (s == "Negative").mean()).sort_values())
    with c2:
        st.subheader("Monthly negative share")
        st.line_chart(f.groupby("month")["sentiment"].apply(lambda s: (s == "Negative").mean()))
    st.subheader("Department routing (by main aspect of each feedback)")
    st.bar_chart(f["department"].value_counts())
    with st.expander("Browse the generated feedback"):
        st.dataframe(f[["feedback_id", "branch", "test", "date", "rating", "department", "sentiment", "feedback"]])


# =====================================================================================
# PAGE: LIVE ANALYZER
# =====================================================================================
def page_live(df):
    st.title("🔎 Live Feedback Analyzer")
    st.caption("Type any patient feedback. It runs through preprocessing, three classifiers, ABSA and department routing.")
    default = "The phlebotomist was polite but the report was delayed and the billing had hidden charges."
    text = st.text_area("Patient feedback", default, height=100)
    if not text.strip():
        return
    vec, models = train_full_models(df)
    Xn = vec.transform([text])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Overall sentiment (Expt 3 models)")
        st.table(pd.DataFrame({"Model": list(models), "Prediction": [m.predict(Xn)[0] for m in models.values()]}))
        st.subheader("Department (Expt 7 routing)")
        st.write(route_department(text))
    with c2:
        st.subheader("Aspect-level sentiment (Expt 7 ABSA)")
        pairs = absa_extract(text)
        if pairs:
            st.table(pd.DataFrame(pairs, columns=["Aspect", "Sentiment"]))
        else:
            st.info("No known aspect found in this text.")
    st.subheader("Preprocessing (Expt 1)")
    steps = preprocess_steps(text)
    st.write("**Filtered tokens:**", steps["Filtered Tokens (stopwords removed)"])
    st.write("**Lemmas:**", steps["Lemmatized Words"])


# =====================================================================================
# PAGE: EXPT 1
# =====================================================================================
def page_expt1(df):
    st.title("Experiment 1 - Complete Text Preprocessing Pipeline")
    st.markdown("**Aim:** lowercase -> punctuation removal -> tokenization -> stopword removal -> stemming -> "
                "lemmatization -> POS tagging, applied to blood-test patient feedback.")
    flags = nltk_status()
    missing = [k for k, v in flags.items() if not v]
    if nltk is None:
        st.warning("NLTK is not installed (`pip install nltk`). Showing fallback tokenization/stopwords only.")
    elif missing:
        st.warning(f"NLTK data not available for: {', '.join(missing)}. Fallbacks are used for those steps "
                   "(regex tokenizer, scikit-learn stopwords, no lemmatization/POS). Run once with internet access "
                   "to let the app download them.")
    samples = [
        "I absolutely loved the care I received during my blood test! The phlebotomist was gentle.",
        "The report was delayed by two days and I'm disappointed.",
        "Excellent hygiene but the billing counter is not accepting UPI payments.",
    ]
    custom = st.text_area("Add your own feedback (optional, one per line)", "")
    reviews = samples + [l for l in custom.splitlines() if l.strip()]
    for i, r in enumerate(reviews, 1):
        with st.expander(f"Feedback {i}: {r[:70]}", expanded=(i == 1)):
            steps = preprocess_steps(r)
            for name, val in steps.items():
                st.markdown(f"**{name}**")
                st.code(str(val), language=None)
    st.info("Notice what stopword removal does to Feedback 3: **'not'** is a stopword, so the negation in "
            "'not accepting UPI payments' is lost. This is why ABSA (Expt 7) handles negation separately.")
    st.subheader("Effect on the whole corpus")
    sw = get_stopwords()
    raw = Counter(w for t in df["feedback"] for w in re.findall(r"[a-z']+", t.lower()))
    clean = Counter(w for t in df["feedback"] for w in re.findall(r"[a-z']+", t.lower()) if w not in sw)
    c1, c2 = st.columns(2)
    c1.write("Top 10 words **before** stopword removal")
    c1.bar_chart(pd.Series(dict(raw.most_common(10))))
    c2.write("Top 10 words **after** stopword removal")
    c2.bar_chart(pd.Series(dict(clean.most_common(10))))
    st.subheader("Post-lab: how do tokenization and stopword removal help a hospital find common service issues?")
    st.markdown("""
Thousands of feedback records are unreadable one by one. **Tokenization** splits each record into words so they can be
counted. **Stopword removal** deletes filler words ("the", "was", "of"), so the counts are dominated by meaningful terms
such as *report, delayed, queue, billing, needle*. Counting these across all records shows which service issues recur
(e.g. report delays or long queues), which departments to prioritise, and gives clean input for the later steps in
this project (classification, clustering, topic modelling). Caution: stopword lists also contain negations like
"not", so for sentiment work they must be kept or handled explicitly.
""")


# =====================================================================================
# PAGE: EXPT 2
# =====================================================================================
@st.cache_resource(show_spinner="Training Word2Vec on the corpus...")
def corpus_word2vec(texts):
    sents = [re.findall(r"[a-z]+", x.lower()) for t in texts for x in re.split(r"[.!?]", t) if x.strip()]
    return Word2Vec(sents, vector_size=50, window=3, min_count=2, sg=1, seed=SEED, epochs=100, workers=1)


def page_expt2(df):
    st.title("Experiment 2 - BoW vs TF-IDF vs Word2Vec")
    sample = ["The phlebotomist has an excellent attitude.", "Report delivery is very late.",
              "Staff attitude is amazing.", "The lab is clean and affordable.", "Report delivery was delayed."]
    st.subheader("Sample dataset")
    docs = [f"D{i + 1}" for i in range(len(sample))]
    st.table(pd.DataFrame({"Document": docs, "Feedback": sample}))

    bow = CountVectorizer()
    Xb = bow.fit_transform(sample)
    vocab = bow.get_feature_names_out()
    st.subheader("Vocabulary")
    st.write(f"{len(vocab)} terms:", list(vocab))
    bow_df = pd.DataFrame(Xb.toarray(), index=docs, columns=vocab)
    st.subheader("Bag of Words matrix (document-term matrix)")
    st.dataframe(bow_df)

    tf = TfidfVectorizer()
    Xt = tf.fit_transform(sample)
    tf_df = pd.DataFrame(Xt.toarray(), index=docs, columns=tf.get_feature_names_out()).round(3)
    st.subheader("TF-IDF matrix")
    st.dataframe(tf_df)

    st.subheader("Most informative words (TF-IDF)")
    c1, c2 = st.columns(2)
    c1.write("Mean TF-IDF, **no** stop-word removal (note noise like 'is', 'the')")
    c1.bar_chart(tf_df.mean().sort_values(ascending=False).head(8))
    tf2 = TfidfVectorizer(stop_words="english")
    X2 = tf2.fit_transform(sample)
    sw_df = pd.DataFrame(X2.toarray(), index=docs, columns=tf2.get_feature_names_out())
    c2.write("Mean TF-IDF, English stop words removed")
    c2.bar_chart(sw_df.mean().sort_values(ascending=False).head(8))
    st.write("Highest-weighted word per document:",
             {d: sw_df.loc[d].idxmax() for d in docs})

    st.subheader("Word2Vec embeddings")
    if gensim is None:
        st.warning("gensim is not installed (`pip install gensim`). Word2Vec section skipped.")
        w2v_doc = None
    else:
        w_small = Word2Vec([re.findall(r"[a-z]+", s.lower()) for s in sample], vector_size=10, window=3,
                           min_count=1, sg=1, seed=SEED, epochs=200, workers=1)
        word = st.selectbox("Word (5-sentence sample, 10 dimensions)", sorted(w_small.wv.index_to_key),
                            index=sorted(w_small.wv.index_to_key).index("report"))
        st.write(np.round(w_small.wv[word], 3))
        st.caption("With only 5 sentences the vectors carry almost no meaning - this just shows the dense format.")
        w2v_doc = np.array([np.mean([w_small.wv[w] for w in re.findall(r"[a-z]+", s.lower())], axis=0) for s in sample])

        w2v = corpus_word2vec(tuple(df["feedback"]))
        st.markdown("**Full corpus Word2Vec (50 dimensions)** - nearest words")
        wc = st.selectbox("Word", [w for w in ["report", "needle", "staff", "billing", "queue", "hygiene", "app"]
                                   if w in w2v.wv], key="w2v_word")
        st.table(pd.DataFrame(w2v.wv.most_similar(wc, topn=6), columns=["Similar word", "Cosine similarity"]).round(3))
        st.warning("Honest note: this corpus is template-generated and small, so neighbours reflect which words "
                   "co-occur in the same phrase rather than true meaning. On real feedback (thousands of free-text "
                   "records) similarities become much more meaningful.")

    st.subheader("Comparison of outputs")
    dens = lambda a: 100 * np.count_nonzero(a) / a.size
    comp = pd.DataFrame({
        "Shape": [str(Xb.shape), str(Xt.shape), str(w2v_doc.shape) if w2v_doc is not None else "n/a"],
        "Value type": ["integer counts", "float weights (0-1)", "dense floats (+/-)"],
        "Non-zero %": [round(dens(Xb.toarray()), 1), round(dens(Xt.toarray()), 1),
                       round(dens(w2v_doc), 1) if w2v_doc is not None else None],
        "Frequency based": ["Yes", "Yes", "No"], "Semantic meaning": ["No", "No", "Yes"],
        "Sparse": ["Yes", "Yes", "No"], "Dense": ["No", "No", "Yes"],
        "Down-weights common words": ["No", "Yes", "N/A (learned)"],
    }, index=["BoW", "TF-IDF", "Word2Vec"]).T
    st.dataframe(comp.astype(str))
    st.subheader("Advantages and limitations")
    st.table(pd.DataFrame({
        "Method": ["BoW", "TF-IDF", "Word2Vec"],
        "Advantages": ["Simple, fast, easy to interpret", "Down-weights common words; better than BoW for retrieval/classification",
                       "Dense, captures similarity between words; low dimension"],
        "Limitations": ["Ignores word order and meaning; sparse; common words dominate",
                        "Still ignores meaning and order; sparse", "Needs lots of text; less interpretable; one vector per word regardless of context"]}))


# =====================================================================================
# PAGE: EXPT 3
# =====================================================================================
@st.cache_data(show_spinner="Training and evaluating classifiers...")
def run_expt3(df, test_size):
    X_tr, X_te, y_tr, y_te = train_test_split(df["feedback"], df["sentiment"], test_size=test_size,
                                              random_state=SEED, stratify=df["sentiment"])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(X_tr), vec.transform(X_te)
    out = {"n_train": len(X_tr), "n_test": len(X_te), "features": Xtr.shape[1], "models": {}}
    for name, m in make_models().items():
        m.fit(Xtr, y_tr)
        p = m.predict(Xte)
        out["models"][name] = dict(
            acc=accuracy_score(y_te, p), prec=precision_score(y_te, p, pos_label="Positive"),
            rec=recall_score(y_te, p, pos_label="Positive"), f1=f1_score(y_te, p, pos_label="Positive"),
            report=pd.DataFrame(classification_report(y_te, p, output_dict=True)).T.round(3),
            cm=confusion_matrix(y_te, p, labels=["Negative", "Positive"]))
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    out["cv"] = {}
    for name, m in make_models().items():
        pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), m)
        s = cross_val_score(pipe, df["feedback"], df["sentiment"], cv=cv, scoring="f1_macro")
        out["cv"][name] = (s.mean(), s.std())
    return out


def page_expt3(df):
    st.title("Experiment 3 - Text Classification: Naive Bayes vs SVM vs Decision Tree")
    st.markdown("Pipeline: dataset -> train/test split -> TF-IDF -> model -> prediction -> evaluation.")
    ts = st.slider("Test size", 0.1, 0.5, 0.30, 0.05)
    r = run_expt3(df, ts)
    st.write(f"Train = **{r['n_train']}**, Test = **{r['n_test']}**, TF-IDF features = **{r['features']}**")
    cols = st.columns(3)
    for col, (name, m) in zip(cols, r["models"].items()):
        with col:
            st.subheader(name)
            st.write(f"Accuracy **{m['acc']:.3f}** | Precision **{m['prec']:.3f}** | Recall **{m['rec']:.3f}** | F1 **{m['f1']:.3f}**")
            st.dataframe(m["report"])
            fig, ax = plt.subplots(figsize=(3.2, 2.8))
            ax.imshow(m["cm"], cmap="Blues")
            for (i, j), v in np.ndenumerate(m["cm"]):
                ax.text(j, i, v, ha="center", va="center", color="black")
            ax.set_xticks([0, 1], ["Neg", "Pos"]); ax.set_yticks([0, 1], ["Neg", "Pos"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix")
            fig_show(fig)
            tn, fp, fn, tp = m["cm"].ravel()
            st.caption(f"Correct {tn + tp}, incorrect {fp + fn} (FP={fp}, FN={fn})")
    comp = pd.DataFrame({n: {"Accuracy": m["acc"], "Precision": m["prec"], "Recall": m["rec"], "F1-score": m["f1"],
                             "5-fold CV macro F1": r["cv"][n][0], "CV std": r["cv"][n][1]}
                         for n, m in r["models"].items()}).T.round(4)
    st.subheader("Comparison table")
    st.dataframe(comp)
    best = comp["F1-score"].idxmax()
    st.success(f"Best on this split by F1: **{best}**. Cross-validation (last columns) shows whether the gap is "
               "real or just a lucky split - if two models are within about one standard deviation, treat them as tied.")
    st.subheader("Predict new, unseen feedback (steps 20-23)")
    default = ("Absolutely loved how gentle the nurse was, totally worth the price.\n"
               "The report came very late and the app kept crashing.\n"
               "Needle prick was painless and the staff were friendly.\n"
               "Billing had hidden charges and the queue was endless.\n"
               "Hygiene was excellent and the results were ready fast.\n"
               "The home pickup was cancelled and nobody called.")
    new = [l for l in st.text_area("One feedback per line", default, height=140).splitlines() if l.strip()]
    if new:
        vec, models = train_full_models(df)
        Xn = vec.transform(new)
        res = pd.DataFrame({"Feedback": new, **{n: m.predict(Xn) for n, m in models.items()}})
        res["All agree"] = res[list(models)].nunique(axis=1).eq(1)
        st.dataframe(res)
        st.caption("Models here are trained on the full corpus. Disagreement shows where models are least reliable "
                   "- often words never seen in training.")


# =====================================================================================
# PAGE: EXPT 4
# =====================================================================================
CLUSTER_DOCS = [
    "The phlebotomist was gentle and the needle prick was painless",
    "The nurse drew my blood quickly with a painless needle",
    "Gentle staff and a painless blood draw with a thin needle",
    "The blood report was delayed by three days",
    "Report delivery was late and results arrived after two days",
    "The billing charges were expensive with hidden fees",
    "The package price was too high and payment charges were hidden",
    "The mobile app crashed during slot booking",
]


def cluster_terms(X, labels, terms, topn=4):
    out = {}
    for c in sorted(set(labels)):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            continue
        mean = np.asarray(X[idx].mean(axis=0)).ravel()
        out[c] = ", ".join(terms[mean.argsort()[::-1][:topn]])
    return out


def page_expt4(df):
    from scipy.cluster.hierarchy import linkage, dendrogram
    st.title("Experiment 4 - Text Clustering: Hierarchical vs DBSCAN")
    st.markdown("Clustering is **unsupervised** - cluster numbers are arbitrary, so we name each cluster by reading "
                "its documents / top terms.")
    st.subheader("Part A - the 8-document problem statement, on patient feedback")
    names = [f"D{i + 1}" for i in range(len(CLUSTER_DOCS))]
    st.table(pd.DataFrame({"Document": names, "Text": CLUSTER_DOCS}))
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(CLUSTER_DOCS)
    st.write("TF-IDF shape:", X.shape)
    dense = X.toarray()
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Hierarchical: number of clusters", 2, 6, 4)
    link_m = c2.selectbox("Linkage", ["ward", "average", "complete"])
    eps = c3.slider("DBSCAN eps (cosine distance)", 0.3, 1.2, 0.8, 0.05)
    minpts = st.slider("DBSCAN min_samples", 2, 4, 2)
    Z = linkage(dense, method=link_m)
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(Z, labels=names, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram"); ax.set_xlabel("Documents"); ax.set_ylabel("Distance")
    fig_show(fig)
    hc = AgglomerativeClustering(n_clusters=k, linkage=link_m).fit_predict(dense)
    db = DBSCAN(eps=eps, min_samples=minpts, metric="cosine").fit_predict(dense)
    res = pd.DataFrame({"Document": names, "Text": CLUSTER_DOCS, "Hierarchical": hc, "DBSCAN": db})
    st.dataframe(res)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Hierarchical interpretation**")
        for c in sorted(set(hc)):
            st.write(f"Cluster {c}: {res[res.Hierarchical == c].Document.tolist()}")
    with c2:
        st.markdown("**DBSCAN interpretation**")
        for c in sorted(set(db)):
            label = "Noise (-1)" if c == -1 else f"Cluster {c}"
            st.write(f"{label}: {res[res.DBSCAN == c].Document.tolist()}")
    st.caption("Expected themes: sample collection (D1-D3), report delay (D4-D5), billing (D6-D7), and D8 (app) as a "
               "likely outlier/noise point. Results depend on eps, min_samples, linkage and preprocessing.")
    st.info("The lab sheet's example uses eps=1.2 with cosine distance. Cosine distance between non-negative TF-IDF "
            "vectors never exceeds 1, so eps=1.2 would put every document in one cluster - values around 0.7-0.85 separate the themes here, 0.9 starts to merge themes and 1.0 or more merges everything.")

    st.subheader("Part B - cluster a sample of the full feedback corpus")
    n = st.slider("Documents to sample", 30, 150, 60, 10)
    sub = df.sample(n, random_state=SEED).reset_index(drop=True)
    vec2 = TfidfVectorizer(stop_words="english", min_df=2)
    X2 = vec2.fit_transform(sub["feedback"])
    terms = vec2.get_feature_names_out()
    k2 = st.slider("Hierarchical clusters (corpus)", 3, 10, 6)
    hc2 = AgglomerativeClustering(n_clusters=k2, linkage="ward").fit_predict(X2.toarray())
    st.write(f"Silhouette score (hierarchical): **{silhouette_score(X2.toarray(), hc2):.3f}**")
    tt = cluster_terms(X2, hc2, terms)
    sizes = pd.Series(hc2).value_counts()
    st.table(pd.DataFrame({"Cluster": list(tt), "Size": [int(sizes[c]) for c in tt], "Top terms": list(tt.values()),
                           "Most common main aspect": [sub[hc2 == c].main_aspect.mode().iloc[0] for c in tt]}))
    c1, c2 = st.columns(2)
    eps2 = c1.slider("DBSCAN eps (corpus)", 0.3, 1.0, 0.6, 0.05)
    mp2 = c2.slider("DBSCAN min_samples (corpus)", 2, 8, 3)
    db2 = DBSCAN(eps=eps2, min_samples=mp2, metric="cosine").fit_predict(X2.toarray())
    st.write(f"DBSCAN (eps={eps2}, min_samples={mp2}): {len(set(db2)) - (1 if -1 in db2 else 0)} clusters, "
             f"{int((db2 == -1).sum())} noise points out of {n}.")
    st.caption("Feedback usually mentions 2-3 aspects at once, so documents overlap heavily and silhouette scores stay "
               "low; DBSCAN is very sensitive to eps - around 0.6 it finds a few dense groups plus noise, at 0.7 and above everything collapses into one cluster, and at 0.4 almost everything is noise.")
    st.subheader("Comparison")
    st.table(pd.DataFrame({
        "Aspect": ["Principle", "Number of clusters", "Dendrogram", "Noise detection", "Parameters", "Useful for"],
        "Hierarchical": ["Repeatedly merge closest documents/clusters", "Chosen later by cutting the dendrogram",
                         "Yes - full merge order", "No - every document is forced into a cluster",
                         "Linkage, distance metric", "Small datasets, seeing relationships at every level"],
        "DBSCAN": ["Group dense neighbourhoods, mark isolated points as noise", "Found automatically from density",
                   "No", "Yes - label -1", "eps, min_samples", "Data with outliers, unknown cluster count"]}))
    st.markdown("**Use-case (KJS-HLC-01 style):** clustering groups unlabelled feedback into recurring themes for "
                "organisation and exploration - it does not diagnose anything.")


# =====================================================================================
# PAGE: EXPT 5
# =====================================================================================
@st.cache_data(show_spinner="Fitting LDA for each iteration count...")
def lda_iterations(texts, n_topics, max_iter, word_idx, doc_idx):
    cv = CountVectorizer(stop_words="english", max_features=500, min_df=2)
    Xc = cv.fit_transform(texts)
    wp, dp = [], []
    for i in range(1, max_iter + 1):
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=i, random_state=SEED)
        lda.fit(Xc)
        tw = lda.components_ / lda.components_.sum(axis=1)[:, None]
        wp.append(tw[:, word_idx])
        dp.append(lda.transform(Xc)[doc_idx])
    return np.array(wp), np.array(dp)


def page_expt5(df):
    st.title("Experiment 5 - Topic Modelling: LDA and NMF")
    texts = tuple(df["feedback"])
    c1, c2 = st.columns(2)
    n_topics = c1.slider("Number of topics", 3, 10, 6)
    max_iter = c2.slider("LDA iterations to track", 3, 15, 10)

    cv = CountVectorizer(stop_words="english", max_features=500, min_df=2)
    Xc = cv.fit_transform(texts)
    vocab_c = cv.get_feature_names_out()
    st.subheader("LDA")
    st.markdown("Each word starts with a random topic and is re-assigned every iteration based on the other words in "
                "its document. Below: how a chosen word's topic probabilities and a chosen document's topic mix evolve.")
    c1, c2 = st.columns(2)
    top_words = [vocab_c[i] for i in np.asarray(Xc.sum(axis=0)).ravel().argsort()[::-1][:30]]
    word = c1.selectbox("Word to track", top_words)
    doc_idx = c2.number_input("Document index to track", 0, len(df) - 1, 0)
    wp, dp = lda_iterations(texts, n_topics, max_iter, int(list(vocab_c).index(word)), int(doc_idx))
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for t in range(n_topics):
            ax.plot(range(1, max_iter + 1), wp[:, t], marker="o", label=f"Topic {t}")
        ax.set_title(f"P(word='{word}' | topic) across iterations"); ax.set_xlabel("Iterations"); ax.set_ylabel("Probability")
        ax.legend(fontsize=6); fig_show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for t in range(n_topics):
            ax.plot(range(1, max_iter + 1), dp[:, t], marker="o", label=f"Topic {t}")
        ax.set_title(f"Topic mix of document {int(doc_idx)} across iterations"); ax.set_xlabel("Iterations")
        ax.set_ylabel("Probability"); ax.legend(fontsize=6); fig_show(fig)
    st.caption("Document being tracked: " + df["feedback"].iloc[int(doc_idx)])

    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=max_iter, random_state=SEED).fit(Xc)
    lda_top = [", ".join(vocab_c[c.argsort()[::-1][:8]]) for c in lda.components_]
    st.write("**LDA topics (top words)**")
    st.table(pd.DataFrame({"Topic": [f"Topic {i}" for i in range(n_topics)], "Top words": lda_top}))

    st.subheader("NMF")
    tf = TfidfVectorizer(stop_words="english", max_features=500, min_df=2)
    Xt = tf.fit_transform(texts)
    terms = tf.get_feature_names_out()
    top_cols = np.asarray(Xt.sum(axis=0)).ravel().argsort()[::-1][:20]
    st.write(f"**Input matrix (TF-IDF), shape {Xt.shape}** - first 12 rows x 20 highest-weight terms")
    st.dataframe(pd.DataFrame(Xt[:12][:, top_cols].toarray(), columns=terms[top_cols]).round(3))
    nmf = NMF(n_components=n_topics, random_state=SEED, init="nndsvda", max_iter=500)
    W = nmf.fit_transform(Xt)
    H = nmf.components_
    st.write(f"**Topic-Term matrix H**, shape {H.shape} (rows = topics, columns = terms) - top 12 terms shown")
    hcols = np.argsort(H.sum(axis=0))[::-1][:12]
    st.dataframe(pd.DataFrame(H[:, hcols], columns=terms[hcols], index=[f"Topic {i + 1}" for i in range(n_topics)]).round(3))
    st.write(f"**Document-Topic matrix W**, shape {W.shape} - first 12 documents")
    st.dataframe(pd.DataFrame(W[:12], columns=[f"Topic {i + 1}" for i in range(n_topics)]).round(3))
    err = np.linalg.norm(Xt.toarray() - W @ H) / np.linalg.norm(Xt.toarray())
    st.write(f"Check: W x H reproduces the input matrix up to a relative error of **{err:.3f}** "
             "(NMF gives an approximation, not an exact product).")
    nmf_top = [", ".join(terms[t.argsort()[::-1][:8]]) for t in H]
    st.table(pd.DataFrame({"Topic": [f"Topic {i + 1}" for i in range(n_topics)], "Top terms": nmf_top}))

    st.subheader("Do the topics line up with real departments?")
    ct = pd.crosstab(df["department"], W.argmax(axis=1) + 1)
    ct.columns = [f"NMF topic {c}" for c in ct.columns]
    st.dataframe(ct)
    st.caption("Each feedback's dominant NMF topic versus the department of its main aspect. Because feedback often "
               "mentions 2-3 aspects, topics will mix departments - that is expected and is the point of a "
               "probabilistic/soft topic view.")
    st.markdown("**Observations:** LDA's word and document probabilities move a lot in the first few iterations and "
                "then settle; NMF yields non-negative, easy-to-read topic-term weights directly from TF-IDF. Both "
                "recover themes such as report delays, billing, collection experience and digital services.")


# =====================================================================================
# PAGE: EXPT 6
# =====================================================================================
IR_DOCS = [
    ("L1", "Mumbai", "05-02-2026", "CBC", "On 5 February 2026 a patient at the Mumbai branch complained that the CBC report was delayed by two days. "
     "The front desk apologised and the report was emailed late in the evening. The patient asked for faster turnaround for routine tests."),
    ("L2", "Pune", "14-03-2026", "Lipid profile", "On 14 March 2026 the Pune branch received praise for a painless blood draw during a lipid profile test. "
     "The phlebotomist found the vein on the first attempt and the patient said the fasting instructions were clear."),
    ("L3", "Nagpur", "22-04-2026", "HbA1c", "At the Nagpur centre on 22 April 2026 a patient waited ninety minutes in a crowded queue for an HbA1c test. "
     "Several patients left without testing and asked for an online booking system with fixed time slots."),
    ("L4", "Thane", "09-05-2026", "Thyroid profile", "A thyroid profile in Thane on 9 May 2026 was billed with hidden charges that were not shown on the package page. "
     "The billing team corrected the invoice after a complaint and promised transparent pricing."),
    ("L5", "Nashik", "30-05-2026", "Vitamin D", "On 30 May 2026 a home collection pickup in Nashik for a Vitamin D test was cancelled without notice. "
     "The patient waited at home for three hours and then booked another slot through the mobile app."),
    ("L6", "Navi Mumbai", "18-06-2026", "Liver function", "In Navi Mumbai on 18 June 2026 a liver function report showed a wrong patient name and an incorrect value. "
     "A retest was arranged free of cost and the quality team started an error review."),
    ("L7", "Mumbai", "11-07-2026", "Kidney function", "On 11 July 2026 the Mumbai branch collection room was praised for hygiene, with sterile equipment and fresh gloves. "
     "Patients doing kidney function tests described the nurse as caring and professional."),
    ("L8", "Pune", "25-07-2026", "Fasting blood sugar", "A patient in Pune on 25 July 2026 found the fasting instructions confusing and arrived after eating breakfast. "
     "The fasting blood sugar test had to be rescheduled and the lab agreed to send clearer preparation guidance."),
    ("L9", "Thane", "03-08-2026", "CBC", "On 3 August 2026 the online portal of the Thane branch crashed repeatedly while patients tried to download CBC reports. "
     "Support staff emailed the reports manually and the technical team investigated the website login problem."),
    ("L10", "Nagpur", "19-08-2026", "Lipid profile", "The Nagpur centre on 19 August 2026 delivered lipid profile results within four hours, which patients called fast and reliable. "
     "The receptionist guided elderly patients politely and the waiting time stayed short."),
    ("L11", "Nashik", "02-09-2026", "HbA1c", "On 2 September 2026 a patient in Nashik reported a painful needle prick and a large bruise after an HbA1c draw. "
     "The technician had made multiple attempts and the branch manager offered a follow up with an experienced phlebotomist."),
    ("L12", "Navi Mumbai", "16-09-2026", "Thyroid profile", "In Navi Mumbai on 16 September 2026 the thyroid profile package was described as affordable and good value. "
     "Patients liked the transparent billing and the reminder messages about fasting and report collection."),
]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
CITIES = ["Navi Mumbai", "Mumbai", "Pune", "Nagpur", "Nashik", "Thane", "Delhi", "Bengaluru", "Chennai", "Kolkata", "Hyderabad"]


def extract_entities(query: str):
    nlp = load_spacy()
    if nlp is not None:
        return [(e.text, e.label_) for e in nlp(query).ents], "spaCy en_core_web_sm"
    ents = []
    for c in CITIES:
        if re.search(rf"\b{c}\b", query, flags=re.I):
            ents.append((c, "GPE (rule-based)"))
            query = re.sub(rf"\b{c}\b", " ", query, flags=re.I)
    for m in re.finditer(rf"\b(\d{{1,2}}\s+)?({MONTHS})(\s+\d{{4}})?\b", query, flags=re.I):
        ents.append((m.group(0).strip(), "DATE (rule-based)"))
    return ents, "rule-based fallback (install spaCy + en_core_web_sm for statistical NER)"


def page_expt6(df):
    st.title("Experiment 6 - Information Retrieval: TF-IDF + NER + Cosine Similarity")
    ir = load_ir_docs()
    st.subheader(f"Corpus - {len(ir)} branch feedback reports (from {IR_CSV.name})")
    st.dataframe(ir)
    docs_raw = ir["Paragraph"]
    docs = docs_raw.apply(simple_clean)
    with st.expander("Preprocessed documents"):
        for i, d in enumerate(docs, 1):
            st.write(f"**Document {i}:** {d}")
    vec = TfidfVectorizer(stop_words="english")
    Xd = vec.fit_transform(docs)
    st.write("TF-IDF matrix shape:", Xd.shape)
    query = st.text_input("Enter query", "Delayed CBC report at the Mumbai branch in February 2026")
    if not query.strip():
        return
    ents, engine = extract_entities(query)
    st.write(f"**Named entities in query** (engine: {engine})")
    if ents:
        st.table(pd.DataFrame(ents, columns=["Entity", "Label"]))
    else:
        st.write("No named entities detected")
    ctx = " ".join(e for e, _ in ents)
    contextual = query + " " + ctx if ctx else query
    processed = simple_clean(contextual)
    st.write("**Original query:**", query)
    st.write("**Contextual query (query + entities):**", contextual)
    st.write("**Preprocessed query:**", processed)
    sims = cosine_similarity(vec.transform([processed]), Xd).ravel()
    res = ir.assign(**{"Cosine Similarity": sims}).sort_values("Cosine Similarity", ascending=False).reset_index(drop=True)
    res.index = res.index + 1
    res.index.name = "Rank"
    st.subheader("Ranked documents")
    st.dataframe(res[["Document_ID", "Location", "Date", "Test", "Cosine Similarity"]].round(4))
    st.bar_chart(res.set_index("Document_ID")["Cosine Similarity"])
    best = res.iloc[0]
    if best["Cosine Similarity"] == 0:
        st.warning("No document shares any term with this query.")
    else:
        st.success(f"Most relevant: **{best['Document_ID']}** ({best['Location']}, {best['Date']}) - "
                   f"similarity {best['Cosine Similarity']:.3f}")
        st.write(best["Paragraph"])
    st.markdown("**Why NER helps:** repeating the entities (city, date) in the query boosts documents that match the "
                "location/date context, instead of matching only the generic words like 'report' or 'delayed'.")


# =====================================================================================
# PAGE: EXPT 7
# =====================================================================================
ABSA_SAMPLE = [
    ("R1", "The phlebotomist was polite and the report was delivered quickly."),
    ("R2", "The waiting time was very long but the needle prick was painless."),
    ("R3", "The fasting instructions were confusing and the billing was expensive."),
    ("R4", "The home pickup was punctual and the staff were professional."),
    ("R5", "The booking app was easy to use but the report was delayed."),
    ("R6", "The collection room was clean and the staff were helpful."),
    ("R7", "The laboratory staff were helpful but the waiting time was very long."),
]


def page_expt7(df):
    st.title("Experiment 7 - Aspect-Based Sentiment Analysis (ABSA)")
    st.markdown("Pipeline: review -> preprocessing -> aspect identification -> sentiment identification -> "
                "aspect-sentiment pair.")
    st.subheader("Sample reviews")
    rows = []
    for rid, text in ABSA_SAMPLE:
        for a, s in absa_extract(text):
            rows.append((rid, text, a, s))
    obs = pd.DataFrame(rows, columns=["Review ID", "Review", "Aspect", "Sentiment"])
    for rid, text in ABSA_SAMPLE:
        st.markdown(f"**{rid}:** {text}")
        for a, s in absa_extract(text):
            st.write(f"&nbsp;&nbsp;&nbsp;{a} → **{s}**")
    st.subheader("Structured output (observation table)")
    st.dataframe(obs)
    st.subheader("Aspect-wise summary (crosstab)")
    st.dataframe(pd.crosstab(obs["Aspect"], obs["Sentiment"]))
    st.subheader("Try your own review")
    own = st.text_input("Review", "The laboratory staff were helpful but the waiting time was not short.")
    st.table(pd.DataFrame(absa_extract(own), columns=["Aspect", "Sentiment"]))

    st.subheader("Whole corpus")
    ab = absa_corpus(df)
    ct = pd.crosstab(ab["Aspect"], ab["Sentiment"])
    c1, c2 = st.columns(2)
    c1.dataframe(ct)
    c2.bar_chart(ct)
    pct = (ab[ab.Sentiment != "Neutral"].groupby("Aspect").Sentiment.apply(lambda s: (s == "Negative").mean())
           .sort_values(ascending=False))
    st.write("**Share of negative opinions by aspect** (priority list for the lab):")
    st.bar_chart(pct)

    truth = {(r.feedback_id, p.split(":")[0], p.split(":")[1]) for r in df.itertuples() for p in r.truth_pairs.split(";")}
    pred = set(ab.itertuples(index=False, name=None))
    tp = len(truth & pred)
    precision, recall = tp / max(len(pred), 1), tp / max(len(truth), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    st.subheader("How good is the rule-based ABSA?")
    st.write(f"Compared with the aspect-sentiment pairs used to generate the data: precision **{precision:.3f}**, "
             f"recall **{recall:.3f}**, F1 **{f1:.3f}** over {len(truth)} true pairs. "
             "Because the data is template-generated, this is an optimistic upper bound - real feedback is messier.")

    st.subheader("Document-level vs aspect-level sentiment")
    doc_pred = df["feedback"].apply(doc_sentiment_lexicon)
    acc = (doc_pred == df["sentiment"]).mean()
    st.write(f"Document-level lexicon classifier (counts positive/negative words, no negation, no aspects): "
             f"accuracy **{acc:.1%}** against the overall label.")
    c1, c2 = st.columns(2)
    c1.dataframe(pd.crosstab(df["sentiment"], doc_pred, rownames=["Actual"], colnames=["Predicted"]))
    c2.bar_chart(doc_pred.value_counts())
    st.table(pd.DataFrame({
        "Document-level sentiment": ["One sentiment per review", "Simpler", "Can lose information", '"Overall Positive"'],
        "Aspect-level sentiment": ["Sentiment per aspect", "More detailed", "Preserves aspect opinions",
                                   "Staff: Positive, Report: Negative"]}))

    st.subheader("Feedback routing to departments (keyword classifier)")
    routed = df["feedback"].apply(route_department)
    racc = (routed == df["department"]).mean()
    involved = df["truth_pairs"].apply(lambda t: {ASPECT_DEPT[p.split(":")[0]] for p in t.split(";")})
    hit = np.mean([r in inv for r, inv in zip(routed, involved)])
    st.write(f"Exact match with the department of each review's **main aspect**: **{racc:.1%}**. "
             f"Predicted department is **one of the departments the review actually involves**: **{hit:.1%}**. "
             "Reviews mentioning several aspects legitimately belong to several departments, so the second number is "
             "the fairer one for a keyword router.")
    st.bar_chart(routed.value_counts())
    st.dataframe(pd.crosstab(df["department"], routed, rownames=["Actual"], colnames=["Predicted"]))
    st.markdown("""
**Use case (KJS-HLC-01):** "The laboratory staff were helpful but the waiting time was very long" -> Staff: Positive,
Waiting time: Negative. A lab manager can fix the queue without changing a team that is already doing well.
""")


# =====================================================================================
# PAGE: EXPT 8
# =====================================================================================
DEFAULT_DOC = """The pathology laboratory network received 412 patient feedback records this week.
Report turnaround was the most frequent complaint, with many CBC and HbA1c reports delayed by more than two days.
The Pune and Thane branches recorded the longest waiting times during morning fasting hours.
Patients praised the phlebotomists for gentle sample collection and clean collection rooms.
Several patients reported hidden charges in package billing and asked for transparent pricing.
The online portal crashed twice during report downloads and the support team sent reports manually.
Home collection slots were missed at the Nashik branch on three days.
Management has planned extra morning staff, a portal fix and a clearer fasting instruction message."""

DEFAULT_REF = ("Report delays and long morning queues were the top patient complaints this week, "
               "while sample collection staff and hygiene were praised; billing transparency, portal crashes and "
               "missed home collection slots need action.")


def lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_scores(reference, summary):
    """ROUGE-1/2/L F1 (uses rouge_score if installed, else a built-in implementation)."""
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(reference, summary)
        return {k: (v.precision, v.recall, v.fmeasure) for k, v in sc.items()}, "rouge-score package"
    except Exception:
        tok = lambda t: re.findall(r"[a-z0-9]+", t.lower())
        r, s = tok(reference), tok(summary)

        def prf(overlap, rn, sn):
            p, rc = overlap / max(sn, 1), overlap / max(rn, 1)
            return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)

        def ngr(t, n):
            return Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))

        out = {}
        for key, n in (("rouge1", 1), ("rouge2", 2)):
            rg, sg = ngr(r, n), ngr(s, n)
            out[key] = prf(sum((rg & sg).values()), sum(rg.values()), sum(sg.values()))
        out["rougeL"] = prf(lcs_len(r, s), len(r), len(s))
        return out, "built-in implementation (pip install rouge-score for the official one)"


@st.cache_resource(show_spinner="Loading transformer model (first time downloads it)...")
def load_bart(name):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tok, model.to(device), device


def bart_summary(name, text, max_length, min_length):
    tok, model, device = load_bart(name)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    ids = model.generate(**inputs, max_length=max_length, min_length=min_length, num_beams=4, do_sample=False,
                         early_stopping=True)
    return tok.decode(ids[0], skip_special_tokens=True)


def page_expt8(df):
    st.title("Experiment 8 - Summarization: TextRank vs Transformers")
    if nx is None:
        st.error("networkx is required for TextRank: `pip install networkx`")
        return
    text = st.text_area("Document (weekly feedback digest) - edit freely", DEFAULT_DOC, height=220)
    sentences = sent_split(text)
    if len(sentences) < 2:
        st.warning("Enter at least two sentences.")
        return
    st.subheader("1. Sentence segmentation")
    st.table(pd.DataFrame({"Sentence": [f"S{i + 1}" for i in range(len(sentences))], "Text": sentences}))
    Xs = TfidfVectorizer().fit_transform(sentences)
    st.write("TF-IDF matrix shape:", Xs.shape)
    sim = cosine_similarity(Xs)
    st.subheader("2. Cosine similarity matrix")
    st.dataframe(pd.DataFrame(np.round(sim, 2), index=[f"S{i + 1}" for i in range(len(sentences))],
                              columns=[f"S{i + 1}" for i in range(len(sentences))]))
    G = nx.from_numpy_array(sim)
    st.write(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    scores = nx.pagerank(G)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    st.subheader("3. TextRank scores")
    st.table(pd.DataFrame({"Sentence": [f"S{i + 1}" for i, _ in ranked], "Score": [round(s, 4) for _, s in ranked],
                           "Text": [sentences[i] for i, _ in ranked]}))
    st.bar_chart(pd.Series({f"S{i + 1}": s for i, s in ranked}))
    top_n = st.slider("Sentences in extractive summary", 1, max(1, len(sentences) - 1), min(3, len(sentences) - 1))
    sel = sorted(i for i, _ in ranked[:top_n])
    tr_summary = " ".join(sentences[i] for i in sel)
    st.subheader("4. TextRank (extractive) summary")
    st.success(tr_summary)

    st.subheader("5. Transformer (abstractive) summary")
    model_name = st.selectbox("Model", ["facebook/bart-large-cnn", "sshleifer/distilbart-cnn-12-6"])
    st.caption("Running a transformer downloads the model (about 1.2-1.6 GB) and needs `pip install transformers torch`. "
               "It is optional - everything else works without it.")
    c1, c2 = st.columns(2)
    max_len = c1.slider("max_length", 30, 120, 60)
    min_len = c2.slider("min_length", 10, 60, 25)
    if st.button("Generate transformer summary"):
        try:
            st.session_state["bart"] = bart_summary(model_name, text, max_len, min_len)
        except Exception as e:
            st.session_state["bart"] = None
            st.error(f"Could not run the transformer model: {type(e).__name__}: {str(e)[:200]}")
    tf_summary = st.session_state.get("bart")
    if tf_summary:
        st.info(tf_summary)

    st.subheader("6. Comparison")
    ow, tw = len(text.split()), len(tr_summary.split())
    rows = {"Original": ow, "TextRank": tw}
    if tf_summary:
        rows["Transformer"] = len(tf_summary.split())
    wc = pd.DataFrame({"Words": rows})
    wc["Compression (words / original)"] = (wc["Words"] / ow).round(2)
    st.dataframe(wc)
    st.bar_chart(wc["Words"])
    orig = set(sentences)
    st.write("TextRank contains only original sentences:", all(s in orig for s in sent_split(tr_summary)))
    if tf_summary:
        st.write("Transformer contains only original sentences:", all(s in orig for s in sent_split(tf_summary)))

    st.subheader("7. ROUGE evaluation")
    ref = st.text_area("Reference summary (the 'ideal' summary written by a human)", DEFAULT_REF, height=80)
    cands = {"TextRank": tr_summary}
    if tf_summary:
        cands["Transformer"] = tf_summary
    table, engine = {}, ""
    for name, s in cands.items():
        sc, engine = rouge_scores(ref, s)
        for k, (p, r, f) in sc.items():
            table[(name, k.upper().replace("ROUGE", "ROUGE-"))] = {"Precision": p, "Recall": r, "F1": f}
    st.dataframe(pd.DataFrame(table).T.round(3))
    st.caption(f"ROUGE computed with: {engine}. Scores depend heavily on the reference you write.")
    st.table(pd.DataFrame({
        "": ["Type", "Training needed", "Output", "Semantic understanding", "Implementation"],
        "TextRank": ["Extractive", "No", "Existing sentences", "Limited", "Simple"],
        "Transformer (BART)": ["Abstractive", "Pretrained model", "Newly generated sentences", "Stronger",
                               "Heavier / needs more compute"]}))
    st.markdown("""
**Healthcare precautions (KJS-HLC-01):** keep patient-specific values, units, test names and abnormal flags unchanged -
extractive summaries preserve them because they copy sentences, while abstractive models can paraphrase or omit details.
Never use an AI summary alone for clinical decisions; always keep the original report available and have a clinician
verify. Remove personal identifiers before sending text to any external model.
""")


# =====================================================================================
# MAIN
# =====================================================================================
PAGES = {
    "Dashboard": page_dashboard,
    "Live Feedback Analyzer": page_live,
    "Expt 1 - Text Preprocessing": page_expt1,
    "Expt 2 - BoW / TF-IDF / Word2Vec": page_expt2,
    "Expt 3 - Text Classification": page_expt3,
    "Expt 4 - Text Clustering": page_expt4,
    "Expt 5 - Topic Modelling": page_expt5,
    "Expt 6 - Information Retrieval": page_expt6,
    "Expt 7 - Aspect-Based Sentiment": page_expt7,
    "Expt 8 - Summarization": page_expt8,
}


def main():
    st.sidebar.title("🩸 Blood Test Feedback Analytics")
    page = st.sidebar.radio("Go to", list(PAGES))
    up = st.sidebar.file_uploader("Use your own feedback CSV (optional)", type="csv",
                                  help="Needs the same columns as patient_feedback_dataset.csv: " + ", ".join(REQUIRED_COLS))
    df, source = load_dataset(up)
    st.sidebar.caption(f"Dataset: {source}. Synthetic data, no real patients.")
    with st.sidebar.expander("Environment"):
        fl = nltk_status()
        st.write({"nltk": nltk is not None, **{f"nltk:{k}": v for k, v in fl.items()},
                  "gensim": gensim is not None, "networkx": nx is not None, "spaCy model": load_spacy() is not None})
    with st.sidebar.expander("Download dataset"):
        st.download_button("patient_feedback_dataset.csv", df.drop(columns=["month"]).to_csv(index=False).encode(),
                           "patient_feedback_dataset.csv", "text/csv")
    PAGES[page](df)


main()
