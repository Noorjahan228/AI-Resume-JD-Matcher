import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Page Setup
st.set_page_config(
    page_title="Resume & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Human-Centric Minimal Professional CSS
st.markdown("""
    <style>
    /* Dark Slate Background for High Contrast */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Clean Cards with Subtle Borders */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .card h3 {
        color: #38bdf8;
        margin-bottom: 8px;
    }
    
    .card p {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Score Badge */
    .score-card {
        background: #064e3b;
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Clean Primary Button */
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: #ffffff;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
    }

    /* Tab Headers */
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to extract text from PDF
def extract_pdf_text(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Resume Matcher", "How It Works"])

# ==============================================================================
# TAB 1: OVERVIEW (Humanized Content)
# ==============================================================================
with tab1:
    st.title("Smart Resume & Job Description Matcher 🎯")
    st.write("A practical tool built to help job seekers optimize their resumes for Applicant Tracking Systems (ATS) before applying.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>📊 Match Scoring</h3>
            <p>Compares keywords and term relevance between your resume and target job listings to estimate ATS alignment.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>🔍 Skill Gap Detection</h3>
            <p>Highlights important skills or keywords present in the job posting that are currently missing from your resume.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h3>📑 PDF & Text Support</h3>
            <p>Easily upload your PDF resume directly or paste text content to get instant feedback.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: RESUME MATCHER (Core Functionality)
# ==============================================================================
with tab2:
    st.subheader("Compare Resume with Job Description")
    st.write("Paste the job details and your resume to check compatibility.")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Job Description")
        jd_text = st.text_area("Paste job requirements here:", height=220, placeholder="Example: Looking for a Python Developer experienced with SQL, Data Analysis, and APIs...")

    with col2:
        st.markdown("##### Your Resume")
        upload_type = st.radio("Choose input method:", ["Upload PDF Resume", "Paste Plain Text"], horizontal=True)
        
        resume_text = ""
        if upload_type == "Upload PDF Resume":
            uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])
            if uploaded_file:
                resume_text = extract_pdf_text(uploaded_file)
                st.success("PDF loaded successfully.")
        else:
            resume_text = st.text_area("Paste resume text here:", height=150, placeholder="Paste your resume content...")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Check Match Score"):
        if not jd_text.strip() or not resume_text.strip():
            st.warning("Please provide both Job Description and Resume details.")
        else:
            # Algorithm Processing
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([jd_text, resume_text])
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            match_percentage = round(similarity_score * 100, 2)
            
            feature_names = vectorizer.get_feature_names_out()
            jd_vector = tfidf_matrix[0].toarray()[0]
            resume_vector = tfidf_matrix[1].toarray()[0]
            
            missing_keywords = [
                feature_names[i] for i in range(len(feature_names)) 
                if jd_vector[i] > 0 and resume_vector[i] == 0
            ]

            # Results View
            st.markdown("---")
            st.markdown("### Match Analysis Output")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                <div class='score-card'>
                    <h1 style='margin:0; color: #34d399; font-size: 3rem;'>{match_percentage}%</h1>
                    <p style='margin:0; color: #f8fafc;'>Estimated Relevance Score</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("##### Missing Keywords")
                if missing_keywords:
                    top_missing = missing_keywords[:15]
                    st.write("Consider adding relevant terms from this list to improve resume match:")
                    st.write(", ".join([f"`{kw}`" for kw in top_missing]))
                else:
                    st.success("Your resume covers all major keywords mentioned in the job description.")

# ==============================================================================
# TAB 3: HOW IT WORKS (Simple & Clean Explanation)
# ==============================================================================
with tab3:
    st.subheader("How This App Works")
    st.write("A simple step-by-step breakdown of the text-matching logic used under the hood.")
    
    st.markdown("""
    <div class='card'>
        <h3>1. Text Ingestion</h3>
        <p>The app parses your uploaded PDF using <code>PyPDF2</code> or reads pasted text directly to clean and prepare the content for analysis.</p>
    </div>
    
    <div class='card'>
        <h3>2. TF-IDF Vectorization</h3>
        <p>Using <code>Scikit-Learn</code>, both texts are converted into mathematical frequency vectors after removing common English stop-words (e.g., 'the', 'is', 'at').</p>
    </div>

    <div class='card'>
        <h3>3. Cosine Similarity Match</h3>
        <p>The app calculates the mathematical angle between the two text vectors using Cosine Similarity to determine how closely the resume aligns with the job description.</p>
    </div>

    <div class='card'>
        <h3>4. Keyword Gap Identification</h3>
        <p>Keywords present in the job description vector but missing in the resume vector are filtered out and displayed as recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
