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

# High-Contrast CSS Fix (Forces White Text Everywhere)
st.markdown("""
    <style>
    /* Dark Slate Background */
    .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Make ALL Text & Labels Forced White for High Visibility */
    p, span, label, div, h1, h2, h3, h4, h5, h6, .stMarkdown {
        color: #f8fafc !important;
    }

    /* Subtitle & Muted Descriptions */
    .sub-text {
        color: #94a3b8 !important;
        font-size: 0.95rem;
    }

    /* Input Textboxes Styling */
    textarea, input {
        background-color: #1e293b !important;
        color: #ffffff !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }

    /* Card Containers */
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 16px;
    }

    /* Result Metric Card */
    .score-card {
        background: #064e3b;
        border: 1px solid #059669;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* Primary Button */
    .stButton>button {
        width: 100%;
        background-color: #2563eb !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border: none !important;
    }
    .stButton>button:hover {
        background-color: #1d4ed8 !important;
    }

    /* Navigation Tabs High Visibility */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
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
# TAB 1: OVERVIEW
# ==============================================================================
with tab1:
    st.title("Smart Resume & Job Description Matcher 🎯")
    st.markdown("<p class='sub-text'>A practical tool built to help job seekers optimize their resumes for Applicant Tracking Systems (ATS) before applying.</p>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #38bdf8 !important;'>📊 Match Scoring</h3>
            <p class='sub-text'>Compares keywords and term relevance between your resume and target job listings to estimate ATS alignment.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #38bdf8 !important;'>🔍 Skill Gap Detection</h3>
            <p class='sub-text'>Highlights important skills or keywords present in the job posting that are currently missing from your resume.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #38bdf8 !important;'>📑 PDF & Text Support</h3>
            <p class='sub-text'>Easily upload your PDF resume directly or paste text content to get instant feedback.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: RESUME MATCHER
# ==============================================================================
with tab2:
    st.subheader("Compare Resume with Job Description")
    st.markdown("<p class='sub-text'>Paste the job details and your resume to check compatibility.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Job Description")
        jd_text = st.text_area("Paste job requirements here:", height=220, placeholder="Example: Looking for a Python Developer experienced with SQL, Data Analysis, and APIs...")

    with col2:
        st.markdown("### Your Resume")
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
                    <h1 style='margin:0; color: #34d399 !important; font-size: 3rem;'>{match_percentage}%</h1>
                    <p style='margin:0; color: #f8fafc !important;'>Estimated Relevance Score</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("### Missing Keywords")
                if missing_keywords:
                    top_missing = missing_keywords[:15]
                    st.write("Consider adding relevant terms from this list to improve resume match:")
                    st.write(", ".join([f"`{kw}`" for kw in top_missing]))
                else:
                    st.success("Your resume covers all major keywords mentioned in the job description.")

# ==============================================================================
# TAB 3: HOW IT WORKS
# ==============================================================================
with tab3:
    st.subheader("How This App Works")
    st.markdown("<p class='sub-text'>A simple step-by-step breakdown of the text-matching logic used under the hood.</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='card'>
        <h3 style='color: #38bdf8 !important;'>1. Text Ingestion</h3>
        <p class='sub-text'>The app parses your uploaded PDF using <code>PyPDF2</code> or reads pasted text directly to clean and prepare the content for analysis.</p>
    </div>
    
    <div class='card'>
        <h3 style='color: #38bdf8 !important;'>2. TF-IDF Vectorization</h3>
        <p class='sub-text'>Using <code>Scikit-Learn</code>, both texts are converted into mathematical frequency vectors after removing common English stop-words (e.g., 'the', 'is', 'at').</p>
    </div>

    <div class='card'>
        <h3 style='color: #38bdf8 !important;'>3. Cosine Similarity Match</h3>
        <p class='sub-text'>The app calculates the mathematical angle between the two text vectors using Cosine Similarity to determine how closely the resume aligns with the job description.</p>
    </div>

    <div class='card'>
        <h3 style='color: #38bdf8 !important;'>4. Keyword Gap Identification</h3>
        <p class='sub-text'>Keywords present in the job description vector but missing in the resume vector are filtered out and displayed as recommendations.</p>
    </div>
    """, unsafe_allow_html=True)
