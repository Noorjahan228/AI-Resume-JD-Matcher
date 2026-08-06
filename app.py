import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Page Setup
st.set_page_config(
    page_title="Resume & Job Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High Visibility & Visual PDF Theme CSS
st.markdown("""
    <style>
    /* Dark Deep Purple & Navy Background */
    .stApp {
        background: linear-gradient(135deg, #090d16 0%, #111827 100%) !important;
        color: #ffffff !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Global White Text Enforcement */
    p, span, label, div, h1, h2, h3, h4, h5, h6, .stMarkdown, .stRadio label {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Textarea & Inputs - Solid High Contrast Background */
    textarea, input {
        background-color: #1e1b4b !important;
        color: #ffffff !important;
        border: 2px solid #6366f1 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
    
    /* File Uploader Box Custom Design */
    div[data-testid="stFileUploader"] {
        background-color: #1e1b4b !important;
        border: 2px dashed #818cf8 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }

    /* Radio Inputs Container */
    div[role="radiogroup"] {
        background-color: #1e1b4b !important;
        padding: 12px !important;
        border-radius: 10px !important;
        border: 1px solid #6366f1 !important;
    }

    /* Visual Banner & Feature Cards */
    .hero-banner {
        background: linear-gradient(135deg, #312e81 0%, #4c1d95 100%);
        border: 1px solid #818cf8;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.25);
    }

    .card {
        background: #1e1b4b;
        border: 1px solid #6366f1;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }

    .score-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3);
    }

    /* Gradient Primary Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        padding: 0.85rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
    }

    /* Navigation Tabs Styling */
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #c084fc !important;
        border-bottom-color: #c084fc !important;
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
    st.markdown("""
    <div class='hero-banner'>
        <h1 style='margin:0; font-size: 2.2rem; color: #ffffff !important;'>📄 AI Resume & ATS Matcher</h1>
        <p style='margin-top:8px; color: #e0e7ff !important; font-size: 1.05rem;'>
            Upload your PDF resume to check instant compatibility against target job listings using natural language processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #818cf8 !important;'>📑 1. PDF Parser</h3>
            <p style='color: #cbd5e1 !important;'>Automatically extracts text content page-by-page directly from your uploaded PDF file.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #c084fc !important;'>📊 2. Match Score</h3>
            <p style='color: #cbd5e1 !important;'>Calculates cosine similarity metrics to estimate how well your resume matches the job post.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #38bdf8 !important;'>🔍 3. Missing Keywords</h3>
            <p style='color: #cbd5e1 !important;'>Identifies missing technical terms and skills needed to pass automated ATS filters.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: RESUME MATCHER
# ==============================================================================
with tab2:
    st.subheader("Compare Resume with Job Description")
    st.write("Paste job post requirements and upload your PDF resume below.")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 1. Target Job Description")
        jd_text = st.text_area("Paste job requirements here:", height=220, placeholder="Example: Looking for a Python Developer experienced with SQL, Data Analysis, and APIs...")

    with col2:
        st.markdown("### 📑 2. Candidate Resume")
        upload_type = st.radio("Choose Input Format:", ["Upload PDF Resume", "Paste Plain Text"], horizontal=True)
        
        resume_text = ""
        if upload_type == "Upload PDF Resume":
            uploaded_file = st.file_uploader("📁 Drag and Drop or Browse PDF Resume File:", type=["pdf"])
            if uploaded_file:
                resume_text = extract_pdf_text(uploaded_file)
                st.success("✅ PDF loaded and parsed successfully!")
        else:
            resume_text = st.text_area("Paste resume text here:", height=150, placeholder="Paste your resume text content...")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Compatibility Score"):
        if not jd_text.strip() or not resume_text.strip():
            st.warning("Please provide both Job Description and Resume details.")
        else:
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

            st.markdown("---")
            st.markdown("### 📊 Match Analysis Output")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                <div class='score-card'>
                    <h1 style='margin:0; color: #34d399 !important; font-size: 3.2rem;'>{match_percentage}%</h1>
                    <p style='margin:0; color: #ffffff !important; font-size: 1.1rem;'>Estimated ATS Match Score</p>
                </div>
                """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("### 🔍 Missing Target Keywords")
                if missing_keywords:
                    top_missing = missing_keywords[:15]
                    st.write("Add these key terms into your resume to pass ATS screening:")
                    st.write(", ".join([f"`{kw}`" for kw in top_missing]))
                else:
                    st.success("Your resume covers all major keywords mentioned in the job description.")

# ==============================================================================
# TAB 3: HOW IT WORKS
# ==============================================================================
with tab3:
    st.subheader("How This App Processes Documents")
    st.write("A breakdown of PDF parsing and natural language processing steps.")
    
    st.markdown("""
    <div class='card'>
        <h3 style='color: #818cf8 !important;'>1. PDF Extraction</h3>
        <p style='color: #cbd5e1 !important;'>The app parses uploaded PDF files page-by-page using <code>PyPDF2</code> to extract raw text strings.</p>
    </div>
    
    <div class='card'>
        <h3 style='color: #c084fc !important;'>2. TF-IDF Vectorization</h3>
        <p style='color: #cbd5e1 !important;'>Converts cleaned text into mathematical term-frequency vectors after filtering stop words.</p>
    </div>

    <div class='card'>
        <h3 style='color: #38bdf8 !important;'>3. Cosine Distance Calculation</h3>
        <p style='color: #cbd5e1 !important;'>Calculates geometric vector alignment to measure document similarity percentage.</p>
    </div>

    <div class='card'>
        <h3 style='color: #f472b6 !important;'>4. Skill Gap Reporting</h3>
        <p style='color: #cbd5e1 !important;'>Identifies terms present in the job posting but missing from your resume vector.</p>
    </div>
    """, unsafe_allow_html=True)
