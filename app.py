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

# Custom High-Visibility Styling
st.markdown("""
    <style>
    /* Background Image with Deep Purple Overlay */
    .stApp {
        background: linear-gradient(rgba(15, 23, 42, 0.88), rgba(88, 28, 135, 0.90)), 
                    url('https://images.unsplash.com/photo-1507842237336-84751139328c?q=80&w=1920&auto=format&fit=crop') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }

    /* FORCED WHITE NAVIGATION TABS AT TOP */
    button[data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px 8px 0px 0px !important;
        margin-right: 8px !important;
        padding: 10px 20px !important;
    }
    button[data-baseweb="tab"] div, button[data-baseweb="tab"] p {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
    }
    button[aria-selected="true"] {
        background-color: #a855f7 !important;
        border-bottom: 3px solid #f0abfc !important;
    }
    button[aria-selected="true"] div, button[aria-selected="true"] p {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Global Headings & Text */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #ffffff !important;
    }

    /* SOLID WHITE INPUT BOXES */
    textarea, input {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        border: 2px solid #a855f7 !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }

    /* Radio Box Container */
    div[role="radiogroup"] {
        background-color: rgba(30, 27, 75, 0.9) !important;
        padding: 10px !important;
        border-radius: 10px !important;
        border: 2px solid #a855f7 !important;
    }

    /* File Uploader Container */
    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #a855f7 !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    div[data-testid="stFileUploader"] span, div[data-testid="stFileUploader"] small {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Glassmorphism Cards */
    .hero-banner {
        background: linear-gradient(135deg, rgba(88, 28, 135, 0.9) 0%, rgba(126, 34, 206, 0.9) 100%);
        border: 2px solid #c084fc;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
    }

    .card {
        background: rgba(30, 27, 75, 0.85);
        border: 2px solid #a855f7;
        backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
    }

    .score-card {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4);
    }

    /* Primary Action Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #7c3aed 0%, #c084fc 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        padding: 0.85rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4) !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #a855f7 100%) !important;
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
        <h1 style='margin:0; font-size: 2.2rem;'>📚 AI Resume & Document Analyzer</h1>
        <p style='margin-top:8px; font-size: 1.05rem;'>
            Upload your PDF resume to check instant compatibility against target job listings using natural language processing.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #f0abfc !important;'>📑 1. PDF Parser</h3>
            <p>Extracts text page-by-page directly from uploaded PDF resumes and documents.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #f0abfc !important;'>📊 2. Match Score</h3>
            <p>Calculates cosine similarity metrics to estimate how well your resume matches the job post.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='card'>
            <h3 style='color: #f0abfc !important;'>🔍 3. Missing Keywords</h3>
            <p>Identifies missing technical terms and skills needed to pass automated ATS filters.</p>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 2: RESUME MATCHER
# ==============================================================================
with tab2:
    st.subheader("Compare Resume with Job Description")
    st.write("Paste job requirements and upload your PDF resume below.")
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
                    <p style='margin:0; font-size: 1.1rem;'>Estimated ATS Match Score</p>
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
        <h3 style='color: #f0abfc !important;'>1. PDF Extraction</h3>
        <p>The app parses uploaded PDF files page-by-page using <code>PyPDF2</code> to extract raw text strings.</p>
    </div>
    
    <div class='card'>
        <h3 style='color: #f0abfc !important;'>2. TF-IDF Vectorization</h3>
        <p>Converts cleaned text into mathematical term-frequency vectors after filtering stop words.</p>
    </div>

    <div class='card'>
        <h3 style='color: #f0abfc !important;'>3. Cosine Distance Calculation</h3>
        <p>Calculates geometric vector alignment to measure document similarity percentage.</p>
    </div>

    <div class='card'>
        <h3 style='color: #f0abfc !important;'>4. Skill Gap Reporting</h3>
        <p>Identifies terms present in the job posting but missing from your resume vector.</p>
    </div>
    """, unsafe_allow_html=True)
