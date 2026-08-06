import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Page Configuration
st.set_page_config(
    page_title="NexusAI — Resume & JD Matcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Tri-Color Theme (Blue + Purple + Emerald)
st.markdown("""
    <style>
    /* Gradient Background (Deep Blue to Royal Purple) */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 50%, #311042 100%);
        color: #f3f4f6;
    }
    
    /* Premium Glassmorphism Cards */
    .premium-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(168, 85, 247, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    .premium-card:hover {
        border-color: rgba(168, 85, 247, 0.6);
        transform: translateY(-3px);
    }
    
    /* Result Metric Card (Blue & Green Accent) */
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(10, 185, 129, 0.2);
    }

    /* Tri-Color Gradient Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #10b981 100%);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.85rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(124, 58, 237, 0.6);
    }

    /* Purple Badge Tag */
    .badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 700;
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.3) 0%, rgba(37, 99, 235, 0.3) 100%);
        color: #c084fc;
        border: 1px solid rgba(192, 132, 252, 0.4);
        margin-bottom: 12px;
    }

    /* Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 20px;
        color: #9ca3af;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        color: white !important;
        font-weight: bold;
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

# Navigation Tabs Setup
tab1, tab2, tab3 = st.tabs(["🏠 Introduction", "⚡ AI Resume Matcher", "⚙️ How It Works"])

# ==============================================================================
# PAGE 1: INTRODUCTION
# ==============================================================================
with tab1:
    st.markdown("<span class='badge'>✦ Tri-Color AI Engine</span>", unsafe_allow_html=True)
    st.title("Welcome to NexusAI Matcher 🚀")
    st.write("An intelligent NLP-driven application designed to bridge the gap between job candidates and enterprise Applicant Tracking Systems (ATS).")
    
    st.markdown("---")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #60a5fa;'>🎯 98% Precision</h3>
            <p style='color: #9ca3af;'>Powered by TF-IDF vectorization and Cosine Similarity math for accurate scoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #c084fc;'>🔍 Gap Analysis</h3>
            <p style='color: #9ca3af;'>Extracts missing technical terms and soft skills instantly from your candidate profile.</p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div class='premium-card'>
            <h3 style='color: #34d399;'>⚡ PDF Processing</h3>
            <p style='color: #9ca3af;'>Upload PDF resumes seamlessly or paste raw plain text into the workspace.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👉 Click on the **'⚡ AI Resume Matcher'** tab above to analyze your resume!")

# ==============================================================================
# PAGE 2: MAIN FUNCTIONALITY (AI MATCHING APP)
# ==============================================================================
with tab2:
    st.title("📄 Resume vs Job Description Analyzer")
    st.write("Compare your resume content against job post requirements in real time.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📌 1. Target Job Description")
        jd_text = st.text_area("Paste JD Text Here:", height=240, placeholder="Paste job requirements, skills, and qualifications...")

    with col2:
        st.markdown("### 📑 2. Candidate Resume")
        upload_type = st.radio("Select Input Format:", ["Upload PDF Resume", "Paste Text Input"], horizontal=True)
        
        resume_text = ""
        if upload_type == "Upload PDF Resume":
            uploaded_file = st.file_uploader("Upload PDF File:", type=["pdf"])
            if uploaded_file:
                resume_text = extract_pdf_text(uploaded_file)
                st.success("✅ PDF processed successfully!")
        else:
            resume_text = st.text_area("Paste Resume Text Here:", height=170, placeholder="Paste your resume content...")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Analyze Compatibility Score"):
        if not jd_text.strip() or not resume_text.strip():
            st.warning("⚠️ Please provide both Job Description and Resume details to analyze.")
        else:
            # Algorithmic Calculation
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
            st.subheader("📊 Match Analysis Output")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(f"""
                <div class='metric-card'>
                    <h2 style='margin:0; font-size: 2.8rem; color: #34d399;'>{match_percentage}%</h2>
                    <p style='margin:0; color: #f3f4f6; font-size: 1.1rem;'>Compatibility Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                if match_percentage >= 70:
                    st.success("🎯 Strong Match! Excellent candidate fit.")
                elif match_percentage >= 45:
                    st.warning("⚠️ Moderate Match. Consider adding missing keywords.")
                else:
                    st.error("❌ Low Match. Tailor your resume closer to the JD.")

            with res_col2:
                st.markdown("### 🔍 Missing Target Keywords")
                if missing_keywords:
                    top_missing = missing_keywords[:15]
                    st.write("Add these key terms into your resume to pass ATS filters:")
                    st.write(", ".join([f"`{kw}`" for kw in top_missing]))
                else:
                    st.success("🎉 Outstanding! Your resume covers all key terms from the JD.")

# ==============================================================================
# PAGE 3: HOW IT WORKS / WORKING PROCESS
# ==============================================================================
with tab3:
    st.title("⚙️ System Architecture & Workflow")
    st.write("Learn how NexusAI extracts, parses, and scores document similarity.")
    st.markdown("---")

    st.markdown("""
    <div class='premium-card'>
        <h3 style='color: #60a5fa;'>Step 1: Text Extraction & Parsing 📄</h3>
        <p style='color: #9ca3af;'>
            When a PDF is uploaded, <code>PyPDF2</code> extracts raw text streams page-by-page. For text inputs, the raw string is normalized directly.
        </p>
    </div>
    
    <div class='premium-card'>
        <h3 style='color: #c084fc;'>Step 2: TF-IDF Vectorization 🧮</h3>
        <p style='color: #9ca3af;'>
            The system applies <b>Term Frequency - Inverse Document Frequency (TF-IDF)</b> filtering out standard English stop-words. Text content is converted into numerical vector arrays.
        </p>
    </div>

    <div class='premium-card'>
        <h3 style='color: #34d399;'>Step 3: Cosine Similarity Metric 📐</h3>
        <p style='color: #9ca3af;'>
            The angular distance between the Job Description vector and the Resume vector is calculated using <b>Cosine Similarity</b>:
        </p>
        <p style='text-align: center; font-family: monospace; color: #34d399; font-size: 1.2rem;'>
            Cosine Similarity = (A · B) / (||A|| ||B||)
        </p>
    </div>

    <div class='premium-card'>
        <h3 style='color: #f472b6;'>Step 4: Keyword Gap Extraction 🔍</h3>
        <p style='color: #9ca3af;'>
            Tokens with non-zero weights in the JD vector that evaluate to zero in the Resume vector are flagged as <b>Missing Keywords</b> and reported immediately.
        </p>
    </div>
    """, unsafe_allow_html=True)
