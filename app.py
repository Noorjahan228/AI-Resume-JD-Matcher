import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Page Configuration (Title and Icon)
st.set_page_config(page_title="AI Resume & JD Matcher", page_icon="📊", layout="centered")

# 2. Header UI
st.title("📊 AI Resume Matcher & Keyword Gap Analyzer")
st.write("Upload or paste your **Resume text** and **Job Description** to analyze match percentage and missing keywords.")

st.markdown("---")

# 3. User Inputs
jd_text = st.text_area("📋 Paste Job Description (JD) Here:", height=150, placeholder="Enter job description...")
resume_text = st.text_area("📄 Paste Your Resume Text Here:", height=150, placeholder="Enter your resume details...")

# 4. Action Button & Core Logic
if st.button("🚀 Analyze Match Score", use_container_width=True):
    if jd_text.strip() and resume_text.strip():
        # Text Vectorization using TF-IDF
        documents = [resume_text, jd_text]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Calculate Cosine Similarity Percentage
        match_score = cosine_similarity(tfidf_matrix)[0][1] * 100
        match_score = round(match_score, 2)
        
        st.markdown("---")
        
        # Display Result
        st.subheader(f"🎯 Match Score: **{match_score}%**")
        
        if match_score >= 70:
            st.success("🌟 Excellent Match! Your resume is strongly aligned with this job.")
        elif match_score >= 40:
            st.warning("⚡ Moderate Match. Consider adding more relevant keywords from the JD.")
        else:
            st.error("⚠️ Low Match. You need to customize your resume significantly.")
            
        # Keyword Gap Analysis
        feature_names = vectorizer.get_feature_names_out()
        jd_words = set([word for word, score in zip(feature_names, tfidf_matrix.toarray()[1]) if score > 0])
        resume_words = set([word for word, score in zip(feature_names, tfidf_matrix.toarray()[0]) if score > 0])
        
        missing_keywords = jd_words - resume_words
        
        st.markdown("---")
        st.subheader("💡 Key Missing Words in Your Resume:")
        
        if missing_keywords:
            # Display missing words as badges
            words_list = list(missing_keywords)[:15]
            st.write(", ".join([f"`{word}`" for word in words_list]))
        else:
            st.write("✅ Great job! No major missing keywords found.")
            
    else:
        st.error("❌ Please paste BOTH Job Description and Resume text to analyze!")