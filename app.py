import streamlit as st
import re
import PyPDF2
import nltk
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


#PAGE CONFIGURATION
st.set_page_config(
    page_title="ResumeAI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


#CUSTOM CSS
st.markdown(
    """
<style>

    /* =====================================================
       MAIN APP
       ===================================================== */

    .stApp {
        background-color: #f7f9fc !important;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #f7f9fc !important;
    }

    [data-testid="stMain"] {
        background-color: #f7f9fc !important;
    }


    /* =====================================================
       MAIN TEXT
       ===================================================== */

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6 {
        color: #111827 !important;
    }

    [data-testid="stMain"] p {
        color: #374151 !important;
    }

    [data-testid="stMain"] label {
        color: #111827 !important;
        font-weight: 600 !important;
    }


    /* =====================================================
       MARKDOWN
       ===================================================== */

    [data-testid="stMarkdownContainer"] p {
        color: #374151 !important;
    }

    [data-testid="stMarkdownContainer"] strong {
        color: #111827 !important;
    }


    /* =====================================================
       TEXT AREA
       ===================================================== */

    [data-testid="stTextArea"] textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
    }

    [data-testid="stTextArea"] textarea::placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        border: 1px dashed #cbd5e1 !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"] section div {
        color: #111827 !important;
    }


    /* =====================================================
       CONTAINERS
       ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 14px !important;
    }


    /* =====================================================
       METRICS
       ===================================================== */

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #475569 !important;
    }


    /* =====================================================
       BUTTON
       ===================================================== */

    .stButton > button {
        width: 100%;
        min-height: 50px;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    [data-testid="stSidebar"] {
        background-color: #252631 !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer-text {
        text-align: center;
        color: #64748b !important;
        margin-top: 40px;
        padding: 20px;
        font-size: 14px;
    }

</style>
""",
    unsafe_allow_html=True
)


#SIDEBAR
st.sidebar.title("📌 ResumeAI")

st.sidebar.subheader("Features")

st.sidebar.write(
    """
📄 Resume PDF Analysis

🎯 Job Match Score

🛠️ Technical Skill Analysis

🔑 Professional Keywords

📋 Resume Section Analysis

🤖 ATS Compatibility

📈 Experience Matching

💼 Job Role Recommendations

🔍 Resume vs Job Comparison

📥 PDF Analysis Report
"""
)

st.sidebar.divider()

st.sidebar.caption(
    "ResumeAI is a project that uses "
    "NLP and machine learning techniques for "
    "resume-job matching."
    "Made by Suman Saurabh"
)


#TECHNICAL SKILLS DATABASE
SKILLS = [
    "python", "java", "c", "c++", "c#", "javascript", "typescript",
    "php", "ruby", "go", "kotlin", "swift", "r",

    "html", "css", "react", "angular", "vue", "node.js", "node",
    "express", "bootstrap", "tailwind",

    "sql", "mysql", "postgresql", "mongodb", "oracle", "sqlite", "redis",

    "machine learning", "deep learning", "artificial intelligence",
    "data science", "data analysis", "pandas", "numpy", "scikit-learn",
    "tensorflow", "pytorch", "keras", "matplotlib", "seaborn", "nlp",
    "computer vision", "statistics",

    "aws", "azure", "gcp", "google cloud",

    "docker", "kubernetes", "jenkins", "linux", "git", "github", "gitlab",

    "django", "flask", "fastapi", "spring", "spring boot",

    "rest api", "rest", "graphql",

    "vs code", "visual studio", "jupyter", "postman", "jira", "figma",
    "power bi", "tableau", "excel"
]


#PROFESSIONAL KEYWORDS
PROFESSIONAL_KEYWORDS = [
    "communication", "leadership", "teamwork", "problem solving",
    "problem-solving", "collaboration", "management", "project management",
    "time management", "adaptability", "creativity", "critical thinking",
    "analytical thinking", "decision making", "decision-making",
    "organization", "organizational", "multitasking", "attention to detail",
    "agile", "scrum", "team player", "interpersonal skills", "presentation",
    "research", "innovation", "flexibility", "mentoring"
]


#RESUME SECTIONS
RESUME_SECTIONS = {
    "Contact Information": ["email", "phone", "mobile", "linkedin", "github", "contact"],
    "Summary / Objective": ["summary", "professional summary", "career objective", "objective", "profile"],
    "Education": ["education", "academic", "degree", "university", "college", "bachelor", "master", "b.tech", "btech"],
    "Skills": ["skills", "technical skills", "technical skill", "technologies", "technical knowledge"],
    "Projects": ["projects", "project", "academic projects", "personal projects"],
    "Experience": ["experience", "work experience", "professional experience", "employment", "internship", "internships"],
    "Certifications": ["certifications", "certification", "certificates", "certificate"],
    "Achievements": ["achievements", "achievement", "awards", "award", "honors", "honour"]
}


#JOB ROLES
JOB_ROLES = {
    "Python Developer": ["python", "django", "flask", "fastapi", "sql", "rest api", "git"],
    "Java Developer": ["java", "spring", "spring boot", "sql", "rest api", "git"],
    "Web Developer": ["html", "css", "javascript", "react", "node.js", "sql", "git"],
    "Data Analyst": ["python", "sql", "pandas", "numpy", "matplotlib", "excel", "power bi", "tableau"],
    "Machine Learning Engineer": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "sql"],
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "scikit-learn", "tensorflow", "statistics", "sql"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "jenkins", "aws", "azure", "git"],
    "Cloud Engineer": ["aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "linux"]
}


#FUNCTIONS
def extract_text_from_pdf(pdf_file):

    reader = PyPDF2.PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + " "

    return text


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z0-9+#.\- ]",
        " ",
        text
    )

    return text


def remove_stopwords(text):

    stop_words = set(
        nltk.corpus.stopwords.words("english")
    )

    words = text.split()

    filtered_words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def extract_skills(text):

    text_lower = text.lower()

    found_skills = []

    sorted_skills = sorted(
        SKILLS,
        key=len,
        reverse=True
    )

    for skill in sorted_skills:

        skill_lower = skill.lower()

        if " " in skill_lower:

            pattern = (
                r"(?<![a-zA-Z0-9])"
                + re.escape(skill_lower)
                + r"(?![a-zA-Z0-9])"
            )

        elif any(
            char in skill_lower
            for char in ["+", "#", ".", "-"]
        ):

            pattern = (
                r"(?<![a-zA-Z0-9])"
                + re.escape(skill_lower)
                + r"(?![a-zA-Z0-9])"
            )

        else:

            pattern = (
                r"\b"
                + re.escape(skill_lower)
                + r"\b"
            )

        if re.search(
            pattern,
            text_lower
        ):

            found_skills.append(
                skill
            )

    return sorted(
        set(found_skills)
    )


def compare_skills(
    resume_text,
    job_description
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    resume_set = {
        skill.lower()
        for skill in resume_skills
    }

    matched_skills = [
        skill
        for skill in job_skills
        if skill.lower() in resume_set
    ]

    missing_skills = [
        skill
        for skill in job_skills
        if skill.lower() not in resume_set
    ]

    return (
        resume_skills,
        job_skills,
        sorted(set(matched_skills)),
        sorted(set(missing_skills))
    )


def calculate_similarity(
    resume_text,
    job_description
):

    resume_clean = remove_stopwords(
        clean_text(resume_text)
    )

    job_clean = remove_stopwords(
        clean_text(job_description)
    )

    if not resume_clean or not job_clean:

        return 0

    vectorizer = TfidfVectorizer()

    matrix = vectorizer.fit_transform(
        [
            resume_clean,
            job_clean
        ]
    )

    similarity = cosine_similarity(
        matrix[0:1],
        matrix[1:2]
    )[0][0]

    return similarity * 100


def calculate_skill_match(
    matched_skills,
    job_skills
):

    if len(job_skills) == 0:

        return 100

    return (
        len(matched_skills)
        / len(job_skills)
    ) * 100


def extract_professional_keywords(text):

    text_lower = text.lower()

    found_keywords = []

    for keyword in PROFESSIONAL_KEYWORDS:

        pattern = (
            r"(?<![a-zA-Z])"
            + re.escape(keyword.lower())
            + r"(?![a-zA-Z])"
        )

        if re.search(
            pattern,
            text_lower
        ):

            found_keywords.append(
                keyword
            )

    return sorted(
        set(found_keywords)
    )


def compare_professional_keywords(
    resume_text,
    job_description
):

    resume_keywords = (
        extract_professional_keywords(
            resume_text
        )
    )

    job_keywords = (
        extract_professional_keywords(
            job_description
        )
    )

    resume_set = {
        keyword.lower()
        for keyword in resume_keywords
    }

    matched_keywords = [
        keyword
        for keyword in job_keywords
        if keyword.lower() in resume_set
    ]

    missing_keywords = [
        keyword
        for keyword in job_keywords
        if keyword.lower() not in resume_set
    ]

    return (
        resume_keywords,
        job_keywords,
        sorted(set(matched_keywords)),
        sorted(set(missing_keywords))
    )


def calculate_keyword_match(
    matched_keywords,
    job_keywords
):

    if len(job_keywords) == 0:

        return 100

    return (
        len(matched_keywords)
        / len(job_keywords)
    ) * 100


def analyze_resume_sections(text):

    text_lower = text.lower()

    section_results = {}

    for section, keywords in RESUME_SECTIONS.items():

        found = False

        for keyword in keywords:

            if keyword.lower() in text_lower:

                found = True
                break

        section_results[section] = found

    return section_results


def calculate_section_score(
    section_results
):

    if len(section_results) == 0:

        return 0

    completed = sum(
        section_results.values()
    )

    return (
        completed
        / len(section_results)
    ) * 100


def recommend_job_roles(
    resume_text
):

    resume_skills = extract_skills(
        resume_text
    )

    resume_set = {
        skill.lower()
        for skill in resume_skills
    }

    role_scores = []

    for role, required_skills in JOB_ROLES.items():

        matched = 0

        for skill in required_skills:

            if skill.lower() in resume_set:

                matched += 1

        if required_skills:

            score = (
                matched
                / len(required_skills)
            ) * 100

        else:

            score = 0

        role_scores.append(
            (role, score)
        )

    role_scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return role_scores


def extract_experience_years(text):

    text_lower = text.lower()

    experience_values = []

    patterns = [

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s*(?:of)?\s*(?:experience|exp)",

        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*years?",

        r"(\d+(?:\.\d+)?)\s*years?\s*(?:experience|exp)"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text_lower
        )

        for match in matches:

            if isinstance(match, tuple):

                for value in match:

                    try:

                        experience_values.append(
                            float(value)
                        )

                    except ValueError:

                        pass

            else:

                try:

                    experience_values.append(
                        float(match)
                    )

                except ValueError:

                    pass

    if experience_values:

        return max(
            experience_values
        )

    return 0


def calculate_experience_match(
    resume_text,
    job_description
):

    resume_experience = (
        extract_experience_years(
            resume_text
        )
    )

    job_experience = (
        extract_experience_years(
            job_description
        )
    )

    if job_experience == 0:

        return (
            100,
            resume_experience,
            job_experience
        )

    if resume_experience >= job_experience:

        return (
            100,
            resume_experience,
            job_experience
        )

    if resume_experience == 0:

        return (
            0,
            resume_experience,
            job_experience
        )

    score = (
        resume_experience
        / job_experience
    ) * 100

    return (
        min(score, 100),
        resume_experience,
        job_experience
    )


def calculate_ats_score(
    similarity_score,
    skill_score,
    keyword_score,
    section_score
):

    score = (

        skill_score * 0.40

        +

        keyword_score * 0.30

        +

        similarity_score * 0.20

        +

        section_score * 0.10
    )

    return min(
        max(score, 0),
        100
    )


def get_ats_rating(score):

    if score >= 85:

        return "Excellent 🟢"

    elif score >= 70:

        return "Good 🟢"

    elif score >= 55:

        return "Average 🟡"

    else:

        return "Needs Improvement 🔴"


def create_comparison_data(
    resume_text,
    job_description
):

    resume_skills = extract_skills(
        resume_text
    )

    job_skills = extract_skills(
        job_description
    )

    resume_set = {
        skill.lower()
        for skill in resume_skills
    }

    comparison_data = []

    for skill in job_skills:

        if skill.lower() in resume_set:

            comparison_data.append(
                {
                    "Requirement": skill,
                    "Resume": "✅",
                    "Status": "Matched"
                }
            )

        else:

            comparison_data.append(
                {
                    "Requirement": skill,
                    "Resume": "❌",
                    "Status": "Missing"
                }
            )

    return comparison_data


#PDF REPORT FUNCTION
def generate_pdf_report(

similarity_score, skill_score, keyword_score, section_score, ats_score, \
ats_rating, experience_score, resume_experience, job_experience, \
matched_skills, missing_skills, matched_keywords, missing_keywords, \
role_scores, suggestions

):

    file_path = "ResumeAI_Analysis_Report.pdf"

    document = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    heading_style = styles["Heading2"]
    normal_style = styles["BodyText"]

    story = []


    story.append(
        Paragraph(
            "ResumeAI - Resume Analysis Report",
            title_style
        )
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "AI-Powered Resume & Job Compatibility Analyzer",
            normal_style
        )
    )

    story.append(
        Spacer(1, 20)
    )


    story.append(
        Paragraph(
            "1. Score Summary",
            heading_style
        )
    )

    score_data = [

        ["Metric", "Score"],

        [
            "Overall Match",
            f"{similarity_score:.1f}%"
        ],

        [
            "Technical Skill Match",
            f"{skill_score:.1f}%"
        ],

        [
            "Professional Keyword Match",
            f"{keyword_score:.1f}%"
        ],

        [
            "Resume Section Quality",
            f"{section_score:.1f}%"
        ],

        [
            "Experience Match",
            f"{experience_score:.1f}%"
        ],

        [
            "ATS Compatibility",
            f"{ats_score:.1f}%"
        ],

        [
            "ATS Rating",
            ats_rating.replace("🟢", "")
            .replace("🟡", "")
            .replace("🔴", "")
        ]
    ]

    score_table = Table(
        score_data,
        colWidths=[
            300,
            150
        ]
    )

    score_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                7
            )
        ])
    )

    story.append(
        score_table
    )

    story.append(
        Spacer(1, 20)
    )


    story.append(
        Paragraph(
            "2. Matched Technical Skills",
            heading_style
        )
    )

    story.append(
        Paragraph(
            ", ".join(matched_skills)
            if matched_skills
            else "None",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "3. Missing Technical Skills",
            heading_style
        )
    )

    story.append(
        Paragraph(
            ", ".join(missing_skills)
            if missing_skills
            else "None",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "4. Professional Keywords",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "<b>Matched:</b> "
            + (
                ", ".join(matched_keywords)
                if matched_keywords
                else "None"
            ),
            normal_style
        )
    )

    story.append(
        Spacer(1, 8)
    )

    story.append(
        Paragraph(
            "<b>Missing:</b> "
            + (
                ", ".join(missing_keywords)
                if missing_keywords
                else "None"
            ),
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "5. Experience Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            f"Resume Experience: "
            f"{resume_experience:.1f} years",
            normal_style
        )
    )

    story.append(
        Paragraph(
            (
                f"Required Experience: "
                f"{job_experience:.1f} years"
            )
            if job_experience > 0
            else
            "Required Experience: Not specified",
            normal_style
        )
    )

    story.append(
        Paragraph(
            f"Experience Match: "
            f"{experience_score:.1f}%",
            normal_style
        )
    )

    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "6. Recommended Job Roles",
            heading_style
        )
    )

    for role, score in role_scores[:5]:

        story.append(
            Paragraph(
                f"<b>{role}</b> - "
                f"{score:.1f}% skill match",
                normal_style
            )
        )

        story.append(
            Spacer(1, 5)
        )


    story.append(
        Spacer(1, 15)
    )


    story.append(
        Paragraph(
            "7. Resume Improvement Suggestions",
            heading_style
        )
    )

    for suggestion in suggestions:

        story.append(
            Paragraph(
                f"- {suggestion}",
                normal_style
            )
        )

        story.append(
            Spacer(1, 5)
        )


    story.append(
        Spacer(1, 20)
    )


    story.append(
        Paragraph(
            "<b>Disclaimer:</b> The ATS score is a "
            "project-level heuristic based on resume-job "
            "similarity, technical skills, professional "
            "keywords and resume section completeness. "
            "It is not an actual score from a commercial ATS.",
            normal_style
        )
    )


    document.build(
        story
    )

    return file_path


#MAIN APPLICATION
def main():

#HEADER
    st.title(
        "📄 ResumeAI"
    )

    st.subheader(
        "AI-Powered Resume & Job Compatibility Analyzer"
    )

    st.write("")


#INPUT SECTION
    st.header(
        "📤 Upload Resume & Job Description"
    )


    input_col1, input_col2 = st.columns(
        2,
        gap="large"
    )


#RESUME UPLOAD
    with input_col1:

        with st.container(border=True):

            st.subheader(
                "📄 Your Resume"
            )

            st.caption(
                "Upload your resume PDF"
            )

            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                help="Upload a PDF resume for analysis."
            )

            if uploaded_file:

                st.success(
                    f"✅ {uploaded_file.name}"
                )


#JOB DESCRIPTION
    with input_col2:

        with st.container(border=True):

            st.subheader(
                "📝 Job Description"
            )

            st.caption(
                "Paste the job description"
            )

            job_description = st.text_area(
                "Job Description",
                height=180,
                placeholder=(
                    "Example:\n\n"
                    "We are looking for a Python Developer "
                    "with knowledge of SQL, Git, REST API..."
                ),
                label_visibility="collapsed"
            )


    st.write("")


#ANALYZE BUTTON
    analyze_button = st.button(
        "🚀 Analyze Resume",
        type="primary",
        use_container_width=True
    )


#ANALYSIS START
    if analyze_button:


        if uploaded_file is None:

            st.error(
                "❌ Please upload your resume PDF."
            )

            return


        if not job_description.strip():

            st.error(
                "❌ Please enter a job description."
            )

            return


        with st.spinner(
            "🔍 Reading and analyzing your resume..."
        ):

            try:

                resume_text = (
                    extract_text_from_pdf(
                        uploaded_file
                    )
                )

            except Exception as error:

                st.error(
                    f"❌ Error reading PDF: {error}"
                )

                return


        if not resume_text.strip():

            st.error(
                "❌ Could not extract text from the PDF. "
                "Please use a text-based PDF."
            )

            return


#OVERALL MATCH
        similarity_score = calculate_similarity(
            resume_text,
            job_description
        )


#TECHNICAL SKILLS
        (
            resume_skills,
            job_skills,
            matched_skills,
            missing_skills
        ) = compare_skills(
            resume_text,
            job_description
        )


        skill_score = calculate_skill_match(
            matched_skills,
            job_skills
        )


#PROFESSIONAL KEYWORDS
        (
            resume_keywords,
            job_keywords,
            matched_keywords,
            missing_keywords
        ) = compare_professional_keywords(
            resume_text,
            job_description
        )


        keyword_score = calculate_keyword_match(
            matched_keywords,
            job_keywords
        )


#RESUME SECTIONS
        section_results = (
            analyze_resume_sections(
                resume_text
            )
        )


        section_score = (
            calculate_section_score(
                section_results
            )
        )


#EXPERIENCE
        (
            experience_score,
            resume_experience,
            job_experience
        ) = calculate_experience_match(
            resume_text,
            job_description
        )


#ATS SCORE
        ats_score = calculate_ats_score(
            similarity_score,
            skill_score,
            keyword_score,
            section_score
        )


        ats_rating = get_ats_rating(
            ats_score
        )


#JOB ROLES
        role_scores = recommend_job_roles(
            resume_text
        )


#SUGGESTIONS
        suggestions = []


        if skill_score < 70:

            suggestions.append(
                "Add relevant technical skills that "
                "you genuinely possess and that appear "
                "in the job description."
            )


        if keyword_score < 70:

            suggestions.append(
                "Use relevant professional keywords "
                "from the job description where they "
                "truthfully describe your experience."
            )


        if similarity_score < 60:

            suggestions.append(
                "Tailor your resume summary and project "
                "descriptions more closely to the target job."
            )


        if section_score < 80:

            suggestions.append(
                "Add or improve important resume sections "
                "such as Experience, Projects, Certifications "
                "or Achievements."
            )


        if (
            experience_score < 70
            and job_experience > 0
        ):

            suggestions.append(
                "Your detected experience is below the "
                "job requirement. Highlight relevant "
                "internships, projects or practical experience."
            )


        if missing_skills:

            suggestions.append(
                "Review the missing technical skills and "
                "learn the ones relevant to your target role."
            )


        if not suggestions:

            suggestions.append(
                "Your resume is well aligned with this "
                "job description. Keep it concise and "
                "tailored to the target position."
            )


#MATCH OVERVIEW
        st.divider()

        st.header(
            "📊 Match Overview"
        )


        score_col1, score_col2, score_col3, score_col4, score_col5 = st.columns(5)


        with score_col1:

            st.metric(
                "🎯 Overall Match",
                f"{similarity_score:.1f}%"
            )


        with score_col2:

            st.metric(
                "🛠️ Skill Match",
                f"{skill_score:.1f}%"
            )


        with score_col3:

            st.metric(
                "🔑 Keyword Match",
                f"{keyword_score:.1f}%"
            )


        with score_col4:

            st.metric(
                "📋 Resume Quality",
                f"{section_score:.1f}%"
            )


        with score_col5:

            st.metric(
                "🤖 ATS Score",
                f"{ats_score:.1f}%"
            )


#ATS COMPATIBILITY
        st.header(
            "🤖 ATS Compatibility"
        )


        ats_col1, ats_col2, ats_col3 = st.columns(3)


        with ats_col1:

            st.metric(
                "ATS Score",
                f"{ats_score:.1f}%"
            )


        with ats_col2:

            st.metric(
                "Rating",
                ats_rating
            )


        with ats_col3:

            st.metric(
                "Skill Contribution",
                f"{skill_score:.1f}%"
            )


        st.progress(
            int(
                min(
                    max(
                        ats_score,
                        0
                    ),
                    100
                )
            )
        )


        st.info(
            "ℹ️ This ATS score is a project-level "
            "heuristic based on technical skills, "
            "professional keywords, text similarity "
            "and resume section completeness. "
            "It is not an actual score from a commercial ATS."
        )


#TECHNICAL SKILLS
        st.header(
            "🛠️ Technical Skill Analysis"
        )


        skill_col1, skill_col2 = st.columns(2)


        with skill_col1:

            with st.container(border=True):

                st.subheader(
                    "✅ Matched Skills"
                )

                if matched_skills:

                    for skill in matched_skills:

                        st.success(
                            f"✓ {skill}"
                        )

                else:

                    st.warning(
                        "No matching technical skills found."
                    )


        with skill_col2:

            with st.container(border=True):

                st.subheader(
                    "❌ Missing Skills"
                )

                if missing_skills:

                    for skill in missing_skills:

                        st.error(
                            f"✗ {skill}"
                        )

                else:

                    st.success(
                        "No major technical skills missing."
                    )


#PROFESSIONAL KEYWORDS
        st.header(
            "🔑 Professional Keywords"
        )


        keyword_col1, keyword_col2 = st.columns(2)


        with keyword_col1:

            with st.container(border=True):

                st.subheader(
                    "✅ Matched Keywords"
                )

                if matched_keywords:

                    for keyword in matched_keywords:

                        st.success(
                            f"✓ {keyword}"
                        )

                else:

                    st.warning(
                        "No matching professional keywords found."
                    )

        with keyword_col2:

            with st.container(border=True):
                st.subheader(
                    "❌ Missing Keywords"
                )
                if missing_keywords:
                    for keyword in missing_keywords:
                        st.error(
                            f"✗ {keyword}"
                        )
                else:
                    st.success(
                        "No major professional keywords missing."
                    )

#RESUME SECTION ANALYSIS
        st.header(
            "📋 Resume Section Analysis"
        )

        section_col1, section_col2 = st.columns(2)

        with section_col1:

            with st.container(border=True):

                st.subheader(
                    "Resume Sections"
                )
                for section, found in section_results.items():
                    if found:
                        st.success(
                            f"✅ {section}"
                        )
                    else:
                        st.error(
                            f"❌ {section}"
                        )

        with section_col2:

            with st.container(border=True):
                st.subheader(
                    "Section Suggestions"
                )

                missing_sections = [
                    section
                    for section, found
                    in section_results.items()

                    if not found
                ]

                if missing_sections:

                    for section in missing_sections:

                        st.warning(
                            f"Consider adding a "
                            f"**{section}** section."
                        )
                else:

                    st.success(
                        "All major resume sections detected."
                    )

#EXPERIENCE MATCHING
        st.header(
            "📈 Experience Matching"
        )

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        with exp_col1:

            st.metric(
                "Resume Experience",
                f"{resume_experience:.1f} years"
            )

        with exp_col2:

            if job_experience > 0:

                st.metric(
                    "Required Experience",
                    f"{job_experience:.1f} years"
                )
            else:

                st.metric(
                    "Required Experience",
                    "Not specified"
                )

        with exp_col3:

            st.metric(
                "Experience Match",
                f"{experience_score:.1f}%"
            )

        if job_experience == 0:

            st.info(
                "The job description does not clearly "
                "specify an experience requirement."
            )
        elif experience_score >= 100:

            st.success(
                "✅ Your detected experience meets "
                "the job requirement."
            )
        elif experience_score >= 50:

            st.warning(
                "⚠️ You partially meet the "
                "experience requirement."
            )
        else:

            st.error(
                "❌ Your detected experience is below "
                "the job requirement."
            )

#RECOMMENDED JOB ROLES
        st.header(
            "💼 Recommended Job Roles"
        )

        role_col1, role_col2 = st.columns(2)

        for index, (
            role,
            score
        ) in enumerate(
            role_scores[:6]
        ):

            if index % 2 == 0:

                target_col = role_col1
            else:

                target_col = role_col2
            if index == 0:

                rank = "🥇"
            elif index == 1:

                rank = "🥈"
            elif index == 2:

                rank = "🥉"
            else:

                rank = "💼"

            with target_col:

                with st.container(border=True):

                    st.subheader(
                        f"{rank} {role}"
                    )

                    st.write(
                        f"**{score:.1f}% skill match**"
                    )

                    st.progress(
                        int(
                            min(
                                max(
                                    score,
                                    0
                                ),
                                100
                            )
                        )
                    )

#RESUME SKILLS
        st.header(
            "🧑‍💻 Skills Detected in Your Resume"
        )


        with st.container(border=True):

            if resume_skills:

                st.write(
                    " • ".join(
                        resume_skills
                    )
                )

            else:

                st.warning(
                    "No technical skills detected."
                )

#JOB REQUIREMENTS
        st.header(
            "📌 Job Requirements Detected"
        )


        with st.container(border=True):

            if job_skills:

                st.write(
                    " • ".join(
                        job_skills
                    )
                )

            else:

                st.warning(
                    "No technical skills detected "
                    "in the job description."
                )

#RESUME VS JOB COMPARISON
        st.header(
            "🔍 Resume vs Job Comparison"
        )
        comparison_data = (
            create_comparison_data(
                resume_text,
                job_description
            )
        )

        if comparison_data:

            with st.container(border=True):

                st.write(
                    "Technical job requirements compared "
                    "with skills detected in your resume."
                )
                comparison_df = pd.DataFrame(
                    comparison_data
                )
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    hide_index=True
                )
        else:

            st.info(
                "No technical requirements were detected "
                "from the job description."
            )

#IMPROVEMENT SUGGESTIONS
        st.header(
            "💡 Resume Improvement Suggestions"
        )

        for suggestion in suggestions:

            st.info(
                f"💡 {suggestion}"
            )

#SCORE CHART
        st.header(
            "📊 Score Comparison"
        )

        scores = {

            "Overall Match":
                similarity_score,

            "Skill Match":
                skill_score,

            "Keyword Match":
                keyword_score,

            "Resume Quality":
                section_score,

            "Experience":
                experience_score,

            "ATS Score":
                ats_score
        }

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.barh(
            list(scores.keys()),
            list(scores.values())
        )

        ax.set_xlim(
            0,
            100
        )

        ax.set_xlabel(
            "Score (%)"
        )

        ax.set_title(
            "Resume Analysis Scores"
        )

        ax.invert_yaxis()

        for i, value in enumerate(
            scores.values()
        ):

            ax.text(
                min(
                    value + 1,
                    96
                ),
                i,
                f"{value:.1f}%",
                va="center"
            )
        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

#FINAL ASSESSMENT
        st.header(
            "🎯 Final Assessment"
        )
        if ats_score >= 85:

            st.success(
                "🌟 Excellent! Your resume is strongly "
                "aligned with this job."
            )
        elif ats_score >= 70:
            st.success(
                "👍 Good match! Your resume has a strong "
                "alignment with the job requirements."
            )
        elif ats_score >= 55:
            st.warning(
                "⚠️ Moderate match. Consider improving "
                "your skills, keywords and resume content."
            )
        else:
            st.error(
                "🔴 Low match. Your resume needs significant "
                "improvement for this particular job."
            )


#PDF REPORT
        st.header(
            "📥 Download Analysis Report"
        )
        report_file = generate_pdf_report(

            similarity_score, skill_score, keyword_score, section_score,
            ats_score, ats_rating, experience_score, resume_experience,
            job_experience, matched_skills, missing_skills, matched_keywords,
            missing_keywords, role_scores, suggestions
        )
        with open(
            report_file,
            "rb"
        ) as pdf_file:
            st.download_button(

                label=(
                    "📥 Download Resume "
                    "Analysis Report"
                ),
                data=pdf_file,
                file_name=(
                    "ResumeAI_Analysis_Report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True
            )

#DISCLAIMER
        st.caption(
            "Note: The ATS score is a project-level "
            "heuristic and should not be represented "
            "as a score generated by a commercial ATS."
        )

#IMPORTANT: FOOTER IS INSIDE MAIN
        st.divider()
        st.markdown(
            '<div class="footer-text">'
            '📄 ResumeAI | AI-Powered Resume & Job Compatibility Analyzer'
            '<br>'
            'Built with Python • Streamlit • NLP • Machine Learning'
            '</div>',
            unsafe_allow_html=True
        )

#RUN APPLICATION
if __name__ == "__main__":
    main()
