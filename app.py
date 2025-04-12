from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

# Inline assessment data
assessment_data = [
    {
        "title": "Cognitive Ability Test",
        "description": "Measures a candidate’s ability to solve problems, analyze data, and think logically under pressure."
    },
    {
        "title": "Situational Judgement Test",
        "description": "Assesses decision-making and interpersonal skills in realistic workplace scenarios."
    },
    {
        "title": "Personality Questionnaire",
        "description": "Evaluates behavioral preferences to predict culture fit and work style."
    },
    {
        "title": "Technical Skills Assessment - Python",
        "description": "Tests proficiency in Python programming including algorithms, data structures, and libraries."
    },
    {
        "title": "Leadership Potential Assessment",
        "description": "Identifies candidates with strong leadership traits and strategic thinking abilities."
    },
    {
        "title": "Numerical Reasoning Test",
        "description": "Assesses a candidate’s ability to interpret, analyze, and draw conclusions from numerical data."
    },
    {
        "title": "Verbal Reasoning Test",
        "description": "Evaluates understanding and interpretation of written information."
    },
    {
        "title": "Deductive Reasoning Test",
        "description": "Tests ability to apply logical rules to arrive at conclusions."
    },
    {
        "title": "Java Developer Assessment",
        "description": "Evaluates Java programming skills including object-oriented design, collections, and concurrency."
    },
    {
        "title": "Customer Service Simulation",
        "description": "Simulates real-world customer interactions to evaluate responsiveness and empathy."
    },
    {
        "title": "Data Analyst Test",
        "description": "Assesses skills in data wrangling, SQL, data visualization, and statistical reasoning."
    },
    {
        "title": "Critical Thinking Assessment",
        "description": "Measures ability to evaluate arguments, identify assumptions, and draw logical conclusions."
    },
    {
        "title": "Cloud Engineering Assessment",
        "description": "Tests cloud architecture, deployment, monitoring, and security using AWS and Azure tools."
    },
    {
        "title": "Project Management Assessment",
        "description": "Evaluates knowledge of Agile, Scrum, risk management, and stakeholder communication."
    },
    {
        "title": "Sales Aptitude Test",
        "description": "Assesses negotiation skills, product knowledge, and persuasion techniques."
    },
    {
        "title": "UX Design Simulation",
        "description": "Evaluates user-centric thinking, wireframing, and usability principles."
    },
    {
        "title": "Financial Analyst Assessment",
        "description": "Tests knowledge in budgeting, forecasting, financial modeling, and Excel."
    },
    {
        "title": "AI/ML Skills Assessment",
        "description": "Evaluates understanding of machine learning algorithms, model evaluation, and Python libraries like scikit-learn."
    },
    {
        "title": "Cybersecurity Awareness Test",
        "description": "Assesses knowledge of phishing, malware, encryption, and security best practices."
    },
    {
        "title": "DevOps Proficiency Test",
        "description": "Evaluates CI/CD pipelines, infrastructure as code, and deployment automation tools."
    }
]


# Prepare DataFrame and model
df = pd.DataFrame(assessment_data)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

@app.post("/recommend")
async def recommend(query: Query):
    job_text = query.query.strip()
    if not job_text or df.empty:
        return []

    descriptions = df["description"].astype(str).tolist()
    job_embedding = model.encode([job_text])
    assessment_embeddings = model.encode(descriptions)

    similarities = cosine_similarity(job_embedding, assessment_embeddings)[0]
    df_results = df.copy()
    df_results["similarity"] = similarities

    # Add a threshold (e.g., 0.4)
    top_results = df_results[df_results["similarity"] >= 0.4].sort_values(
        by="similarity", ascending=False).head(5)

    return top_results.to_dict(orient="records")
