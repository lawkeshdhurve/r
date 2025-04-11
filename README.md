# SHL Assessment Recommendation Engine

This is a simple content-based recommendation system built using **Python** and **FastAPI**. It helps users find the most relevant SHL assessments based on their input, leveraging SHL's product catalog and semantic search techniques.

---

## 🔍 Features

- 🔎 Search for SHL assessments based on job role, skills, or keywords  
- 🤖 Recommends top relevant assessments using semantic similarity  
- ⚡ Powered by Sentence Transformers for improved results  
- 🧾 Clean and minimal API using FastAPI  

---

## 🛠️ Tech Stack

- Python  
- FastAPI  
- Uvicorn  
- Sentence Transformers  
- scikit-learn  
- Pandas  

---

## 🚀 How to Run

Follow these steps to run the project locally:

### 1. Clone the Repository

```bash
git clone https://github.com/lawkeshdhurve/r.git  
cd r
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn pandas sentence-transformers scikit-learn
```

### 4. Run the FastAPI App

```bash
uvicorn app:app --reload
```

Visit the FastAPI docs at: `http://127.0.0.1:8000/docs`

### 5. Open HTML Pages (If Applicable)

If your project includes an `index.html` file, right-click it in VS Code and choose:

> **Open with Live Server**  
*(Localhost: 5500)*

---

## 🖼️ Preview

### 🏠 Home Page
![Home](https://github.com/lawkeshdhurve/r/blob/886e2f49d3045c3393ed8868ff56d56f35ffc61b/Home_Preview.jpg?raw=true)

### 📋 Result Page
![Result](https://github.com/lawkeshdhurve/r/blob/886e2f49d3045c3393ed8868ff56d56f35ffc61b/result_image.jpg?raw=true)

---

## 📬 Feedback or Contributions

Feel free to fork the repository, raise issues, or submit pull requests to improve the project!
