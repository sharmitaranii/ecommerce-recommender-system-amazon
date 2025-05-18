# 🛒 E-commerce Product Recommender System

A hybrid recommendation engine that helps users discover relevant products using both collaborative and content-based filtering techniques.

---

## 🔍 Why This Project?

Recommender systems power every major e-commerce platform today. This project explores **LightFM** (a hybrid matrix factorization algorithm) along with **TF-IDF + cosine similarity** to provide:

- 🔹 Personalized suggestions based on past user behavior (collaborative filtering)
- 🔸 Item similarity using product metadata (content-based filtering)
- 🤝 A hybrid approach combining both for smarter recommendations

---

## 🚀 Features

- 📦 Product recommendations using user interaction data
- 🧠 Content-based recommendations using product descriptions
- 🌀 Hybrid recommender combining LightFM + cosine similarity
- 📊 Evaluation using `precision@k` and `recall@k`
- 🌐 Streamlit web app for interactive use

---

## 📁 Files and Structure

E-commerce Product Recommender/
│
├── app.py # Streamlit web app
├── lightfm_model.pkl # Trained LightFM model + dataset
├── cosine_sim.npy # Content-based similarity matrix
├── unique_products.csv # Cleaned product info
├── processed.csv # Final preprocessed interaction data
├── requirements.txt # Dependencies for deployment
└── README.md # Project documentation

