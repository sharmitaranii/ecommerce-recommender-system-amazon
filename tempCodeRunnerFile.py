import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.title("🛍️ E-commerce Hybrid Recommender System")

# ------------------------------
# Load files
# ------------------------------
@st.cache_resource
def load_model_data():
    with open("lightfm_model.pkl", "rb") as f:
        model, dataset, interactions = pickle.load(f)

    cosine_sim = np.load("cosine_sim.npy")
    products = pd.read_csv("unique_products.csv")

    user_id_map, _, item_id_map, _ = dataset.mapping()
    reverse_user_map = {v: k for k, v in user_id_map.items()}
    reverse_item_map = {v: k for k, v in item_id_map.items()}

    return model, dataset, interactions, cosine_sim, products, user_id_map, item_id_map, reverse_user_map, reverse_item_map

model, dataset, interactions, cosine_sim, products, user_id_map, item_id_map, reverse_user_map, reverse_item_map = load_model_data()


# ------------------------------
# Recommend from LightFM
# ------------------------------
st.header("🔍 Enter a User ID")
user_input = st.text_input("Enter a valid user ID:")

def recommend_lightfm(user_id, top_n=5):
    if user_id not in user_id_map:
        return f"❌ User ID '{user_id}' not found in training data.", []

    user_index = user_id_map[user_id]
    n_items = len(item_id_map)
    scores = model.predict(user_ids=user_index, item_ids=np.arange(n_items))
    top_items = np.argsort(-scores)[:top_n]
    top_product_ids = [reverse_item_map[i] for i in top_items]
    recs = products[products['product_id'].isin(top_product_ids)][['product_name', 'about_product']]
    return "🎯 LightFM Recommendations:", recs


# ------------------------------
# Recommend using content-based filtering
# ------------------------------
def recommend_content_based(user_id, top_n=5):
    if user_id not in user_id_map:
        return f"❌ User ID '{user_id}' not found.", []

    user_index = user_id_map[user_id]
    interacted_items = interactions.tocsr()[user_index].indices

    if len(interacted_items) == 0:
        return "⚠️ No previous interactions found for this user.", []

    avg_sim = cosine_sim[interacted_items].mean(axis=0)
    similar_indices = np.argsort(-avg_sim)[:top_n]
    recs = products.iloc[similar_indices][['product_name', 'about_product']]
    return "📚 Content-Based Recommendations:", recs


# ------------------------------
# Hybrid Recommendation Logic
# ------------------------------
def recommend_hybrid(user_id, top_n=5, alpha=0.5):
    if user_id not in user_id_map:
        return f"❌ User ID '{user_id}' not found.", []

    user_index = user_id_map[user_id]
    n_items = len(item_id_map)
    scores_cf = model.predict(user_ids=user_index, item_ids=np.arange(n_items))
    interacted_items = interactions.tocsr()[user_index].indices

    if len(interacted_items) == 0:
        return "⚠️ No hybrid recommendation possible (no user history).", []

    scores_cb = cosine_sim[interacted_items].mean(axis=0)
    hybrid_scores = alpha * scores_cf + (1 - alpha) * scores_cb
    top_items = np.argsort(-hybrid_scores)[:top_n]
    top_product_ids = [reverse_item_map[i] for i in top_items]
    recs = products[products['product_id'].isin(top_product_ids)][['product_name', 'about_product']]
    return "🤝 Hybrid Recommendations:", recs


# ------------------------------
# Display all recommendations
# ------------------------------
if user_input:
    msg1, lightfm_recs = recommend_lightfm(user_input)
    st.subheader(msg1)
    st.dataframe(lightfm_recs)

    msg2, content_recs = recommend_content_based(user_input)
    st.subheader(msg2)
    st.dataframe(content_recs)

    msg3, hybrid_recs = recommend_hybrid(user_input)
    st.subheader(msg3)
    st.dataframe(hybrid_recs)
