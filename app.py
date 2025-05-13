import gradio as gr
import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load preprocessed dataset
df = pd.read_csv("ratings_Electronics.csv")
df.columns = ['userId', 'productId', 'rating', 'timestamp']
df = df[df['userId'].map(df['userId'].value_counts()) >= 5]
df = df[df['productId'].map(df['productId'].value_counts()) >= 5]
df.reset_index(drop=True, inplace=True)

# Content-Based Filtering Setup
top_users = df['userId'].value_counts().head(5000).index
top_products = df['productId'].value_counts().head(5000).index
filtered_df = df[df['userId'].isin(top_users) & df['productId'].isin(top_products)]
item_user_matrix = filtered_df.pivot_table(index='productId', columns='userId', values='rating').fillna(0)
product_similarity = cosine_similarity(item_user_matrix)
product_similarity_df = pd.DataFrame(product_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

# LightFM Setup
dataset = Dataset()
dataset.fit(users=df['userId'], items=df['productId'])
(interactions, _) = dataset.build_interactions((row['userId'], row['productId'], row['rating']) for _, row in df.iterrows())
model = LightFM(loss='warp')
model.fit(interactions, epochs=5, num_threads=2)

# Mappings
user_ids = df['userId'].unique().tolist()
product_ids = df['productId'].unique().tolist()

# Recommender functions
def recommend_products_lightfm(user_id, n=5):
    user_id_map, _, item_id_map, _ = dataset.mapping()
    if user_id not in user_id_map:
        return ["User ID not found."]
    user_index = user_id_map[user_id]
    scores = model.predict(user_index, np.arange(interactions.shape[1]))
    top_indices = np.argsort(-scores)[:n]
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    return [reverse_item_map[i] for i in top_indices]

def get_similar_products(product_id, n=5):
    if product_id not in product_similarity_df.columns:
        return ["Product ID not found."]
    scores = product_similarity_df[product_id].sort_values(ascending=False)
    return scores.iloc[1:n+1].index.tolist()

def hybrid_recommender(user_id, product_id, n=5):
    user_id_map, _, item_id_map, _ = dataset.mapping()
    if user_id not in user_id_map or product_id not in product_similarity_df.columns:
        return ["User or Product ID not found."]
    user_index = user_id_map[user_id]
    scores = model.predict(user_index, np.arange(interactions.shape[1]))
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    all_pids = [reverse_item_map[i] for i in np.argsort(-scores)]
    sim_scores = product_similarity_df[product_id]
    ranked = sorted([(pid, sim_scores[pid]) for pid in all_pids if pid in sim_scores and pid != product_id], key=lambda x: -x[1])
    return [pid for pid, _ in ranked[:n]]

def recommend(user_id, product_id, method, n):
    if method == "Collaborative Filtering":
        return recommend_products_lightfm(user_id, n)
    elif method == "Content-Based Filtering":
        return get_similar_products(product_id, n)
    else:
        return hybrid_recommender(user_id, product_id, n)

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Product Recommender")
    with gr.Row():
        user_input = gr.Dropdown(label="User ID", choices=user_ids, value=user_ids[0])
        product_input = gr.Textbox(label="Product ID", value=product_ids[0])
        method_input = gr.Radio(choices=["Collaborative Filtering", "Content-Based Filtering", "Hybrid Recommender"], value="Hybrid Recommender", label="Method")
    num_slider = gr.Slider(1, 10, step=1, value=5, label="Number of Recommendations")
    output = gr.Textbox(label="Recommended Products")
    gr.Button("Recommend").click(fn=recommend, inputs=[user_input, product_input, method_input, num_slider], outputs=output)

demo.launch()
