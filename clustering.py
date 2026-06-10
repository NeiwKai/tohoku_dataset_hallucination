import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer
from umap import UMAP
import plotly.express as px

if __name__ == '__main__':
    df = pd.read_csv("question_list.csv")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        df["question"].tolist(),
        normalize_embeddings=True
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    clusters = kmeans.fit_predict(embeddings)

    '''
    # Reduce dimensions for visualization
    reducer = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42
    )

    embedding_2d = reducer.fit_transform(embeddings)


    plot_df = pd.DataFrame({
        "x": embedding_2d[:, 0],
        "y": embedding_2d[:, 1],
        "cluster": clusters,
        "question": df["question"]
    })

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=plot_df["cluster"].astype(str),
        hover_data=["question"]
    )

    fig.show()
    '''

    df["question_type"] = clusters

    for cluster_id in sorted(df["question_type"].unique()):
        if cluster_id == -1:
            continue

        print(f"\n=== Cluster {cluster_id} ===")

        samples = (
            df[df["question_type"] == cluster_id]
            .sample(min(20, len(df[df["question_type"] == cluster_id])))
        )

        for q in samples["question"]:
            print("-", q)

    df.to_csv("dataset_logging_cluster.csv", index=False, quoting=1)
