from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import base64
import folium
import geopandas as gpd


app = Flask(__name__)

# Load dataset (ensure the CSV is accessible by your Flask app)
main_df = pd.read_csv('dataset/main.csv', delimiter=",")

# Preprocessing steps (same as your notebook)
main_df = main_df.drop([0, 1, 2], axis=0).reset_index(drop=True)
main_df = main_df.rename(columns={
    '38 Provinsi': 'Provinsi',
    'Unnamed: 1': '2021',
    'Unnamed: 2': '2019',
    'Unnamed: 3': '2018',
    'Unnamed: 4': '2017'
})
main_df.drop(columns=['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], inplace=True)
main_df['2021'] = pd.to_numeric(main_df['2021'], errors='coerce')


@app.route("/", methods=["GET", "POST"])
def index():
    selected_columns = []
    cluster_plot = None
    map_html = None  # Initialize map_html
    n_clusters = 3

    if request.method == "POST":
        selected_columns = request.form.getlist("columns")
        n_clusters = int(request.form.get("n_clusters", 3))

        if selected_columns:
            features, features_scaled = perform_kmeans(selected_columns)

            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            features['Cluster'] = cluster_labels

            # Generate cluster plot
            cluster_plot = create_cluster_plot(selected_columns, features, features_scaled)

            # Generate map
            map_html = create_map(main_df['Provinsi'], features)

    # Available columns for selection
    available_columns = main_df.columns.tolist()[1:]

    return render_template("index.html", available_columns=available_columns, selected_columns=selected_columns, cluster_plot=cluster_plot, map_html=map_html, n_clusters=n_clusters)


def perform_kmeans(selected_columns):
    selected_features = main_df[list(selected_columns)]
    if len(selected_features.columns) > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(selected_features)
    else:
        scaled_data = selected_features.values
    return selected_features, scaled_data


def create_cluster_plot(selected_columns, selected_features, features_scaled):
    # Plotting
    if len(selected_features.columns) > 2 :
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(features_scaled)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        pca_df['Cluster'] = selected_features['Cluster']
        plt.figure(figsize=(8, 6))
        for cluster in pca_df['Cluster'].unique():
            cluster_data = pca_df[pca_df['Cluster'] == cluster]
            plt.scatter(cluster_data['PC1'], cluster_data['PC2'], label=f'Cluster {cluster}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA Plot of Clusters')
        plt.legend()
    else:
        plt.figure(figsize=(8,6))
        for cluster in selected_features['Cluster'].unique():
          cluster_data = selected_features[selected_features['Cluster'] == cluster]
          plt.scatter(cluster_data.index, cluster_data[selected_columns[0]], label=f'Cluster {cluster}')
        plt.xlabel("Index")
        plt.ylabel(selected_columns[0])
        plt.title(f'Cluster Plot for {selected_columns[0]}')
        plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close() # Close the figure to prevent display issues
    return plot_url


def create_map(provinces, features):
    def to_pascal_case(name):
        parts = name.lower().split()
        return ' '.join(part.capitalize() for part in parts)
    
    provinces = provinces.apply(to_pascal_case)
    prov_name = {
        "Dki Jakarta": "Jakarta Raya",
        "Kep. Bangka Belitung": "Bangka-Belitung",
        "Kep. Riau": "Kepulauan Riau",
        "Di Yogyakarta": "Yogyakarta",
    }
    provinces.replace(prov_name, inplace=True)

    gdf = gpd.read_file("dataset/indonesia.geojson")  # Path to GeoJSON
    df = pd.DataFrame({"provinsi": provinces.tolist(), "cluster": features['Cluster'].tolist()})
    gdf = gdf.merge(df, left_on="state", right_on="provinsi", how="left")

    m = folium.Map(location=[-2.5, 118], zoom_start=5)
    folium.Choropleth(
        geo_data=gdf.to_json(),  # Directly convert to JSON
        data=df,
        columns=["provinsi", "cluster"],
        key_on="feature.properties.state",
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name="Cluster"
    ).add_to(m)

    for index, row in gdf.iterrows():
        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=f"{row['provinsi']}, Cluster: {row['cluster']}",
        ).add_to(m)


    return m._repr_html_() # Get the HTML representation


if __name__ == "__main__":
    app.run(debug=True)