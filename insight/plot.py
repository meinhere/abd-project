import pandas as pd
import folium
import branca
import geopandas as gpd
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA

def create_cluster_plot(selected_columns, selected_features, features_scaled):
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


def create_map(provinces, features, predictions):
    gdf = gpd.read_file("dataset/indonesia.geojson")  # Path to GeoJSON
    df = pd.DataFrame({"provinsi": provinces.tolist(), "cluster": features['Cluster'].tolist()})
    gdf = gdf.merge(df, left_on="state", right_on="provinsi", how="left")

    m = folium.Map(location=[-0.7893, 113.9213], zoom_start=5)
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
        labels_cluster = {0: 'Rendah', 1: 'Tinggi', 2: 'Sedang'}

        provinsi = row['provinsi']
        cluster = labels_cluster[row['cluster']]
        prediction_image = predictions[provinsi]['image']
        prediction_data = predictions[provinsi]['data']

        html = f"""
        <div style="min-width: 750px">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                <div class="modal-header mb-2">
                    <h2 id="predictionModalLabel">Hasil Prediksi Produksi Padi <span id="province-name">({provinsi})</span></h2>
                    <h3 id="predictionModalLabel">Cluster MPP Cabai: <span id="cluster-name">{cluster}</span></h3>
                </div>
                <div class="modal-body">
                    <div id="chart" class="mb-4">
                    <h4>Grafik Prediksi</h4>
                    {prediction_image and f'<img src="data:image/png;base64,{prediction_image}" alt="Prediksi Produksi Padi" class="img-fluid" />'}
                    </div>
                    <h4>Data Prediksi</h4>
                    <table class="table table-striped">
                    <thead>
                        <tr>
                        <th>Year</th>
                        <th>Predicted Production</th>
                        </tr>
                    </thead>
                    <tbody id="prediction-data">
                        {"".join([f"<tr><td>{data['Year']}</td><td>{data['Produksi']:,}</td></tr>" for data in prediction_data])}
                    </tbody>
                    </table>
                </div>
                </div>
            </div>
        </div>
        """

        folium.Marker(
            location=[row.geometry.centroid.y, row.geometry.centroid.x],
            popup=html,
        ).add_to(m)

    return m._repr_html_() # Get the HTML representation