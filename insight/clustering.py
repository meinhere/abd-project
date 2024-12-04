import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_mpp_data(data):
    def to_pascal_case(name):
        parts = name.lower().split()
        return ' '.join(part.capitalize() for part in parts)

    data = data.drop([0, 1, 2], axis=0).reset_index(drop=True)
    data = data.rename(columns={
        '38 Provinsi': 'Provinsi',
        'Unnamed: 1': '2021',
        'Unnamed: 2': '2019',
        'Unnamed: 3': '2018',
        'Unnamed: 4': '2017'
    })
    data.drop(columns=['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], inplace=True)
    data['2021'] = pd.to_numeric(data['2021'], errors='coerce')
    
    data['Provinsi'] = data['Provinsi'].apply(to_pascal_case)
    prov_name = {
        "Dki Jakarta": "Jakarta Raya",
        "Kep. Bangka Belitung": "Bangka-Belitung",
        "Kep. Riau": "Kepulauan Riau",
        "Di Yogyakarta": "Yogyakarta",
    }
    data['Provinsi'].replace(prov_name, inplace=True)

    return data


def perform_kmeans(main_df, selected_columns):
    selected_features = main_df[list(selected_columns)]
    if len(selected_features.columns) > 1:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(selected_features)
    else:
        scaled_data = selected_features.values
    return selected_features, scaled_data


def cluster_data(features_scaled, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    return cluster_labels