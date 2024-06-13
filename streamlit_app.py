import subprocess

# # Diagnostic information
# st.write(f"Python executable: {sys.executable}")
# st.write(f"Python version: {sys.version}")
# st.write("Installed packages:")
# st.code(subprocess.check_output([sys.executable, "-m", "pip", "list"]).decode("utf-8"))

import sys
sys.path.append('src')

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import umap
from sklearn.metrics import calinski_harabasz_score, silhouette_score 
import folium
import json
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

# Import modules from the appended path
from fi_functions import *

# Import the data
@st.cache_data
def load_data():
    df_final_merge2 = pd.read_csv('data/df_final_merge2.csv')
    df_scaled_df = pd.read_csv('data/df_scaled_df.csv')
    
    return df_final_merge2, df_scaled_df

df_final_merge2, df_scaled_df = load_data()

###SIDEBAR 
st.title("French Industry Project")
# Add the project logo 
logo_url = "img/dst-logo.png"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Contexte et objectifs du projet", "Le Jeu De Données", "Data Vizualization", "Préparation des données" ,"Modélisation", "Conclusion"]
page=st.sidebar.radio("Sélectionnez une partie :", pages)

# Add a line before the section
st.sidebar.markdown("---") 

# Adding logo, names, and LinkedIn links with photos
st.sidebar.markdown("## Project Team")

# Define the function to create the HTML for each team member
def create_team_member(name, photo_url, linkedin_url):
    html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <img src="{photo_url}" alt="{name}" style="width:50px; height:50px; border-radius:50%; margin-right:10px;">
        <div>
            <h4 style="margin: 0;">{name}</h4>
            <a href="{linkedin_url}" target="_blank">LinkedIn</a>
        </div>
    </div>
    """
    return html

# Add team members
team_members = [
    {"name": "Nataliia KOSTIUK", "photo_url": "img/Nataliia.jpeg", "linkedin_url": "https://www.linkedin.com/in/nataliia-kostiuk-7334bb62/"},
    {"name": "Xavier COMTE", "photo_url": "img/Xavier.jpeg", "linkedin_url": "https://www.linkedin.com/in/xaviercompte75/"},
    {"name": "Maxime CARRION", "photo_url": "img/Maxime.png", "linkedin_url": "https://www.linkedin.com/in/maximecarrion/"}
]

for member in team_members:
    # Create a layout with image and text side by side
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.image(member['photo_url'])
    with col2:
        st.markdown(f"**{member['name']}**")
        st.markdown(f"[LinkedIn]({member['linkedin_url']})")

# Add a line after the section
st.sidebar.markdown("---")


## Introduction Page 
if page == pages[0] : 
  st.write("### Introduction")
  st.markdown("""
              #### Contexte  
              L'INSEE, Institut national de la statistique et des études économiques, est l'organisme officiel français chargé de recueillir une variété de données sur le territoire français. 
              Ces données, qu'elles soient démographiques (telles que les naissances, les décès, la densité de la population...) ou économiques (comme les salaires, le nombre d'entreprises par secteur d'activité ou par taille...), offrent une vision complète de la société française. Elles constituent ainsi une ressource précieuse pour analyser et comprendre les dynamiques sociales, économiques et territoriales du pays.
              """)
  st.markdown("""
            #### Les objectifs de notre analyse
            Cette étude vise à comparer les inégalités en France selon plusieurs dimensions. 
            - Tout d'abord, nous nous pencherons sur les disparités entre les entreprises, en examinant leur **localisation** géographique et leur taille. 
            - Ensuite, nous nous intéresserons aux inégalités au sein de la population, en analysant les **variations de salaires** en fonction de différents critères tels que la catégorie **d’emploi** et la **localisation** géographique. 
            - Enfin, nous concentrerons notre attention sur une grande ville en particulier, afin d'étudier de manière approfondie les inégalités qui peuvent exister à l'échelle locale et les problématiques que ces inégalités pourraient provoquer dans les enjeux industriels évoqués précédemment.
              
            **Nous répondrons aux problématiques suivantes :**
            - Comment varie le nombre d’emplois (entreprises) en fonction de la région.
            - Comment les disparités salariales varient-elles selon le genre et l’âge dans ces différentes régions et d'en comprendre les raisons ? 
            """)


## Le Jeu De Données Page 
if page == pages[1]:
    st.write("### Le jeu de données")
    st.write("Les sources employées pour ce projet proviennent toutes de l’INSEE, Institut national de la statistique et des études économiques. Ce sont donc des données officielles concernant les différents aspects économiques, démographiques et géographiques traités dans ce travail.")

    st.markdown("""
    - La table **'base_etablissement_par_tranche_effectif.csv'** : Informations sur le nombre d'entreprises dans chaque ville française classées par taille.
    - La table **'entreprises24'** : Informations sur le nombre d'entreprises dans chaque ville française classées par taille (données de 2021).
    - La table **'name_geographic_information'** : Informations géographiques sur les villes françaises, y compris la latitude et la longitude, ainsi que les codes et noms des régions et départements.
    - La table **'net_salary_per_town_categories'** : Informations sur les salaires nets moyens par heure dans chaque ville française classées par catégories d'emploi, âge et sexe.
    - La table **'population'** : Informations démographiques par ville, y compris le niveau géographique, le mode de cohabitation, les catégories d'âge et le sexe.
    - La table **'diplomes_2020_new'** : Informations sur le nombre de personnes non scolarisées de 15 ans ou plus classées par niveau de diplôme (aucun diplôme, CEP, Bac +5 ou plus).
    """)

    st.write("#### Afficher les données")
    
    if st.checkbox("Afficher un aperçu de la table 'Final Merge'") :  
        st.dataframe(df_final_merge2.head())
    
    if st.checkbox("Afficher la table sommaire du DataFrame 'Final Merge'") :      
        st.dataframe(summary(df_final_merge2))


## DataViz Page
if page == pages[2] : 
    
    # def plot_median_salary_by_region(df):
    #     fig = px.box(df, x='REG_nom', y='SNHM20', title='Salaire médian par région', labels={'REG_nom': 'Région', 'SNHM20': 'Salaire Médian'})
    #     fig.update_layout(xaxis_tickangle=-45)
    #     return fig
    
    @st.cache_data
    def plot_moyen_salary_by_region(df):
        fig = px.box(df, x='REG_nom', y='mean_net_salary_hour_overall', title='Salaire moyen par région', labels={'REG_nom': 'Région', 'mean_net_salary_hour_overall': 'Salaire Moyen'})
        fig.update_layout(xaxis_tickangle=-45)
        pio.renderers.default = 'svg'  
        return fig

    @st.cache_data
    def plot_companies_by_region(df):
        reg_name_enterprises = df.groupby('REG_nom')['Total_Salaries'].sum().reset_index()
        reg_name_enterprises_sorted = reg_name_enterprises.sort_values(by='Total_Salaries', ascending=False)
        fig = px.bar(reg_name_enterprises_sorted, x='REG_nom', y='Total_Salaries',
                    title='Nombre total d\'entreprises par région',
                    labels={'REG_nom': 'Nom de la Région', 'Total_Salaries': 'Nombre total d\'entreprises'},
                    color='Total_Salaries',
                    color_continuous_scale='viridis')
        fig.update_layout(xaxis_tickangle=-45, xaxis_title=None, yaxis_title=None)
        return fig

    @st.cache_data
    def plot_salary_distribution(df):
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['mean_net_salary_hour_female_over_50'], histnorm='percent', name='Femmes', marker_color='purple', opacity=0.6))
        fig.add_trace(go.Histogram(x=df['mean_net_salary_hour_male_over_50'], histnorm='percent', name='Hommes', marker_color='orange', opacity=0.6))
        fig.update_layout(barmode='overlay', title='Distribution des salaires moyens entre femmes et hommes de plus de 50 ans dans l\'industrie en France', xaxis_title='Salaire Moyen', yaxis_title='Pourcentage', bargap=0.1)
        fig.update_traces(opacity=0.75)
        fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        return fig

    @st.cache_data
    def plot_average_salaries_over_50_by_region(df):
        df_region_salary = df.groupby('REG_nom').agg({'mean_net_salary_hour_female_over_50': 'mean', 'mean_net_salary_hour_male_over_50': 'mean'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_region_salary['REG_nom'], y=df_region_salary['mean_net_salary_hour_female_over_50'], name='Femmes', marker_color='purple'))
        fig.add_trace(go.Bar(x=df_region_salary['REG_nom'], y=df_region_salary['mean_net_salary_hour_male_over_50'], name='Hommes', marker_color='orange'))
        fig.update_layout(barmode='group', title='Salaires des femmes et des hommes de plus de 50 ans par région', xaxis_title='Région', yaxis_title='Salaire Moyen')
        return fig

    @st.cache_data
    def plot_average_salaries_by_category_and_region(df):
        # Aggregate data by region and calculate mean salaries for each category
        agg_df = df.groupby('REG_nom').agg({
            'mean_net_salary_hour_overall': 'mean',  # Mean salary per hour
            'mean_net_salary_hour_executives': 'mean',  # Mean salary per hour for executives
            'mean_net_salary_hour_avg_executive': 'mean',  # Mean salary per hour for middle management
            'mean_net_salary_hour_employee': 'mean',  # Mean salary per hour for employees
            'mean_net_salary_hour_worker': 'mean',  # Mean salary per hour for workers
        }).reset_index()

        # Create a figure
        fig = go.Figure()

        # Add bar traces for each category, using the aggregated data
        fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_overall'], name='Salaire net moyen par heure', marker_color='blue'))
        fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_executives'], name='Salaire net moyen par heure pour les cadres', marker_color='red'))
        fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_avg_executive'], name='Salaire net moyen par heure pour un cadre moyen', marker_color='green'))
        fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_employee'], name='Salaire net moyen par heure pour l\'employé', marker_color='orange'))
        fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_worker'], name='Salaire net moyen par heure pour le travailleur', marker_color='purple'))

        # Update layout
        fig.update_layout(
            title='Salaires moyens par catégorie et région',
            xaxis_title='Région',
            yaxis_title='Salaire moyen',
            xaxis_tickangle=-45,
            barmode='group',
            legend_title='Catégorie'
        )

        return fig
    @st.cache_data
    def plot_top_5_salaries_idf(df):
        idf_data = df[df['REG_nom'] == 'Île-de-France']
        idf_data_sorted = idf_data.sort_values(by='mean_net_salary_hour_overall', ascending=False)
        top_salaries_unique = idf_data_sorted.drop_duplicates(subset=['mean_net_salary_hour_overall'], keep='first')
        top_5_salaries = top_salaries_unique.head(5)
        fig = px.bar(top_5_salaries, x='COM_name', y='mean_net_salary_hour_overall',
                     title='Les 5 valeurs extrêmes de salaire dans la région Île-de-France',
                     labels={'COM_name': 'Noms des villes', 'mean_net_salary_hour_overall': 'Salaire'},
                     color='mean_net_salary_hour_overall',
                     color_continuous_scale='Blues')
        fig.update_layout(xaxis_tickangle=45)
        return fig

    @st.cache_data
    def plot_top_5_salaries_idf_comparison(df):
        idf_data = df[df['REG_nom'] == 'Île-de-France']
        
        # Salaires des hommes
        idf_data_sorted_male = idf_data.sort_values(by='mean_net_salary_hour_male_over_50', ascending=False)
        top_salaries_unique_male = idf_data_sorted_male.drop_duplicates(subset=['mean_net_salary_hour_male_over_50'], keep='first')
        top_5_salaries_male = top_salaries_unique_male.head(5)
        
        # Salaires des femmes
        idf_data_sorted_female = idf_data.sort_values(by='mean_net_salary_hour_female_over_50', ascending=False)
        top_salaries_unique_female = idf_data_sorted_female.drop_duplicates(subset=['mean_net_salary_hour_female_over_50'], keep='first')
        top_5_salaries_female = top_salaries_unique_female.head(5)
        
        # Graphique pour les hommes
        fig_male = go.Figure()
        fig_male.add_trace(go.Bar(x=top_5_salaries_male['COM_name'], y=top_5_salaries_male['mean_net_salary_hour_male_over_50'], 
                                name='Hommes', marker_color='orange'))
        fig_male.update_layout(
            title='Les 5 villes avec les salaires moyens les plus élevés pour les hommes (50+ ans) en Île-de-France',
            xaxis_title='Noms des villes',
            yaxis_title='Salaire moyen',
            xaxis_tickangle=45
        )
        
        # Graphique pour les femmes
        fig_female = go.Figure()
        fig_female.add_trace(go.Bar(x=top_5_salaries_female['COM_name'], y=top_5_salaries_female['mean_net_salary_hour_female_over_50'], 
                                    name='Femmes', marker_color='purple'))
        fig_female.update_layout(
            title='Les 5 villes avec les salaires moyens les plus élevés pour les femmes (50+ ans) en Île-de-France',
            xaxis_title='Noms des villes',
            yaxis_title='Salaire moyen',
            xaxis_tickangle=45
        )
        
        return fig_male, fig_female

    st.write("### Visualisation des données")
    
    choix = ['Salaire moyen par région', 'Entreprises par région', 'Distribution des salaires', 'Salaires moyens par région (50+)', 'Salaires moyens par catégorie et région',  'Top 5 salaires en Île-de-France', 'Top 5 salaires en Île-de-France (50+)']
   

    option = st.selectbox('Choix du graphique', choix)
    st.write('Le graphique choisi est :', option)

    if option == 'Salaire moyen par région':
        fig = plot_moyen_salary_by_region(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Entreprises par région':
        fig = plot_companies_by_region(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Distribution des salaires':
        fig = plot_salary_distribution(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Salaires moyens par région (50+)':
        fig = plot_average_salaries_over_50_by_region(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Salaires moyens par catégorie et région':
        fig = plot_average_salaries_by_category_and_region(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Top 5 salaires en Île-de-France':
        fig = plot_top_5_salaries_idf(df_final_merge2)
        st.plotly_chart(fig)

    elif option == 'Top 5 salaires en Île-de-France (50+)':
        fig_male, fig_female = plot_top_5_salaries_idf_comparison(df_final_merge2)
        st.plotly_chart(fig_male)
        st.plotly_chart(fig_female)

## Préparation des données Page
#if page == pages[3] : 
#    st.write("### Préparation des données")

#### MODELISATION

# Global variable to store clustering means
cluster_means_global = {}
features_num_selected = ['Total_Salaries', 'nb_auto_entrepreneur',
       'nb_micro_entreprises', 'nb_small_entreprises', 'nb_medium_entreprises',
       'nb_large_entreprises','DIST_COM_CL_DEPT', 'DIST_COM_CL_REG',
       'DIST_COM_PARIS', 'mean_net_salary_hour_overall',
       'mean_net_salary_hour_executives', 'mean_net_salary_hour_avg_executive',
       'mean_net_salary_hour_employee', 'mean_net_salary_hour_worker',
       'mean_net_salary_hour_female', 'mean_net_salary_hour_female_executives',
       'mean_net_salary_hour_avg_female_executive',
       'mean_net_salary_hour_female_employee',
       'mean_net_salary_hour_female_worker', 'mean_net_salary_hour_male',
       'mean_net_salary_hour_male_executives',
       'mean_net_salary_hour_avg_male_executive',
       'mean_net_salary_hour_male_employee',
       'mean_net_salary_hour_male_worker', 'mean_net_salary_hour_18_25',
       'mean_net_salary_hour_26_50', 'mean_net_salary_hour_over_50',
       'mean_net_salary_hour_female_18_25',
       'mean_net_salary_hour_female_26_50',
       'mean_net_salary_hour_female_over_50',
       'mean_net_salary_hour_male_18_25', 'mean_net_salary_hour_male_26_50',
       'mean_net_salary_hour_male_over_50', 'men_over_15_no_diploma',
       'men_over_15_bac_plus_5_gratuated', 'women_over_15_no_diploma',
       'women_over_15_bac_plus_5_gratuated']

@st.cache_data
def plot_scatter(data, labels, title='Scatter Plot', x_label='X-axis', y_label='Y-axis', color_map='viridis'):
    plt.figure(figsize=(10, 4))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_map)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar(label='Cluster')
    st.pyplot(plt)


@st.cache_data
def plot_dendrogram(data, p=10, color_threshold=150):
    linked = linkage(data, method='ward')

    plt.figure(figsize=(10, 5))
    dendrogram(linked, orientation='top', distance_sort='descending', truncate_mode='level', p=p, 
               color_threshold=color_threshold, show_leaf_counts=True)
    plt.axhline(y=color_threshold, color='r', linestyle='--')
    plt.title('Dendrogramme des clusters')
    plt.xlabel('Index des points de données')
    plt.ylabel('Distance')
    st.pyplot(plt)


# Initialize a dictionary to store the scores
scores = {
    'Method': [],
    'Calinski-Harabasz Index': [],
    'Silhouette Score': []
}

@st.cache_data
# Function to evaluate and store scores
def evaluate_clustering(method_name, data, labels):
    ch_score = calinski_harabasz_score(data, labels)
    silhouette_avg = silhouette_score(data, labels)
    scores['Method'].append(method_name)
    scores['Calinski-Harabasz Index'].append(ch_score)
    scores['Silhouette Score'].append(silhouette_avg)


def get_cluster_column_name(method, reducer, n_clusters):
    if reducer:
        return f'{method.capitalize()} clustering après {reducer.upper()} Clustering'
    return f'{method.capitalize()} clustering sans ACP Clustering'


@st.cache_data
def clustering_generic(data, n_clusters, method='kmeans', reducer=None, n_components=None, perplexity=None, learning_rate=None, n_iter=None, n_neighbors=None, min_dist=None):
    if reducer == 'pca' and n_components:
        pca = PCA(n_components=n_components)
        data_transformed = pca.fit_transform(data)
        title_suffix = f'ACP avec {n_components} composantes principales'
    elif reducer == 'tsne' and n_components:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        data_transformed = tsne.fit_transform(data)
        title_suffix = f'T-SNE (perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter})'
    elif reducer == 'umap' and n_components:
        reducer_model = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
        data_transformed = reducer_model.fit_transform(data)
        title_suffix = f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})'
    else:
        data_transformed = data.values
        title_suffix = 'sans réduction de dimension'

    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
        clustering_key = f'KMeans {title_suffix}'
    elif method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clustering_key = f'AgglomerativeClustering {title_suffix}'
    
    clustering.fit(data_transformed)
    labels = clustering.labels_
    df_final_merge2[clustering_key] = labels

    cluster_means = df_final_merge2.groupby(clustering_key)[features_num_selected].mean().reset_index()

    st.write(f"Moyennes des caractéristiques par cluster ({title_suffix})")
    st.dataframe(cluster_means)

    plot_scatter(data_transformed, labels, 
                 title=f'{method.capitalize()} Clustering (k={n_clusters}) {title_suffix}', 
                 x_label='Composante principale 1', 
                 y_label='Composante principale 2')

    # Evaluation du clustering
    evaluate_clustering(clustering_key, data_transformed, labels)
    
    clustering_results = {clustering_key: {'data': data_transformed, 'labels': labels}}
    
    return clustering_results

def generate_popup(dept_name, cluster_num, cluster_means):
    return f"<b>Department:</b> {dept_name}<br><b>Cluster:</b> {cluster_num}<br><b>Cluster Means:</b> {cluster_means[cluster_num]}"

def plot_cluster_map(data, geojson_path, cluster_column, cluster_colors, legend_html, popup_html_func, cluster_means):
    # Initialize map
    m = folium.Map(location=[46.2276, 2.2137], zoom_start=6)
    
    # Load GeoJSON data
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Iterate through each feature of the GeoJSON and add it to the map
    for feature in geojson_data['features']:
        dept_code = feature['properties']['code'] 
        matching_row = data[data['DEPT_code'] == dept_code]
        if not matching_row.empty:
            cluster_num = matching_row.iloc[0][cluster_column]
            dept_name = matching_row.iloc[0]['DEPT_name']
            folium.GeoJson(
                feature,
                style_function=lambda x, c=cluster_num: {
                    'fillColor': cluster_colors[c],
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
                popup=folium.Popup(popup_html_func(dept_name, cluster_num, cluster_means), max_width=300)
            ).add_to(m)

    # Add legend
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# Define cluster colors and create the base map
cluster_colors_acp_optimised = {
    0: '#EF779F',  # Purple
    1: '#ff9f00',  # Orange
    2: '#00a693',  # Green
    3: '#d62728',  # Red
    4: '#1180FF', # Blue
}

# Add a custom legend
legend_html = """
<div style="position: fixed; 
     bottom: 50px; left: 50px; width: 200px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px; 
     background:white; padding:10px;">
     <h4 style="margin:8px;">Cluster Legend</h4>
     <div style="background:#EF779F;width:10px;height:10px;display:inline-block;"></div> Cluster 0 - Purple<br>
     <div style="background:#ff9f00;width:10px;height:10px;display:inline-block;"></div> Cluster 1 - Orange<br>
     <div style="background:#00a693;width:10px;height:10px;display:inline-block;"></div> Cluster 2 - Green<br>
     <div style="background:#d62728;width:10px;height:10px;display:inline-block;"></div> Cluster 3 - Red<br>
     <div style="background:#1180FF;width:10px;height:10px;display:inline-block;"></div> Cluster 4 - Blue<br>
</div>
"""


cluster_means_acp_kmeans_optimised = df_final_merge2.groupby('Cluster-ACP-KMeans-best')[features_num_selected].mean().reset_index()



geojson_path = 'src/departements-avec-outre-mer.geojson'


# Load the geospatial data for France
@st.cache_data
def load_geospatial_data():
    gdf = gpd.read_file(geojson_path)
    # Reproject to a projected CRS
    gdf = gdf.to_crs(epsg=3857)
    # Calculate centroids
    gdf['longitude'] = gdf.geometry.centroid.x
    gdf['latitude'] = gdf.geometry.centroid.y
    # Reproject back to the original geographic CRS
    gdf = gdf.to_crs(epsg=4326)
    return gdf


gdf = load_geospatial_data()

# Afficher la section de modélisation
if page == pages[4]:
    st.write("### Modélisation")
    st.write("""
    Le clustering est une technique d'analyse de données qui divise un ensemble de données en groupes homogènes appelés clusters.

    En regroupant les départements selon leurs caractéristiques socio-économiques telles que le salaire, le niveau d'éducation (avec ou sans diplôme bac+5) et le type d'entreprise, le clustering nous permet de mettre en lumière à la fois les similarités et les disparités entre les différents groupes.

    Pour tester différentes méthodes de clustering, nous allons utiliser les approches suivantes :

    - K-means clustering sans ACP
    - K-means clustering après ACP
    - K-means clustering après T-SNE
    - AgglomerativeClustering après T-SNE
    - AgglomerativeClustering après ACP
    - K-means clustering après UMAP

    Ces différentes méthodes nous permettront de comparer les résultats et de choisir la plus adaptée à notre analyse.
    """)

    # Afficher df_scaled_df 
    if st.checkbox("Afficher un aperçu de la table 'Final Merge' standardisée") :  
        st.dataframe(df_scaled_df.head())

    # Sélecteur pour choisir le modèle de clustering
    models = [
        'K-means clustering sans ACP',
        'K-means clustering après ACP',
        'K-means clustering après T-SNE',
        'AgglomerativeClustering après T-SNE',
        'AgglomerativeClustering après ACP',
        'K-means clustering après UMAP'
    ]
    
    st.write("#### Sélection du nombre des clusters optimal")
    plot_dendrogram(df_scaled_df)
    
    selected_model = st.selectbox('Choisissez la méthode de clustering à utiliser', models)
    
    n_clusters = st.number_input('Nombre de clusters', min_value=2, max_value=10, value=5)

    if 'après ACP' in selected_model or 'T-SNE' in selected_model or 'UMAP' in selected_model:
        n_components = st.number_input('Nombre de composantes principales', min_value=2, max_value=10, value=2)
    else:
        n_components = None

    if 'T-SNE' in selected_model:
        st.write("#### Paramètres T-SNE")
        col1, col2, col3 = st.columns(3)
        with col1:
            perplexity = st.number_input('Perplexity', min_value=5, max_value=50, value=30)
        with col2:
            learning_rate = st.number_input('Learning Rate', min_value=10, max_value=1000, value=200)
        with col3:
            n_iter = st.number_input('Number of Iterations', min_value=250, max_value=10000, value=1000)
    else:
        perplexity = None
        learning_rate = None
        n_iter = None

    if 'UMAP' in selected_model:
        st.write("#### Paramètres UMAP")
        col1, col2 = st.columns(2)
        with col1:
            n_neighbors = st.number_input('Nombre de Voisins (n_neighbors)', min_value=2, max_value=100, value=15)
        with col2:
            min_dist = st.number_input('Distance Minimum (min_dist)', min_value=0.0, max_value=1.0, value=0.1)
    else:
        n_neighbors = None
        min_dist = None

    st.write(f'Le modèle sélectionné est : {selected_model}')

    # Bouton pour lancer le clustering
    if st.button('Lancer le clustering'):
        if 'sans ACP' in selected_model:
            st.write("### Clustering")
            clustering_results = clustering_generic(df_scaled_df, n_clusters=n_clusters, method='kmeans')
        elif 'après ACP' in selected_model:
            method = 'kmeans' if 'K-means' in selected_model else 'agglomerative'
            st.write("### Clustering")
            clustering_results = clustering_generic(df_scaled_df, n_clusters=n_clusters, method=method, reducer='pca', n_components=n_components)
        elif 'après T-SNE' in selected_model:
            method = 'kmeans' if 'K-means' in selected_model else 'agglomerative'
            st.write("### Clustering")
            clustering_results = clustering_generic(df_scaled_df, n_clusters=n_clusters, method=method, reducer='tsne', n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
        elif 'après UMAP' in selected_model:
            st.write("### Clustering")
            clustering_results = clustering_generic(df_scaled_df, n_clusters=n_clusters, method='kmeans', reducer='umap', n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist)


        # Afficher les scores de clustering
        st.write("### Scores de Clustering")
        scores_df = pd.DataFrame(scores)
        st.dataframe(scores_df)


if page == pages[5]: 
    st.write("### Conclusion")