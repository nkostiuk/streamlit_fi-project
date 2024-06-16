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
df_final_merge2, df_scaled_df = load_data()


st.title("French Industry Project")

###SIDEBAR ###
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
## Data Visualization Page 
if page == pages[2] : 
    st.write("### Visualisation des données")
    
    choix = [
        'Salaire moyen par région', 
        'Entreprises par région', 
        'Distribution des salaires', 
        'Salaires moyens par région (50+)', 
        'Salaires moyens par catégorie et région',  
        'Top 5 salaires en Île-de-France', 
        'Top 5 salaires en Île-de-France (50+)'
    ]
   
    option = st.selectbox('Choix du graphique', choix)
    st.write('Le graphique choisi est :', option)

    if option == 'Salaire moyen par région':
        st.write("Voici la répartition des salaires moyens par région en France. Ce graphique montre les variations de salaire moyen entre différentes régions.")
        fig = plot_moyen_salary_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("On peut observer que la région Île-de-France se distingue par un salaire moyen plus élevé comparé aux autres régions.")

    elif option == 'Entreprises par région':
        st.write("Ce graphique présente le nombre total d'entreprises par région. Il montre où se concentrent la majorité des entreprises en France.")
        fig = plot_companies_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("L'Île-de-France et la région Auvergne-Rhône-Alpes se distinguent par le nombre élevé d'entreprises, reflétant leur dynamisme économique.")

    elif option == 'Distribution des salaires':
        st.write("La distribution des salaires moyens entre les femmes et les hommes de plus de 50 ans dans l'industrie en France est illustrée ici.")
        fig = plot_salary_distribution(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Il est évident que les hommes ont tendance à avoir des salaires plus élevés que les femmes dans cette tranche d'âge.")

    elif option == 'Salaires moyens par région (50+)':
        st.write("Ce graphique compare les salaires moyens des hommes et des femmes de plus de 50 ans par région.")
        fig = plot_average_salaries_over_50_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("On constate des différences significatives de salaire entre les genres dans certaines régions, avec les hommes gagnant généralement plus.")

    elif option == 'Salaires moyens par catégorie et région':
        st.write("Voici les salaires moyens par catégorie professionnelle et par région.")
        fig = plot_average_salaries_by_category_and_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Les cadres ont les salaires les plus élevés, suivis par les employés et les travailleurs. Ces différences se reflètent également à travers les régions.")

    elif option == 'Top 5 salaires en Île-de-France':
        st.write("Les cinq valeurs extrêmes de salaire dans la région Île-de-France sont présentées ici.")
        fig = plot_top_5_salaries_idf(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Ces villes se distinguent par leurs salaires élevés, montrant une forte disparité au sein de la région Île-de-France.")

    elif option == 'Top 5 salaires en Île-de-France (50+)':
        st.write("Les cinq villes avec les salaires moyens les plus élevés pour les hommes et pour les femmes (50+ ans) en Île-de-France.")
        fig_male, fig_female = plot_top_5_salaries_idf_comparison(df_final_merge2)
        st.plotly_chart(fig_male)
        st.plotly_chart(fig_female)
        st.write("Ces graphiques montrent que même parmi les meilleures villes, il existe des disparités salariales entre les hommes et les femmes de plus de 50 ans.")


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


# Initialize a dictionary to store the scores
scores = {
    'Method': [],
    'Calinski-Harabasz Index': [],
    'Silhouette Score': []
}

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


# @st.cache_data
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
    # evaluate_clustering(clustering_key, data_transformed, labels)
    
    clustering_results = {clustering_key: {'data': data_transformed, 'labels': labels}}
    
    return clustering_results


geojson_path = 'src/departements-avec-outre-mer.geojson'


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

       
        # Automatically evaluate after clustering
        for method, results in clustering_results.items():
            evaluate_clustering(method, results['data'], results['labels'])

        # Afficher les scores de clustering
        st.write("### Scores de Clustering")
        if scores['Method']:
            scores_df = pd.DataFrame(scores)
            st.dataframe(scores_df)
        else:
            st.write("Aucun score à afficher. Veuillez lancer l'évaluation.")


if page == pages[5]: 
    st.write("### Conclusion")