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
df_final_merge2, df_scaled_df, results_df = load_data()


st.title("French Industry Project")

###SIDEBAR ###
# Add the project logo 
logo_url = "img/dst-logo.png"
st.sidebar.image(logo_url)

st.sidebar.title("Sommaire")
pages=["Contexte et objectifs du projet", "Le Jeu De Données", "Data Vizualization", "Préparation des données" ,"Modélisation", "Analyse des Résultats", "Conclusion"]
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
            - Comment varie le nombre d’entreprises en fonction de la région.
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
    
    # if st.checkbox("Afficher la table sommaire du DataFrame 'Final Merge'") :      
    #     st.dataframe(summary(df_final_merge2))


## Data Visualization Page ##
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
        st.write("Voici la répartition des salaires net moyens par heure par région en France. Ce graphique montre les variations de salaire moyen entre différentes régions.")
        fig = plot_moyen_salary_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("On peut observer que la région Île-de-France se distingue par un salaire net moyen par heure plus élevé comparé aux autres régions.")

    elif option == 'Entreprises par région':
        st.write("Ce graphique présente le nombre total d'entreprises par région. Il montre où se concentrent la majorité des entreprises en France.")
        fig = plot_companies_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("L'Île-de-France et la région Auvergne-Rhône-Alpes se distinguent par le nombre élevé d'entreprises, reflétant leur dynamisme économique.")

    elif option == 'Distribution des salaires':
        st.write("La distribution des salaires net moyens par heure entre les femmes et les hommes de plus de 50 ans dans l'industrie en France est illustrée ici.")
        fig = plot_salary_distribution(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Il est évident que les hommes ont tendance à avoir des salaires plus élevés que les femmes dans cette tranche d'âge.")

    elif option == 'Salaires moyens par région (50+)':
        st.write("Ce graphique compare les salaires net moyens par heure des hommes et des femmes de plus de 50 ans par région.")
        fig = plot_average_salaries_over_50_by_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("On constate des différences significatives de salaire entre les genres dans certaines régions, avec les hommes gagnant généralement plus.")

    elif option == 'Salaires moyens par catégorie et région':
        st.write("Voici les salaires net moyens par heure par catégorie professionnelle et par région.")
        fig = plot_average_salaries_by_category_and_region(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Les cadres ont les salaires les plus élevés, suivis par les employés et les travailleurs. Ces différences se reflètent également à travers les régions.")

    elif option == 'Top 5 salaires en Île-de-France':
        st.write("Les cinq valeurs extrêmes de salaire net moyen par heure dans la région Île-de-France sont présentées ici.")
        fig = plot_top_5_salaries_idf(df_final_merge2)
        st.plotly_chart(fig)
        st.write("Ces villes se distinguent par leurs salaires élevés, montrant une forte disparité au sein de la région Île-de-France.")

    elif option == 'Top 5 salaires en Île-de-France (50+)':
        st.write("Les cinq villes avec les salaires net moyens par heure les plus élevés pour les hommes et pour les femmes (50+ ans) en Île-de-France.")
        fig_male, fig_female = plot_top_5_salaries_idf_comparison(df_final_merge2)
        st.plotly_chart(fig_male)
        st.plotly_chart(fig_female)
        st.write("Ces graphiques montrent que même parmi les villes où les salaires sont les meilleurs, il existe des disparités salariales entre les hommes et les femmes de plus de 50 ans.")



### Préparation des données Page ###
if page == pages[3]:
    # Example of the column renaming code
    
    st.write("### Préparation des données")

    st.write("""
    La préparation des données est une étape cruciale dans tout projet d'analyse de données. Elle consiste à nettoyer, transformer et standardiser les données brutes pour les rendre exploitables par les algorithmes de modélisation et d'analyse. 
    
    Dans notre projet, nous avons suivi les étapes suivantes pour préparer nos données:

    **1. Nettoyage des données:**
    - **Gestion des valeurs manquantes:** Nous avons traité les valeurs manquantes dans les colonnes de latitude et longitude en recherchant et en intégrant des coordonnées GPS supplémentaires pour enrichir notre dataset.
    - **Suppression des doublons:** Nous avons supprimé les doublons pour assurer l'unicité de nos enregistrements.
    - **Correction des types de données:** Nous avons corrigé les types de données pour certaines colonnes, en particulier la latitude et la longitude, en remplaçant les virgules par des points et en les convertissant en type float.
             
    """)

    st.markdown("""
    **Suppression des doublons:**
    ```python
    # Suppression des doublons
    df_final_merge2.drop_duplicates(inplace=True)
    ```

    **Correction des types de données:**
    ```python
    # Correction des types de données
    df_final_merge2['latitude'] = df_final_merge2['latitude'].str.replace(',', '.').astype(float)
    df_final_merge2['longitude'] = df_final_merge2['longitude'].str.replace(',', '.').astype(float)
    ```
    """)

    st.write("""
    **2. Transformation des données:**
    - **Uniformisation des formats (ajout de 0 sur GEOCOD):** Nous avons uniformisé les formats des codes géographiques en ajoutant des zéros au début des valeurs de la colonne CODGEO pour assurer une longueur uniforme.
    - **Renommage des colonnes (Français/Anglais):** Nous avons renommé les colonnes pour assurer la cohérence et la clarté entre les différents jeux de données, en créant des noms explicites en anglais pour les colonnes initialement en français.
    """)


    st.markdown("""
    **Uniformisation des formats (ajout de 0 sur GEOCOD):**

    ```python
    # Convert the specified column to string type and add leading zeros to match the desired length
    def add_leading_zeros(df, column_name, desired_length):
        df[column_name] = df[column_name].astype(str)
        df[column_name] = df[column_name].str.zfill(desired_length)

    # Applying the function
    add_leading_zeros(df_name_geographic_final,'COM_code_insee', 5)
    ```

    **Renommage des colonnes (Français/Anglais):**
    
    ```python
    # Renommage des colonnes pour cohérence
    df_final_merge2.rename(columns={
        'SNHM20': 'mean_net_salary_hour_overall',
        'SNHMC20': 'mean_net_salary_hour_executives',
        'SNHMP20': 'mean_net_salary_hour_avg_executive',
        'SNHME20': 'mean_net_salary_hour_employee',
        'SNHMO20': 'mean_net_salary_hour_worker',
        'SNHMF20': 'mean_net_salary_hour_female',
        'SNHMFC20': 'mean_net_salary_hour_female_executives',
        'SNHMFP20': 'mean_net_salary_hour_avg_female_executive',
        'SNHMFE20': 'mean_net_salary_hour_female_employee',
        'SNHMFO20': 'mean_net_salary_hour_female_worker',
        'SNHMH20': 'mean_net_salary_hour_male',
        'SNHMHC20': 'mean_net_salary_hour_male_executives',
        'SNHMHP20': 'mean_net_salary_hour_avg_male_executive',
        'SNHMHE20': 'mean_net_salary_hour_male_employee',
        'SNHMHO20': 'mean_net_salary_hour_male_worker',
        'SNHM1820': 'mean_net_salary_hour_18_25',
        'SNHM2620': 'mean_net_salary_hour_26_50',
        'SNHM5020': 'mean_net_salary_hour_over_50',
        'SNHMF1820': 'mean_net_salary_hour_female_18_25',
        'SNHMF2620': 'mean_net_salary_hour_female_26_50',
        'SNHMF5020': 'mean_net_salary_hour_female_over_50',
        'SNHMH1820': 'mean_net_salary_hour_male_18_25',
        'SNHMH2620': 'mean_net_salary_hour_male_26_50',
        'SNHMH5020': 'mean_net_salary_hour_male_over_50',
        'DEPT_nom': 'DEPT_name',
        'P20_HNSCOL15P_DIPLMIN': 'men_over_15_no_diploma',
        'P20_HNSCOL15P_SUP5': 'men_over_15_bac_plus_5_graduated',
        'P20_FNSCOL15P_DIPLMIN': 'women_over_15_no_diploma',
        'P20_FNSCOL15P_SUP5': 'women_over_15_bac_plus_5_graduated'
    }, inplace=True)
    ```
    """)


    st.write("""
    **3. Enrichissement du dataset:**
    - Nombre d'entreprises par région
    - Informations sur le nombre de personnes sans diplôme, ou titulaires d'un CEP, ou d'un diplôme de niveau BAC+5 ou plus
    """)

    st.markdown("""
    Nous avons constaté qu'il y avait beaucoup de zéros dans les colonnes suivantes : 'E14TS6', 'E14TS10', 'E14TS20', 'E14TS50', 'E14TS100', 'E14TS200', 'E14TS500'. Pour améliorer cela, nous avons créé de nouvelles colonnes qui catégorisent les données en fonction de tailles d'entreprises plus vastes :
    
    ```python
    # Création de nouvelles colonnes pour les tailles d'entreprises
    df_final_merge2['nb_micro_entreprises'] = df_final_merge2[['ET_BE_0sal', 'ET_BE_1_4', 'ET_BE_5_9']].sum(axis=1)
    df_final_merge2['nb_small_entreprises'] = df_final_merge2[['ET_BE_10_19', 'ET_BE_20_49']].sum(axis=1)
    df_final_merge2['nb_medium_entreprises'] = df_final_merge2[['ET_BE_50_99', 'ET_BE_100_199']].sum(axis=1)
    df_final_merge2['nb_large_entreprises'] = df_final_merge2[['ET_BE_200_499', 'ET_BE_500P']].sum(axis=1)
    ```
    """)


    st.write("""
    **4. Fusion des jeux de données:**
    - Intégration des différentes sources de données
    """)

    st.markdown("""
    Nous avons fusionné les différents jeux de données (entreprises, informations géographiques, salaires, population, diplômes) sur la colonne CODGEO pour créer un dataset complet et intégré.
    ```python
    # Fusion des jeux de données sur la colonne CODGEO
    df_merge1=df_entreprises24.merge(df_name_geographic_final,left_on='CODGEO',right_on='COM_code_insee')
    df_merge2=df_merge1.merge(df_salary,left_on='CODGEO',right_on='CODGEO')
    df_merge3=df_merge2.merge(df_population,left_on='CODGEO',right_on='CODGEO')
    df_merge4=df_merge3.merge(df_diploma,left_on='CODGEO',right_on='CODGEO')
    ```
                
    Après la fusion, nous avons supprimé les colonnes inutilisées avant de procéder à l'apprentissage automatique pour optimiser notre dataset.

    ```python
    # Suppression des colonnes inutilisées
    df_final_merge2 = df_final_merge2.drop(columns=['NIVGEO', 'LIBGEO', 'MOCO', 'AGEQ80_17', 'SEXE', 'NB'])  
    df_final_merge2 = df_final_merge2.drop_duplicates().reset_index(drop=True)
    ```

    """)

    st.write("""
    **5. Standardisation des variables numériques:**
    """)

    st.markdown("""
    Nous avons standardisé les variables numériques pour les mettre sur une échelle comparable, facilitant ainsi les analyses ultérieures comme le clustering.
    
    ```python
    from sklearn.preprocessing import StandardScaler

    # Sélection des colonnes numériques à standardiser
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

    sc = StandardScaler()
    df_selected = df_final_merge2[features_num_selected].reset_index(drop=True) 

    df_selected_scaled = sc.fit_transform(df_selected)
    df_scaled_df = pd.DataFrame(df_selected_scaled, columns = features_num_selected)
    ```
    """)

    st.write("### Résumé et Résultats")

    st.write("#### Résultats du nettoyage et de la transformation des données")
    st.write("Après le nettoyage et la transformation des données, nous avons obtenu un jeu de données prêt pour le Machine Learning")

    if st.checkbox("Afficher la table sommaire du DataFrame 'Final Merge'"):
        st.dataframe(summary(df_final_merge2))

    st.write("#### Résultats de la standardisation des données")
    st.write("Les données standardisées fournissent une base solide pour les méthodes d'analyse avancées telles que le clustering.")

    if st.checkbox("Afficher un aperçu de la table 'Final Merge' standardisée") :  
        st.dataframe(df_scaled_df.head())


   


#### MODELISATION ####


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


#@st.cache_data
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


###### Analyse ######
features_num_selected_acp_criteria = ['Total_Salaries', 'nb_auto_entrepreneur',
       'nb_micro_entreprises', 'nb_small_entreprises', 'nb_medium_entreprises',
       'nb_large_entreprises', 'mean_net_salary_hour_overall',
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

# Calculate cluster means for the best algo
cluster_means_acp_kmeans_optimised = df_final_merge2.groupby('Cluster-ACP-KMeans-best')[features_num_selected_acp_criteria].mean().reset_index()



# Extract data for Paris (Cluster 3) and the neighboring departments (they are clusters 0, 1, and 2)
# Extract data for Paris and neighboring departments
paris_data = get_cluster_data(cluster_means_acp_kmeans_optimised,3)
hauts_de_seine_data = get_cluster_data(cluster_means_acp_kmeans_optimised,1)
seine_saint_denis_data = get_cluster_data(cluster_means_acp_kmeans_optimised,2)
val_de_marne_data = get_cluster_data(cluster_means_acp_kmeans_optimised,0)


# Define the human-readable names in French
human_readable_names = {
    'Total_Salaries': 'Nombre Total d\'Entreprises',
    'nb_auto_entrepreneur': 'Nombre d\'Auto-Entrepreneurs',
    'nb_micro_entreprises': 'Micro-Entreprises',
    'nb_small_entreprises': 'Petites Entreprises',
    'nb_medium_entreprises': 'Moyennes Entreprises',
    'nb_large_entreprises': 'Grandes Entreprises',
    'mean_net_salary_hour_overall': 'Salaire net moyen par heure',
    'mean_net_salary_hour_executives': 'Salaire net moyen par heure pour les cadres',
    'mean_net_salary_hour_avg_executive': 'Salaire net moyen par heure pour un cadre moyen',
    'mean_net_salary_hour_employee': 'Salaire net moyen par heure pour l\'employé',
    'mean_net_salary_hour_worker': 'Salaire net moyen par heure pour le travailleur',
    'mean_net_salary_hour_female': 'Salaire net moyen pour les femmes',
    'mean_net_salary_hour_female_executives': 'Salaire net moyen par heure pour les cadres féminins',
    'mean_net_salary_hour_avg_female_executive': 'Salaire net moyen par heure pour les cadres moyens féminins',
    'mean_net_salary_hour_female_employee': 'Salaire net moyen par heure pour une employée',
    'mean_net_salary_hour_female_worker': 'Salaire net moyen par heure pour une travailleuse',
    'mean_net_salary_hour_male': 'Salaire net moyen pour un homme',
    'mean_net_salary_hour_male_executives': 'Salaire net moyen par heure pour un cadre masculin',
    'mean_net_salary_hour_avg_male_executive': 'Salaire net moyen par heure pour les cadres moyens masculins',
    'mean_net_salary_hour_male_employee': 'Salaire net moyen par heure pour un employé masculin',
    'mean_net_salary_hour_male_worker': 'Salaire net moyen par heure pour un travailleur masculin',
    'mean_net_salary_hour_18_25': 'Salaire net moyen par heure pour les 18-25 ans',
    'mean_net_salary_hour_26_50': 'Salaire net moyen par heure pour les 26-50 ans',
    'mean_net_salary_hour_over_50': 'Salaire net moyen par heure pour les plus de 50 ans',
    'mean_net_salary_hour_female_18_25': 'Salaire net moyen par heure pour les femmes âgées de 18 à 25 ans',
    'mean_net_salary_hour_female_26_50': 'Salaire net moyen par heure pour les femmes âgées de 26 à 50 ans',
    'mean_net_salary_hour_female_over_50': 'Salaire net moyen par heure pour les femmes de plus de 50 ans',
    'mean_net_salary_hour_male_18_25': 'Salaire net moyen par heure pour les hommes âgés de 18 à 25 ans',
    'mean_net_salary_hour_male_26_50': 'Salaire net moyen par heure pour les hommes âgés de 26 à 50 ans',
    'mean_net_salary_hour_male_over_50': 'Salaire net moyen par heure pour les hommes de plus de 50 ans',
    'men_over_15_no_diploma': 'Hommes de plus de 15 ans sans diplôme',
    'men_over_15_bac_plus_5_gratuated': 'Hommes de plus de 15 ans avec Bac+5',
    'women_over_15_no_diploma': 'Femmes de plus de 15 ans sans diplôme',
    'women_over_15_bac_plus_5_gratuated': 'Femmes de plus de 15 ans avec Bac+5'
}

# Define criteria for each category
enterprise_criteria = [
    'Total_Salaries', 'nb_auto_entrepreneur', 'nb_micro_entreprises', 
    'nb_small_entreprises', 'nb_medium_entreprises', 'nb_large_entreprises'
]

salary_criteria = [
    'mean_net_salary_hour_overall', 'mean_net_salary_hour_executives', 
    'mean_net_salary_hour_avg_executive', 'mean_net_salary_hour_employee', 
    'mean_net_salary_hour_worker', 'mean_net_salary_hour_female', 
    'mean_net_salary_hour_female_executives', 'mean_net_salary_hour_avg_female_executive', 
    'mean_net_salary_hour_female_employee', 'mean_net_salary_hour_female_worker', 
    'mean_net_salary_hour_male', 'mean_net_salary_hour_male_executives', 
    'mean_net_salary_hour_avg_male_executive', 'mean_net_salary_hour_male_employee', 
    'mean_net_salary_hour_male_worker', 'mean_net_salary_hour_18_25', 
    'mean_net_salary_hour_26_50', 'mean_net_salary_hour_over_50', 
    'mean_net_salary_hour_female_18_25', 'mean_net_salary_hour_female_26_50', 
    'mean_net_salary_hour_female_over_50', 'mean_net_salary_hour_male_18_25', 
    'mean_net_salary_hour_male_26_50', 'mean_net_salary_hour_male_over_50'
]

education_criteria = [
    'men_over_15_no_diploma', 'men_over_15_bac_plus_5_gratuated', 
    'women_over_15_no_diploma', 'women_over_15_bac_plus_5_gratuated'
]

# Extract data for each category
def extract_comparison_data(criteria):
    comparison_data = {
        'Critères': [human_readable_names[crit] for crit in criteria],
        'Paris': paris_data[criteria].values,
        'Hauts-de-Seine': hauts_de_seine_data[criteria].values,
        'Seine-Saint-Denis': seine_saint_denis_data[criteria].values,
        'Val-de-Marne': val_de_marne_data[criteria].values
    }
    comparison_df = pd.DataFrame(comparison_data)
    for col in ['Hauts-de-Seine', 'Seine-Saint-Denis', 'Val-de-Marne']:
        comparison_df[col] = comparison_df.apply(lambda row: add_arrow(row[col], row['Paris']), axis=1)
    return comparison_df

# Extract data for each category
def extract_comparison_data_without_arrows(criteria):
    comparison_data = {
        'Critères': [human_readable_names[crit] for crit in criteria],
        'Paris': paris_data[criteria].values,
        'Hauts-de-Seine': hauts_de_seine_data[criteria].values,
        'Seine-Saint-Denis': seine_saint_denis_data[criteria].values,
        'Val-de-Marne': val_de_marne_data[criteria].values
    }
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


# Create comparison tables
enterprise_df = extract_comparison_data(enterprise_criteria)
salary_df = extract_comparison_data(salary_criteria)
education_df = extract_comparison_data(education_criteria)

# Create comparison tables only with numbers
enterprise_df_numbers = extract_comparison_data_without_arrows(enterprise_criteria)
salary_df_numbers = extract_comparison_data_without_arrows(salary_criteria)
education_df_numbers = extract_comparison_data_without_arrows(education_criteria)

# Convert DataFrames to HTML
enterprise_html = enterprise_df.to_html(escape=False, index=False)
salary_html = salary_df.to_html(escape=False, index=False)
education_html = education_df.to_html(escape=False, index=False)

# st.dataframe(cluster_means_acp_kmeans_optimised.columns)

# Données du tableau
data = {
    "Indice": ["Silhouette Score", "Indice de Calinski-Harabasz"],
    "Description": [
        "Mesure à la fois la cohésion des clusters (proximité des points au sein d'un même cluster) et la séparation entre les clusters (distance entre différents clusters).",
        "Évalue la compacité et la séparation des clusters. Calculé en utilisant le rapport entre la somme de la dispersion entre les clusters et la somme de la dispersion au sein des clusters."
    ],
    "Interprétation": [
        "Variant de -1 à 1, des valeurs plus élevées indiquent des clusters bien formés et distincts, facilitant ainsi l'interprétation des résultats.",
        "Un score élevé indique que les clusters sont denses et bien séparés."
    ]
}

# Création du DataFrame
df_metric = pd.DataFrame(data)


# Données du tableau
data_methods = {
    "Method": [
        "KMeans without ACP", "KMeans after ACP", "KMeans after ACP-2", 
        "KMeans after TSNE", "AgglomerativeClustering after TSNE", 
        "AgglomerativeClustering after ACP", "KMeans after UMAP"
    ],
    "Calinski-Harabasz Index": [
        1375.163267, 2161.259580, 2161.311531, 3054.531292, 
        2274.970079, 1951.461117, 6371.374481
    ],
    "Silhouette Score": [
        0.275796, 0.476795, 0.476796, 0.294976, 0.227908, 0.282589, 0.422500
    ]
}

df_methods = pd.DataFrame(data_methods)


# # Visualization of Calinski-Harabasz scores
# fig = go.Figure()



if page == pages[5]: 
    st.write("### Analyse des Résultats de la Modélisation") 

    st.markdown('''
    Pendant notre analyse, la méthode de clustering retenue est KMeans avec Analyse en Composantes Principales (ACP).
                
    **Justification du Choix de la Méthode**

    Les métriques de performance utilisées pour comparer ces méthodes étaient le Silhouette Score et l'Indice de Calinski-Harabasz.
    ''')


    # Affichage du tableau des scores dans Streamlit
    # Convert DataFrame to HTML 
    st.markdown(df_metric.to_html(index=False, justify='center'), unsafe_allow_html=True)

    # Affichage des tableaux dans Streamlit
    st.write('''
             
    **Comparaison des performances des méthodes de clustering:**
             
    Nous avons appliqué ces indices aux résultats des différentes méthodes de clustering et comparé les scores obtenus pour déterminer quelle méthode offre la meilleure performance.  ''' )
    st.dataframe(df_methods)


    st.markdown('''

    - **KMeans après ACP** présente les meilleurs résultats avec un score de silhouette de 0.476798, indiquant une meilleure cohésion interne et une meilleure séparation des clusters.
    - **KMeans après UMAP** offre les meilleurs résultats en termes d'indice de Calinski-Harabasz avec une valeur de 6371.374481, indiquant une très bonne formation des clusters.
                
    Cependant, pour obtenir un bon équilibre entre la cohésion interne et la séparation des clusters, **KMeans après ACP** semble être la meilleure option grâce à son score de silhouette supérieur.

    **Évaluation des combinaisons de dimensions réduites et nombres de clusters:**
    
    Nous avons testé différentes combinaisons de composantes principales (n_components) et de nombres de clusters (n_clusters) en utilisant l'Analyse en Composantes Principales (ACP) suivie de KMeans pour le clustering. Les performances de ces combinaisons ont été évaluées à l'aide de deux métriques : le Silhouette Score et le Calinski-Harabasz Score. Les résultats ont été visualisés pour aider à déterminer les paramètres optimaux.
    ''')

    param_grid = {'n_components': [2, 3, 4, 5]}

    # Visualization of Silhouette scores
    fig1 = go.Figure()

    for n_components in param_grid['n_components']:
        subset = results_df[results_df['n_components'] == n_components]
        fig1.add_trace(go.Scatter(x=subset['n_clusters'], y=subset['silhouette_score'], mode='lines+markers', name=f'n_components={n_components}'))

    fig1.update_layout(
        title='Silhouette Score vs. Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Silhouette Score'
    )

    # Visualization of Calinski-Harabasz scores
    fig2 = go.Figure()

    for n_components in param_grid['n_components']:
        subset = results_df[results_df['n_components'] == n_components]
        fig2.add_trace(go.Scatter(x=subset['n_clusters'], y=subset['calinski_harabasz_score'], mode='lines+markers', name=f'n_components={n_components}'))

    fig2.update_layout(
        title='Calinski-Harabasz Score vs. Number of Clusters',
        xaxis_title='Number of Clusters',
        yaxis_title='Calinski-Harabasz Score'
    )

    # Display the plots within the Streamlit app
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    
    st.markdown('''
    Les meilleures performances sont obtenues avec **n_components=2 et n_clusters=3, 4, 5, ou 6**. 
                
    Nous choisissons la combinaison **n_components=2 et n_clusters=5** pour un bon équilibre entre cohésion et séparation des clusters, comme indiqué par les scores Silhouette et Calinski-Harabasz.''')

    st.markdown("### Comparaison entre Paris et la petite couronne:")
    st.markdown("#### Comparaison des entreprises")
    st.markdown(enterprise_html, unsafe_allow_html=True)


    # DataViz pour entreprises
    fig = go.Figure()

    for region in ["Paris", "Hauts-de-Seine", "Seine-Saint-Denis", "Val-de-Marne"]:
        fig.add_trace(go.Bar(
            x=enterprise_df_numbers["Critères"],
            y=enterprise_df_numbers[region],
            name=region
        ))

    # Ajouter des titres et des labels
    fig.update_layout(
        title="Comparaison des entreprises",
        xaxis_title="Critères",
        yaxis_title="Nombre d'entreprises",
        barmode='group',
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig)

    st.markdown("#### Comparaison des salaires")
    
    # Adding scroll to the table
    html_with_scroll = f"""
                        <div style="height:300px; overflow:auto;">
                            {salary_html}
                        </div>
    
                       """
    # Show table
    st.markdown(html_with_scroll, unsafe_allow_html=True)
    
    # DataViz Salaire homme/femme basé sur les catégories d'emploi, age
    ##DataViz par genre
    plot_salary_comparison_by_region_and_gender(salary_df_numbers)

    ##DataViz par genre et categorie emploie
    categories = ['Travailleur', 'Employé', 'Cadre moyen', 'Cadre']
    selected_category = st.selectbox("Choix du graphique", categories)
    
    plot_salary_comparison_by_category(salary_df_numbers, selected_category)

    ## DataViz par emploie
    plot_salary_comparison_by_job_category(salary_df_numbers)

    ##DataViz par age group
    plot_salary_comparison_by_age_group(salary_df_numbers)


    ## Test ## 

    st.markdown("#### Comparaison des niveaux d'éducation")
    st.markdown(education_html, unsafe_allow_html=True)
    
    plot_education_comparison(education_df_numbers)

     ## Test finished## 
    
    st.image("img/map_kmeans_acp_result.png", caption="Cluster Map", use_column_width=True)

    st.markdown(''' 
    **Résultats:**

    Voici les résultats de la visualisation des clusters avec ces paramètres optimaux :
    
    ![Cluster Map]("img/map_kmeans_acp_result.png")
                TO DO MAP HERE 
                

    **Analyse des résultats:**

    - Le nombre d'entreprises varie significativement selon la région, avec Paris ayant un nombre beaucoup plus élevé de micro-entreprises comparé à la petite couronne.
    - Les disparités salariales selon le genre et l'âge sont également marquées, avec des salaires moyens plus élevés à Paris, particulièrement pour les cadres masculins.
    - Les raisons de ces disparités peuvent être multiples, incluant des facteurs économiques, sociaux et culturels spécifiques à chaque région.
                
                
    Ces analyses montrent des inégalités significatives entre Paris et la petite couronne, mettant en lumière des disparités socio-économiques influencées par divers facteurs régionaux.

    En conclusion, le choix de la méthode KMeans après ACP et des paramètres optimaux a permis une meilleure segmentation des départements selon leurs caractéristiques socio-économiques, fournissant ainsi une base solide pour l'analyse des inégalités régionales.
                
                ''')


if page == pages[6]: 
    st.write("### Conclusion")