import sys
import subprocess
import streamlit as st

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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio

# Import modules from the appended path
from fi_functions import *

# Import the data
@st.cache_data
def load_data():

    # read .csv form Google Drive
    #file_id_entreprise = '1tP5j1NU6cT5kiEypf7ejQKrxNCrD4Cov' # .csv file
    #file_id_salary = '1NLw8ymnnzLONUM1IVrYsH_A7DsqT84df' # .csv file
    #file_id_name_geographic = '1rgltgPmoDDzNT-YWYRvc11isdgmSmeAG' # .csv file
    #file_id_diploma = '11-KEM4rJ2PqxsTn-BxW1pOfJobx6jjuv' # .csv file

   
    # Construct the direct download link
    # download_link_entreprise = f'https://drive.google.com/uc?export=download&id={file_id_entreprise}'
    # download_link_salary = f'https://drive.google.com/uc?export=download&id={file_id_salary}'
    # download_link_name_geographic = f'https://drive.google.com/uc?export=download&id={file_id_name_geographic}'
    # download_link_diploma = f'https://drive.google.com/uc?export=download&id={file_id_diploma}'

    # # Read df
    # df_entreprises=pd.read_csv(download_link_entreprise, dtype={'CODGEO': object})     # OK  : from .csv file in google drive
    # df_salary = pd.read_csv(download_link_salary, sep = ';')   # OK  : from .csv file in google drive
    # df_name_geographic=pd.read_csv(download_link_name_geographic, dtype={0: 'object'})   # OK  : from .csv file in google drive
    # df_population=pd.read_parquet('data/population.parquet.gzip')  # OK  : from .csv file in Jupyter

    # df_diploma=pd.read_csv(download_link_diploma, sep=';')     # OK  : from .csv file in google drive

    #df_merge = pd.read_parquet('data/df_final_merge.parquet')

    df_final_merge2 = pd.read_csv('data/df_final_merge2.csv')
    df_scaled_df = pd.read_csv('data/df_scaled_df.csv')
    
    # return df_entreprises, df_salary, df_name_geographic, df_population, df_diploma,df_merge, df_final_merge2
    return df_final_merge2, df_scaled_df

df_final_merge2, df_scaled_df = load_data()


st.title("French Industry Project")
# Add the project logo (Replace 'logo_url' with your logo's URL)
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
                                name='Hommes', marker_color='blue'))
        fig_male.update_layout(
            title='Les 5 villes avec les salaires moyens les plus élevés pour les hommes (50+ ans) en Île-de-France',
            xaxis_title='Noms des villes',
            yaxis_title='Salaire moyen',
            xaxis_tickangle=45
        )
        
        # Graphique pour les femmes
        fig_female = go.Figure()
        fig_female.add_trace(go.Bar(x=top_5_salaries_female['COM_name'], y=top_5_salaries_female['mean_net_salary_hour_female_over_50'], 
                                    name='Femmes', marker_color='red'))
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
if page == pages[3] : 
    st.write("### Préparation des données")

if page == pages[4] : 
    st.write("### Modélisation")

if page == pages[5] : 
    st.write("### Conclusion")
