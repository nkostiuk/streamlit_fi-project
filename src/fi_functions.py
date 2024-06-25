# french_industry library
import pandas as pd
import numpy as np

def summary(df):
    """
    Provides a summary of the DataFrame with information about the number of unique values, 
    percentage of missing values, data type of each column, mean or mode depending on the data type,
    and potential alerts such as too many missing values or imbalanced categories.
    
    Parameters:
    df (DataFrame): The DataFrame for which the summary is required.
    
    Returns:
    None
    """
    
    table = pd.DataFrame(
        index=df.columns,
        columns=['type_info', '%_missing_values', 'nb_unique_values', 'nb_zero_values','%_zero_values', 'list_unique_values', "mean_or_mode", "flag"])
    
    # Fill in column 'type_info' with data types of each column
    table.loc[:, 'type_info'] = df.dtypes.values
    
    # Calculate the percentage of missing values for each column
    table.loc[:, '%_missing_values'] = np.round((df.isna().sum().values / len(df)) * 100)
    
    # Calculate the number of unique values for each column
    table.loc[:, 'nb_unique_values'] = df.nunique().values

    # Calculate the number of 0 values for each column
    table.loc[:, 'nb_zero_values'] = df.isin([0]).sum().values

    # Calculate the percentage of 0 values for each column
    table.loc[:, '%_zero_values'] = np.round((df.isin([0]).sum().values/len(df)) * 100)
    
    # Function to get the list of unique values for a column, showing "Too many categories..." if there are too many
    def get_list_unique_values(colonne):
        if colonne.nunique() < 11:
            return colonne.unique()
        else:
            return "Too many categories..." if colonne.dtypes == "Object" else "Too many values..."
    
    # Function to get the mean or mode depending on the data type
    def get_mean_mode(colonne):
        if (colonne.dtypes == "O") or (len(colonne.unique()) < 20):
            return colonne.astype('category').mode()[0]
        else: 
            return colonne.mean()
    
    # Function to check for potential alerts such as too many missing values or imbalanced categories
    def alerts(colonne, thresh_na=0.25, thresh_balance=0.8):
        if (colonne.isna().sum()/len(df)) > thresh_na:
            return "Too many missing values!"
        elif colonne.value_counts(normalize=True).values[0] > thresh_balance:
            return "It's imbalanced!"
        else:
            return "Nothing to report"
        
    # Fill in the columns with the respective information
    table.loc[:, 'list_unique_values'] = df.apply(get_list_unique_values)
    table.loc[:, 'mean_or_mode'] = df.apply(get_mean_mode)
    table.loc[:, 'flag'] = df.apply(alerts)
    
    # Display the summary table
    return table

# Data type verification  
def detection_error(element):
    try:
        float(element)
    except ValueError:
        return element


# Function that provides a short summary of DataFrame with information about the number of unique values, 
# percentage of missing values, number of missing values, and the data type of each column.

def summary_short(df):
    """
    Function that provides a short summary of DataFrame with information about the number of unique values, 
    percentage of missing values, number of missing values, and the data type of each column.
    
    Is used with not cleaned DataFrames.
    
    Parameters:
    df (DataFrame): The DataFrame for which the summary is required.
    
    Returns:
    DataFrame with 
    """
    
    tab = pd.DataFrame(index = df.columns, columns = ['nb_unique_values','%_missing_values','nb_missing_values','type'])
    tab.loc[:,'nb_unique_values'] = df.nunique().values
    tab.loc[:,'%_missing_values'] = np.round((df.isna().sum().values / len(df)) * 100)
    tab.loc[:,'nb_missing_values'] = df.isna().sum().values
    tab.loc[:, 'type'] = df.dtypes.values
    
    return tab

# Convert column names in the DataFrame to lowercase
def convert_columns_to_lower(df):
    
    """
    Convert column names in the DataFrame to lowercase.
    
    Parameters:
        df (DataFrame): Input DataFrame.
        
    Returns:
        DataFrame: DataFrame with column names converted to lowercase.
    """
    
    df.columns = [col.lower() for col in df.columns]
    return df

# Get the unique lengths of strings in the specified column
def get_unique_lengths(column):
    """
    Get the unique lengths of strings in the specified column.
    
    Parameters:
        column (pandas.Series): The column containing strings.
        
    Returns:
        numpy.ndarray: An array containing the unique lengths of strings in the column.
    """
    # Calculate the length of each string in the column
    lengths = column.astype(str).str.len()
    
    # Find unique lengths
    unique_lengths = lengths.unique()
    
    return unique_lengths


from sklearn.preprocessing import StandardScaler

def standardize_numerical_columns(df, columns):
    """
    Standardizes specified numerical columns in the DataFrame using StandardScaler.

    Parameters:
        df (DataFrame): The DataFrame containing numerical columns to be standardized.
        columns (list): A list of numerical column names to be standardized.

    Returns:
        DataFrame: The DataFrame with specified numerical columns standardized.
    """
    # Initialize StandardScaler
    scaler = StandardScaler()

    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_standardized = df.copy()

    # Standardize specified numerical columns
    df_standardized[columns] = scaler.fit_transform(df_standardized[columns])

    return df_standardized

# Example usage:
# Specify the columns to be standardized
# columns_to_standardize = ['Total_Salaries', 'mean_net_salary_hour_overall', ...]  # Add columns here
# df_standardized = standardize_numerical_columns(df, columns_to_standardize)



# Function to create HTML for a team member
def create_team_member(name, photo_url, linkedin_url):
    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 15px;">
        <img src="{photo_url}" style="width: 50px; height: 50px; border-radius: 50%; margin-right: 15px;">
        <div>
            <p style="margin: 0; font-weight: bold;">{name}</p>
            <a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: #0e76a8;">
                LinkedIn
            </a>
        </div>
    </div>
    """


import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# Import the data
@st.cache_data
def load_data():
    df_final_merge2 = pd.read_csv('data/df_final_merge2.csv')
    df_scaled_df = pd.read_csv('data/df_scaled_df.csv')
    results_df = pd.read_csv('data/nb_comp_clust_results_df.csv') 

    return df_final_merge2, df_scaled_df, results_df


@st.cache_data
def plot_moyen_salary_by_region(df):
    fig = px.box(df, x='REG_nom', y='mean_net_salary_hour_overall', title='Salaire net moyen par heure par région', labels={'REG_nom': 'Région', 'mean_net_salary_hour_overall': 'Salaire Net Moyen Par Heure'})
    fig.update_layout(xaxis_tickangle=-45)
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
    fig.update_layout(barmode='overlay', title='Distribution des salaires net moyens par heure entre femmes et hommes de plus de 50 ans dans l\'industrie en France', xaxis_title='Salaire Net Moyen Par Heure', yaxis_title='Pourcentage', bargap=0.1)
    fig.update_traces(opacity=0.75)
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

@st.cache_data
def plot_average_salaries_over_50_by_region(df):
    df_region_salary = df.groupby('REG_nom').agg({'mean_net_salary_hour_female_over_50': 'mean', 'mean_net_salary_hour_male_over_50': 'mean'}).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_region_salary['REG_nom'], y=df_region_salary['mean_net_salary_hour_female_over_50'], name='Femmes', marker_color='purple'))
    fig.add_trace(go.Bar(x=df_region_salary['REG_nom'], y=df_region_salary['mean_net_salary_hour_male_over_50'], name='Hommes', marker_color='orange'))
    fig.update_layout(barmode='group', title='Salaires net moyens par heure des femmes et des hommes de plus de 50 ans par région', xaxis_title='Région', yaxis_title='Salaire Net Moyen Par Heure')
    return fig

@st.cache_data
def plot_average_salaries_by_category_and_region(df):
    agg_df = df.groupby('REG_nom').agg({
        'mean_net_salary_hour_overall': 'mean',
        'mean_net_salary_hour_executives': 'mean',
        'mean_net_salary_hour_avg_executive': 'mean',
        'mean_net_salary_hour_employee': 'mean',
        'mean_net_salary_hour_worker': 'mean',
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_overall'], name='Salaire net moyen par heure', marker_color='blue'))
    fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_executives'], name='Salaire net moyen par heure pour les cadres', marker_color='red'))
    fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_avg_executive'], name='Salaire net moyen par heure pour un cadre moyen', marker_color='green'))
    fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_employee'], name='Salaire net moyen par heure pour l\'employé', marker_color='orange'))
    fig.add_trace(go.Bar(x=agg_df['REG_nom'], y=agg_df['mean_net_salary_hour_worker'], name='Salaire net moyen par heure pour le travailleur', marker_color='purple'))

    # Ajouter la ligne horizontale pour le SMIC
    fig.add_hline(y=10, line=dict(color="black", width=3), annotation_text="SMIC", annotation_position="top right")


    fig.update_layout(
        title='Salaires moyens par catégorie et région',
        xaxis_title='Région',
        yaxis_title='Salaire net moyen par heure',
        xaxis_tickangle=-45,
        barmode='group',
        legend_title='Catégorie',
        annotations=[dict(
            x=0.5,
            y=10,
            xref='paper',
            yref='y',
            text='SMIC',
            showarrow=False,
            font=dict(
                size=12,
                color="black"
            ),
            bgcolor="white",
            opacity=0.8
        )]
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
                    labels={'COM_name': 'Noms des villes', 'mean_net_salary_hour_overall': 'Salaire Net Moyen Par Heure'},
                    color='mean_net_salary_hour_overall',
                    color_continuous_scale='Blues')
    fig.update_layout(xaxis_tickangle=45)
    return fig

@st.cache_data
def plot_top_5_salaries_idf_comparison(df):
    idf_data = df[df['REG_nom'] == 'Île-de-France']
    
    idf_data_sorted_male = idf_data.sort_values(by='mean_net_salary_hour_male_over_50', ascending=False)
    top_salaries_unique_male = idf_data_sorted_male.drop_duplicates(subset=['mean_net_salary_hour_male_over_50'], keep='first')
    top_5_salaries_male = top_salaries_unique_male.head(5)
    
    idf_data_sorted_female = idf_data.sort_values(by='mean_net_salary_hour_female_over_50', ascending=False)
    top_salaries_unique_female = idf_data_sorted_female.drop_duplicates(subset=['mean_net_salary_hour_female_over_50'], keep='first')
    top_5_salaries_female = top_salaries_unique_female.head(5)
    
    fig_male = go.Figure()
    fig_male.add_trace(go.Bar(x=top_5_salaries_male['COM_name'], y=top_5_salaries_male['mean_net_salary_hour_male_over_50'], 
                            name='Hommes', marker_color='orange'))
    fig_male.update_layout(
        title='Les 5 villes avec les salaires moyens les plus élevés pour les hommes (50+ ans) en Île-de-France',
        xaxis_title='Noms des villes',
        yaxis_title='Salaire net moyen par heure',
        xaxis_tickangle=45
    )
    
    fig_female = go.Figure()
    fig_female.add_trace(go.Bar(x=top_5_salaries_female['COM_name'], y=top_5_salaries_female['mean_net_salary_hour_female_over_50'], 
                                name='Femmes', marker_color='purple'))
    fig_female.update_layout(
        title='Les 5 villes avec les salaires moyens par heure les plus élevés pour les femmes (50+ ans) en Île-de-France',
        xaxis_title='Noms des villes',
        yaxis_title='Salaire net moyen par heure',
        xaxis_tickangle=45
    )
    
    return fig_male, fig_female

import matplotlib.pyplot as plt

@st.cache_data
def plot_scatter(data, labels, title='Scatter Plot', x_label='X-axis', y_label='Y-axis', color_map='viridis'):
    plt.figure(figsize=(10, 4))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=color_map)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.colorbar(label='Cluster')
    st.pyplot(plt)

from scipy.cluster.hierarchy import dendrogram, linkage

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


# Add arrow indicators with colored formatting
def add_arrow(value, reference):
    if value > reference:
        return f"{value:.2f} <span style='color:green;'>↑</span>"
    elif value < reference:
        return f"{value:.2f} <span style='color:red;'>↓</span>"
    else:
        return f"{value:.2f} -"
    
# Function to extract data for each cluster
def get_cluster_data(cluster_means_acp_kmeans_optimised, cluster_number):
    return cluster_means_acp_kmeans_optimised[cluster_means_acp_kmeans_optimised['Cluster-ACP-KMeans-best'] == cluster_number].iloc[0]



# Function to plot scores
def plot_scores(results_df, score_column, title, ylabel):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for n_components in results_df['n_components'].unique():
        subset = results_df[results_df['n_components'] == n_components]
        ax.plot(subset['n_clusters'], subset[score_column], marker='o', label=f'n_components={n_components}')
    ax.set_title(title)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel(ylabel)
    ax.legend()
    return fig




# def plot_salary_comparison_by_category(df, category):
#     # Define the relevant columns based on the selected category
#     if category == 'Travailleur':
#         cols = [9, 14]  # Columns for "Salaire net moyen par heure pour une travailleuse" and "Salaire net moyen par heure pour un travailleur masculin"
#         title = "Comparaison des salaires par région pour les Travailleurs"
#     elif category == 'Employé':
#         cols = [8, 13]  # Columns for "Salaire net moyen par heure pour une employée" and "Salaire net moyen par heure pour un employé masculin"
#         title = "Comparaison des salaires par région pour les Employés"
#     elif category == 'Cadre moyen':
#         cols = [7, 12]  # Columns for "Salaire net moyen par heure pour les cadres moyens féminins" and "Salaire net moyen par heure pour les cadres moyens masculins"
#         title = "Comparaison des salaires par région pour les Cadres Moyens"
#     elif category == 'Cadre':
#         cols = [6, 11]  # Columns for "Salaire net moyen par heure pour les cadres féminins" and "Salaire net moyen par heure pour un cadre masculin"
#         title = "Comparaison des salaires par région pour les Cadres"
#     else:
#         st.error("Invalid category selected.")
#         return
    
#     df_simplified = df.iloc[cols].copy().transpose().reset_index()
#     df_simplified.columns = ["Région", f"Salaire net moyen par heure pour une {category}e", f"Salaire net moyen par heure pour un {category} masculin"]
    
#     df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"
    
#     # Melting the dataframe to make it suitable for plotly express
#     df_melted = df_simplified.melt(id_vars="Région", var_name="Catégorie", value_name="Salaire")
    
#     # Visualization using plotly
#     fig = px.bar(
#         df_melted,
#         x="Région",
#         y="Salaire",
#         color="Catégorie",
#         color_discrete_map={
#             f"Salaire net moyen par heure pour une {category}e": "purple",
#             f"Salaire net moyen par heure pour un {category} masculin": "orange"
#         },
#         barmode="group",
#         title=title,
#         labels={"Salaire": "Salaire (Euro)"}
#     )
    
#     st.plotly_chart(fig)

# Dictionary to handle gender-specific endings
gender_specific_endings = {
    "Travailleur": "Travailleuse",
    "Employé": "Employée",
    "Cadre moyen": "Cadre moyen féminin",
    "Cadre": "Cadre féminin"
}


def plot_salary_comparison_by_category(df, category):
    # Get the correct female form and male form
    if category in gender_specific_endings:
        female_form = gender_specific_endings[category]
        if category == "Cadre moyen":
            male_form = "Cadre moyen masculin"
        elif category == "Cadre":
            male_form = "Cadre masculin"
        else:
            male_form = f"{category}"
    else:
        st.error("Invalid category selected.")
        return
    
    # Define the relevant columns based on the selected category
    if category == 'Travailleur':
        cols = [9, 14]  # Columns for "Salaire net moyen par heure pour une travailleuse" and "Salaire net moyen par heure pour un travailleur"
        title = "Comparaison des salaires par région pour les Travailleurs"
    elif category == 'Employé':
        cols = [8, 13]  # Columns for "Salaire net moyen par heure pour une employée" and "Salaire net moyen par heure pour un employé"
        title = "Comparaison des salaires par région pour les Employés"
    elif category == 'Cadre moyen':
        cols = [7, 12]  # Columns for "Salaire net moyen par heure pour les cadres moyens féminins" and "Salaire net moyen par heure pour les cadres moyens masculins"
        title = "Comparaison des salaires par région pour les Cadres Moyens"
    elif category == 'Cadre':
        cols = [6, 11]  # Columns for "Salaire net moyen par heure pour les cadres féminins" and "Salaire net moyen par heure pour un cadre"
        title = "Comparaison des salaires par région pour les Cadres"
    else:
        st.error("Invalid category selected.")
        return
    
    df_simplified = df.iloc[cols].copy().transpose().reset_index()
    df_simplified.columns = ["Région", f"Salaire net moyen par heure pour une {female_form}", f"Salaire net moyen par heure pour un {male_form}"]
    
    df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"
    
    # Melting the dataframe to make it suitable for plotly express
    df_melted = df_simplified.melt(id_vars="Région", var_name="Catégorie", value_name="Salaire")
    
    # Visualization using plotly
    fig = px.bar(
        df_melted,
        x="Région",
        y="Salaire",
        color="Catégorie",
        color_discrete_map={
            f"Salaire net moyen par heure pour une {female_form}": "purple",
            f"Salaire net moyen par heure pour un {male_form}": "orange"
        },
        barmode="group",
        title=title,
        labels={"Salaire": "Salaire (Euro)"}
    )
    
    st.plotly_chart(fig)
    



def plot_salary_comparison_by_job_category(df):
    job_categories = [1, 2, 3, 4]
    df_simplified = df.iloc[job_categories].copy().transpose().reset_index()
    df_simplified.columns = ["Region", "Cadres", "Cadre moyen", "Employé", "Travailleur"]

    df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"

    # Melting the dataframe to make it suitable for plotly express
    df_melted = df_simplified.melt(id_vars="Region", var_name="Job Category", value_name="Salary")

    # Visualization using plotly
    fig = px.bar(
        df_melted,
        x="Region",
        y="Salary",
        color="Job Category",
        barmode="group",
        title="Comparaison des salaires par la catégorie d'emploi",
        labels={"Salary": "Salary (Euro)"}
    )

    st.plotly_chart(fig)


def plot_salary_comparison_by_age_group(df):
    age_groups = [15, 16, 17]
    df_simplified = df.iloc[age_groups].copy().transpose().reset_index()
    df_simplified.columns = ["Region", "18-25 ans", "26-50 ans", "Plus de 50 ans"]

    df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"

    # Melting the dataframe to make it suitable for plotly express
    df_melted = df_simplified.melt(id_vars="Region", var_name="Age Group", value_name="Salary")

    # Visualization using plotly
    fig = px.bar(
        df_melted,
        x="Region",
        y="Salary",
        color="Age Group",
        barmode="group",
        title="Comparaison des salaires par groupe d'âge",
        labels={"Salary": "Salary (Euro)"}
    )

    st.plotly_chart(fig)


# def plot_average_hourly_wage(df):
#     df_simplified = df.iloc[[0]].copy().transpose().reset_index()
#     df_simplified.columns = ["Region", "Salaire net moyen par heure"]

#     df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"

#     # Visualization using plotly
#     fig = px.bar(
#         df_simplified,
#         x="Region",
#         y="Salaire net moyen par heure",
#         title="Average Hourly Wage by Region",
#         labels={"Salaire net moyen par heure": "Salary (Euro)"}
#     )

#     st.plotly_chart(fig)

def plot_salary_comparison_by_region_and_gender(df):
    df_simplified = df.iloc[[5, 10]].copy().transpose().reset_index()
    df_simplified.columns = ["Région", "Salaire net moyen pour les femmes", "Salaire net moyen pour les hommes"]
    
    df_simplified = df_simplified.iloc[1:]  # Remove the first row which corresponds to "Critères"

    # Melting the dataframe to make it suitable for plotly express
    df_melted = df_simplified.melt(id_vars="Région", var_name="Catégorie", value_name="Salaire")

    # Visualization using plotly
    fig = px.bar(
        df_melted,
        x="Région",
        y="Salaire",
        color="Catégorie",
        color_discrete_map={
            "Salaire net moyen pour les femmes": "purple",
            "Salaire net moyen pour les hommes": "orange"
        },
        barmode="group",
        title="Comparaison des salaires par région et genre",
        labels={"Salaire": "Salaire (Euro)"}
    )

    st.plotly_chart(fig)


def plot_education_comparison(df):
    """
    This function takes a DataFrame with education levels and creates a bar chart
    comparing the salary for different regions and education levels.
    """
    # Melting the dataframe to make it suitable for plotly express
    df_melted = df.melt(id_vars="Critères", var_name="Région", value_name="Salaire")

    # Visualization using plotly
    fig = px.bar(
        df_melted,
        x="Région",
        y="Salaire",
        color="Critères",
        barmode="group",
        title="Comparaison des niveaux d'éducation",
        labels={"Salaire": "Le nombre de personnes"}
    )

    st.plotly_chart(fig)