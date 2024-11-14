import numpy as np
import pandas as pd
from dash import Dash, _dash_renderer, Input, Output, html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go

# preprocess the dataset
df2 = pd.read_csv('./datasets/dataset.csv')
df2 = df2.assign(Genre=df2['Genre'].str.split(', ')).explode('Genre') # tokenise genre

#data summary

# Attributes:
# Poster
# Title
# Year
# Certificate
# Duration (min)
# Genre
# Rating
# Metascore
# Director
# Cast
# Votes
# Description
# Review Count
# Review Title
# Review
# Revenue
# Country

# #Create the line fig
# df_line = df2.drop_duplicates(subset=['Year', 'Genre'])
# df_line_grouped = df2.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()
# fig_line = px.line(df_line_grouped, x='Year', y='Rating', color='Genre', title='Movie Popularity by Genre Over Time')

# # create stream fig
# df_stream = df2.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()
# fig_stream = px.area(df_stream, x='Year', y='Rating', color='Genre', title='Movie Popularity by Genre Over Time')

# # create sankey fig
# # Filter the data to include only data from 2020 onwards
# df_filtered = df2[df2['Year'] >= 2020]

# # Prepare the data
# df_sankey = df_filtered.groupby(['Year', 'Genre']).agg(
#     Rating=('Rating', 'mean'),
#     Count=('Rating', 'size')
# ).reset_index()

# # Sort the data within each year group by the average rating
# df_sankey = df_sankey.sort_values(by=['Year', 'Rating'], ascending=[True, False])

# # Create lists for source, target, and value
# years = sorted(df_sankey['Year'].unique())
# genres = df_sankey['Genre'].unique()

# source = []
# target = []
# value = []

# # Dictionary to store the average score and count for each node
# node_scores = {}
# node_counts = {}

# for i in range(len(years) - 1):
#     year_current = years[i]
#     year_next = years[i + 1]
    
#     df_current = df_sankey[df_sankey['Year'] == year_current]
#     df_next = df_sankey[df_sankey['Year'] == year_next]
    
#     for genre in genres:
#         rating_current = df_current[df_current['Genre'] == genre]['Rating'].values
#         rating_next = df_next[df_next['Genre'] == genre]['Rating'].values
#         count_current = df_current[df_current['Genre'] == genre]['Count'].values
#         count_next = df_next[df_next['Genre'] == genre]['Count'].values
        
#         if len(rating_current) > 0 and len(rating_next) > 0:
#             source_label = f"{year_current} - {genre}"
#             target_label = f"{year_next} - {genre}"
            
#             source.append(source_label)
#             target.append(target_label)
#             value.append(abs(rating_next[0] - rating_current[0]))
            
#             # Store the average score and count for each node
#             node_scores[source_label] = rating_current[0]
#             node_scores[target_label] = rating_next[0]
#             node_counts[source_label] = count_current[0]
#             node_counts[target_label] = count_next[0]

# # Create a list of all unique labels
# labels = list(set(source + target))

# # Create a mapping from labels to indices
# label_indices = {label: i for i, label in enumerate(labels)}

# # Map source and target labels to indices
# source_indices = [label_indices[label] for label in source]
# target_indices = [label_indices[label] for label in target]

# # Define a color palette for genres
# color_palette = px.colors.qualitative.Plotly
# genre_colors = {genre: color_palette[i % len(color_palette)] for i, genre in enumerate(genres)}

# # Map colors to nodes
# node_colors = []
# for label in labels:
#     genre = label.split(" - ")[1]
#     node_colors.append(genre_colors[genre])

# # Create the hover template
# hover_template = "Genre: %{label}<br>Average Rating: %{customdata[0]:.2f}<br>Data Points: %{customdata[1]}<extra></extra>"

# # Sort labels and custom data by average rating within each year group
# sorted_labels = []
# sorted_customdata = []
# sorted_node_colors = []

# for year in years:
#     year_labels = [label for label in labels if label.startswith(f"{year} -")]
#     year_labels_sorted = sorted(year_labels, key=lambda x: node_scores[x], reverse=True)
#     sorted_labels.extend(year_labels_sorted)
#     sorted_customdata.extend([[node_scores[label], node_counts[label]] for label in year_labels_sorted])
#     sorted_node_colors.extend([genre_colors[label.split(" - ")[1]] for label in year_labels_sorted])

# # Create the Sankey diagram
# fig_sankey = go.Figure(data=[go.Sankey(
#     node=dict(
#         pad=15,
#         thickness=20,
#         line=dict(color="black", width=0.5),
#         label=sorted_labels,
#         color=sorted_node_colors,
#         customdata=sorted_customdata,
#         hovertemplate=hover_template
#     ),
#     link=dict(
#         source=[sorted_labels.index(label) for label in source],
#         target=[sorted_labels.index(label) for label in target],
#         value=value
#     )
# )])

# create parallel fig
df_parallel = df2[df2['Year'] >= 2020]
df_parallel = df_parallel[df_parallel['Genre'].isin(['Action', 'Comedy', 'Drama', 'Thriller'])]
# create color scale
genres = df_parallel['Genre'].unique()
genre_color = {
    'Action': '#FFA500',
    'Comedy': 'green',
    'Drama': 'red',
    'Thrilller': 'purple'
}
df_parallel['Color'] = df_parallel['Genre'].map(genre_color)
# px parallel fig
fig_parallel = px.parallel_coordinates(df_parallel, dimensions=['Year', 'Rating', 'Revenue'])
fig_parallel.update_layout(template='plotly_dark')

#Create the treemap fig
fig_one = px.treemap(names = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"])
fig_one.update_traces(root_color="lightgrey")
fig_one.update_layout(margin = dict(t=50, l=25, r=25, b=25), template='plotly_dark')


df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig_two = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
fig_two.update_layout(template='plotly_dark')


fig_three = px.pie(names=['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen'], values=[4500, 2500, 1053, 500], title="Atmosphere Composition")

# set the stylesheet
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'
]

#initialize the app
app = Dash(__name__, external_stylesheets=external_stylesheets)

#app layout
app.layout = dmc.Container([
    dmc.Title('Movie and TV Shows Data Visualiztaion', color="blue", size="h3"),
    dmc.Title("COMP4462 Group 8 (Daisy Har, Aatrox Deng, Lyam Tang)", size ="h6" ),
    html.Div([
        html.I(className='fas fa-search search-icon', style={'margin-right': '5px'}),
        dbc.Input(id='search-input', type='text', placeholder='Search...', style={'flex': '1', 'background-color': 'transparent', 'color': 'white'}),
    ], style={'display': 'flex', 'align-items': 'center'}),
    dcc.Graph(id="graph-parallel", figure=fig_parallel),
    dmc.Grid([
        dmc.Col([
            dcc.Graph(id='graph-one',figure=fig_one)
        ], span=6),
        dmc.Col([
            dcc.Graph(id="graph-two", figure=fig_two)
        ], span=6)
    ]),
    dmc.Grid([
        dmc.Col([
            dmc.Text("Dropdown selectors for Tree Map")
        ], span=6),
        dmc.Col([
            dmc.Text("Dropdown selectors for Choropleth Map")
        ], span=6)

    ]),
    dmc.Grid([
        dmc.Col([
            dcc.Dropdown(['Genres', 'Rating','Directors'], multi=True)
        ], span=6),
        dmc.Col([
            dcc.Dropdown(['Genres', 'Rating','Directors'])
        ], span=6)

    ])


], fluid=True)

#Add interactive buttons


#run app
if __name__ == '__main__':
    app.run(debug=True)

