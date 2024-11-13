import numpy as np
import pandas as pd
from dash import Dash, _dash_renderer, Input, Output, html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go

# preprocess the dataset
df2 = pd.read_csv('./datasets/dataset.csv')
df2.dropna(subset=['Year', 'Rating', 'Genre'], inplace=True) # drop the nan value
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
# Country

#Create the line fig
df_line = df2.drop_duplicates(subset=['Year', 'Genre'])
df_line_grouped = df2.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()
fig_line = px.line(df_line_grouped, x='Year', y='Rating', color='Genre', title='Movie Popularity by Genre Over Time')

# create stream fig
df_stream = df2.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()
fig_stream = px.area(df_stream, x='Year', y='Rating', color='Genre', title='Movie Popularity by Genre Over Time')

# create sankey fig
# Filter the data to include only data from 2020 onwards
df_filtered = df2[df2['Year'] >= 2020]

# Prepare the data
df_sankey = df_filtered.groupby(['Year', 'Genre'])['Rating'].mean().reset_index()

# Create lists for source, target, and value
years = sorted(df_sankey['Year'].unique())
genres = df_sankey['Genre'].unique()

source = []
target = []
value = []

for i in range(len(years) - 1):
    year_current = years[i]
    year_next = years[i + 1]
    
    df_current = df_sankey[df_sankey['Year'] == year_current]
    df_next = df_sankey[df_sankey['Year'] == year_next]
    
    for genre in genres:
        rating_current = df_current[df_current['Genre'] == genre]['Rating'].values
        rating_next = df_next[df_next['Genre'] == genre]['Rating'].values
        
        if len(rating_current) > 0 and len(rating_next) > 0:
            source.append(f"{year_current} - {genre}")
            target.append(f"{year_next} - {genre}")
            value.append(abs(rating_next[0] - rating_current[0]))

# Create a list of all unique labels
labels = list(set(source + target))

# Create a mapping from labels to indices
label_indices = {label: i for i, label in enumerate(labels)}

# Map source and target labels to indices
source_indices = [label_indices[label] for label in source]
target_indices = [label_indices[label] for label in target]

# Define a color palette for genres
color_palette = px.colors.qualitative.Plotly
genre_colors = {genre: color_palette[i % len(color_palette)] for i, genre in enumerate(genres)}

# Map colors to nodes
node_colors = []
for label in labels:
    genre = label.split(" - ")[1]
    node_colors.append(genre_colors[genre])

# Create the Sankey diagram
fig_sankey = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
        color=node_colors
    ),
    link=dict(
        source=source_indices,
        target=target_indices,
        value=value
    )
)])

#Create the treemap fig
fig_one = px.treemap(names = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],
    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"])
fig_one.update_traces(root_color="lightgrey")
fig_one.update_layout(margin = dict(t=50, l=25, r=25, b=25))

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig_two = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

fig_three = px.pie(names=['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen'], values=[4500, 2500, 1053, 500], title="Atmosphere Composition")

#initialize the app
app = Dash()

#app layout
app.layout = dmc.Container([
    dmc.Title('Movie and TV Shows Data Visualiztaion', color="blue", size="h3"),
    dmc.Title("COMP4462 Group 8 (Daisy Har, Aatrox Deng, Lyam Tang)", size ="h6" ),
    html.Div(className='row', children=[
        dmc.Title("Dataset", size="h10"),
        dash_table.DataTable(data=df2.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'})
    ]),
    dcc.Graph(id="graph-sankey", figure=fig_sankey),
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

