import numpy as np
import pandas as pd
from scipy.stats import norm
from dash import Dash, _dash_renderer, Input, Output, html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go

# preprocess the dataset
df2 = pd.read_csv('./datasets/dataset.csv')
df2 = df2.assign(Genre=df2['Genre'].str.split(', ')).explode('Genre') # tokenise genre
df2 = df2.assign(Cast=df2['Cast'].str.split(', ')).explode('Cast') # tokenise cast
df2 = df2.assign(Director=df2['Director'].str.split(', ')).explode('Director') # tokenise director
df2 = df2.assign(Country=df2['Country'].str.split(', ')).explode('Country') # tokenise country
df2 = df2[df2['Revenue'] >= 0] # filter out movies with negative revenue
df2 = df2[df2['Year'] >= 2020] # filter out movies before 2020

country_iso_df = pd.read_csv('./datasets/countries_with_iso_codes.csv')
df3 = (df2['Country'].value_counts()).reset_index()
df3.columns = ['Country','Number of Production']
df3 = df3.merge(country_iso_df, on='Country', how='left')
df3 = df3[['Country','ISO_Code','Number of Production']]

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

def gen_parallel(df_filtered):
    # create parallel fig
    df_parallel = df_filtered[['Year', 'Genre', 'Rating', 'Metascore', 'Votes', 'Revenue']]
    # df_parallel = df_parallel[df_parallel['Genre'].isin(['Action', 'Comedy', 'Drama', 'Thriller'])]
    # print(df_parallel.columns)
    # px parallel fig
    fig_parallel = px.parallel_coordinates(df_parallel,
                                            dimensions=['Year', 'Rating', 'Revenue']
                                            )
    fig_parallel.update_layout(template='plotly_dark')

    return fig_parallel

fig_parallel = gen_parallel(df2)

def gen_bar(df_filtered):
    df_sorted = df_filtered.drop_duplicates(subset='Title', keep='first').sort_values(by='Revenue', ascending=False).head(10)
    fig_bar = px.bar(df_sorted, x='Revenue', y='Title', title='Top 10 Revenue Movies', orientation='h')

    fig_bar.update_yaxes(categoryorder='total ascending')
    fig_bar.update_layout(template='plotly_dark')

    return fig_bar

fig_bar = gen_bar(df2)

# star graph
avg_revenue = df2['Revenue'].mean()
sd_revenue = df2['Revenue'].std()

def gen_star(filtered):
    score_genre = norm.cdf((df2[df2['Genre'].isin(filtered['Genre'].unique())]['Revenue'].mean() - avg_revenue) / sd_revenue) * 10
    score_director = norm.cdf((df2[df2['Director'].isin(filtered['Director'].unique())]['Revenue'].mean() - avg_revenue) / sd_revenue) * 10
    score_cast = norm.cdf((df2[df2['Cast'].isin(filtered['Cast'].unique())]['Revenue'].mean() - avg_revenue) / sd_revenue) * 10
    score_country = norm.cdf((df2[df2['Country'].isin(filtered['Country'].unique())]['Revenue'].mean() - avg_revenue) / sd_revenue) * 10
    score_duration = norm.cdf((df2[df2['Duration (min)'].between(filtered['Duration (min)'].min(), filtered['Duration (min)'].max())]['Revenue'].mean() - avg_revenue) / sd_revenue) * 10
    scores = pd.DataFrame({
        'Attributes': ['Genre', 'Director', 'Cast', 'Country', 'Duration (min)'],
        'Values': [score_genre, score_director, score_cast, score_country, score_duration]
    })

    fig_star = px.line_polar(scores, r='Values', theta='Attributes', line_close=True)

    fig_star.update_traces(fill='toself')
    fig_star.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        template='plotly_dark'
    )

    return fig_star

fig_star = gen_star(df2)
    

#Create the treemap fig
def gen_treemap(df_filtered, primary_attr, secondary_attr, comparing_attr):
    # Group by primary and secondary attributes, averaging the comparing attribute
    df_treemap = df_filtered.groupby([primary_attr, secondary_attr])[comparing_attr].mean().reset_index()
    
    # Filter top 10 primary attribute groups based on the average comparing attribute
    top_primary = df_treemap.groupby(primary_attr)[comparing_attr].mean().nlargest(10).index
    df_treemap = df_treemap[df_treemap[primary_attr].isin(top_primary)]
    
    # Filter top 10 secondary attribute groups within each primary attribute group
    df_treemap = df_treemap.groupby(primary_attr).apply(lambda x: x.nlargest(10, comparing_attr)).reset_index(drop=True)
    
    # Prepare the data for the treemap: Each row should have 'parent', 'name', and 'value'
    treemap_data = []
    
    # Add secondary attribute nodes with primary attribute as their parent
    for _, row in df_treemap.iterrows():
        primary = row[primary_attr]
        secondary = row[secondary_attr]
        value = row[comparing_attr]
        treemap_data.append({
            'name': secondary,  # Subcategory (secondary attribute)
            'parent': primary,   # Parent category (primary attribute)
            'value': value       # Value to size the box (comparing attribute)
        })
    
    # Add top-level primary attribute nodes to act as parents
    for primary in top_primary:
        treemap_data.append({
            'name': primary,
            'parent': '',
            'value': 0  # Top-level primary attributes act as parent nodes with no value
        })
    
    # Create the Treemap figure using Plotly Express
    fig_treemap = px.treemap(
        pd.DataFrame(treemap_data),
        path=['parent', 'name'],
        values='value',
        title=f"Average {comparing_attr} Distribution by {primary_attr} and {secondary_attr}"
    )
    
    fig_treemap.update_layout(template='plotly_dark')
    
    return fig_treemap


# Generate the treemap figure
fig_treemap = gen_treemap(df2, 'Genre', 'Director', 'Revenue')

# set the stylesheet
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'
]

fig_choropleth = px.choropleth(df3, locations='ISO_Code', color='Number of Production', hover_name='Country', color_continuous_scale='Viridis')
fig_choropleth.update_layout(template='plotly_dark')


#initialize the app
app = Dash(__name__, external_stylesheets=external_stylesheets)

#app layout
app.layout = dmc.Container([
    dmc.Title('Movie and TV Shows Data Visualization', color="blue", size="h3"),
    dmc.Title("COMP4462 Group 8 (Daisy Har, Aatrox Deng, Lyam Tang)", size="h6"),
    dbc.Row(
        [
            html.Button(id='filter-button', children=[
                html.I(className='fas fa-filter', style={'margin-right': '5px', 'color': 'white'}),
            ], style={'background-color': 'transparent', 'border': 'none'}),
            dbc.Col(
                html.Div([
                    dcc.Dropdown(
                        id='genre-dropdown',
                        options=[{'label': genre, 'value': genre} for genre in df2['Genre'].unique()],
                        multi=True,
                        searchable=True,
                        placeholder='Select Genre',
                        className='custom-dropdown',
                        style={
                            'color': 'black',  # Text color for selected options
                            'background-color': 'white'  # Dropdown background color
                        },
                    ),
                    dcc.Dropdown(
                        id='country-dropdown',
                        options=[{'label': country, 'value': country} for country in df2['Country'].unique()],
                        multi=True,
                        searchable=True,
                        placeholder='Select Country',
                        className='custom-dropdown',
                        style={
                            'color': 'black',  # Text color for selected options
                            'background-color': 'white'  # Dropdown background color
                        },
                    ),
                    html.Link(rel='stylesheet', href='./styles.css'),
                    html.Div(id='filtered-data', style={'display': 'none'})
                ])
            ),
            dbc.Col(
                html.Div([
                    dcc.Dropdown(
                        id='director-dropdown',
                        options=[{'label': director, 'value': director} for director in df2['Director'].unique()],
                        multi=True,
                        searchable=True,
                        placeholder='Select Director',
                        className='custom-dropdown',
                        style={
                            'color': 'black',  # Text color for selected options
                            'background-color': 'white'  # Dropdown background color
                        },
                    ),
                    dcc.Dropdown(
                        id='cast-dropdown',
                        options=[{'label': cast, 'value': cast} for cast in df2['Cast'].unique()],
                        multi=True,
                        searchable=True,
                        placeholder='Select Cast',
                        className='custom-dropdown',
                        style={
                            'color': 'black',  # Text color for selected options
                            'background-color': 'white'  # Dropdown background color
                        },
                    )
                ])
            ),
            dbc.Col(
                html.Div([
                    dcc.RangeSlider(
                        id='year-slider',
                        min=2020,
                        max=2025,
                        step=1,
                        marks={i: str(i) for i in range(2020, 2026)},
                        value=[2020, 2025],
                    ),
                    dcc.RangeSlider(
                        id='reveue-slider',
                        min=df2['Revenue'].min(),
                        max=df2['Revenue'].max(),
                        step=100000000,
                        marks={i: f'{i:,}' for i in range(0, int(df2['Revenue'].max()), 1000000000)},
                        value=[0, df2['Revenue'].max()]
                    ),
                    dcc.RangeSlider(
                        id='duration-slider',
                        min=df2['Duration (min)'].min(),
                        max=df2['Duration (min)'].max(),
                        step=10,
                        marks={i: f'{i:,}' for i in range(df2['Duration (min)'].min().astype(int), df2['Duration (min)'].max().astype(int), 10)},
                        value=[df2['Duration (min)'].min(), df2['Duration (min)'].max()],
                    ),
                ])
            )
        ],
        style={
            'border': '1px solid #343a40',
            'padding': '10px',
            'background-color': '#212329',
            'border-radius': '5px'
        }
    ),
    html.Div([
        html.I(className='fas fa-search search-icon', style={'margin-right': '5px'}),
        dbc.Input(id='search-input', type='text', placeholder='Search...', style={'flex': '1', 'background-color': 'transparent', 'color': 'white'}),
    ], style={'display': 'flex', 'align-items': 'center'}),
    dmc.Grid([
        dmc.Col([
            dcc.Graph(id='fig-parallel', figure=fig_parallel)
        ], span=8),
        dmc.Col([
            dcc.Graph(id='fig-bar', figure=fig_bar),
            dcc.Graph(id='fig-star', figure=fig_star)
        ], span=4)
    ]),
    dmc.Grid(justify='center',
             align='center',
             children=[
                 dmc.Col([
                     dmc.Title("Map Selector", size="h4", align='center', color='blue'),
                     dcc.RadioItems(
                         id='map-radio',
                         options=['Number of Production', 'Revenue', 'Average Metascore', 'Average Votes']
                     ),

                 ], span=2),
                 dmc.Col([
                     dcc.Graph(id="graph-choropleth", figure=fig_choropleth)
                 ], span=10)
             ]),
    dmc.Grid([
        dmc.Col([
            html.Div([
                dcc.Dropdown(
                    id='primary-attr-dropdown',
                    options=[
                        {'label': 'Genre', 'value': 'Genre'},
                        {'label': 'Country', 'value': 'Country'},
                        {'label': 'Director', 'value': 'Director'},
                        {'label': 'Cast', 'value': 'Cast'}
                    ],
                    placeholder='Select Primary Attribute',
                    className='custom-dropdown',
                    style={
                        'color': 'black',  # Text color for selected options
                        'background-color': 'white'  # Dropdown background color
                    },
                ),
                dcc.Dropdown(
                    id='secondary-attr-dropdown',
                    options=[
                        {'label': 'Genre', 'value': 'Genre'},
                        {'label': 'Country', 'value': 'Country'},
                        {'label': 'Director', 'value': 'Director'},
                        {'label': 'Cast', 'value': 'Cast'}
                    ],
                    placeholder='Select Secondary Attribute',
                    className='custom-dropdown',
                    style={
                        'color': 'black',  # Text color for selected options
                        'background-color': 'white'  # Dropdown background color
                    },
                ),
                dcc.Dropdown(
                    id='comparing-attr-dropdown',
                    options=[
                        {'label': 'Revenue', 'value': 'Revenue'},
                        {'label': 'Rating', 'value': 'Rating'},
                        {'label': 'Votes', 'value': 'Votes'}
                    ],
                    placeholder='Select Comparing Attribute',
                    className='custom-dropdown',
                    style={
                        'color': 'black',  # Text color for selected options
                        'background-color': 'white'  # Dropdown background color
                    },
                ),
            ], style={'margin-bottom': '10px'}),
            dcc.Graph(id='graph-one', figure=fig_treemap)
        ], span=6),
        dmc.Col([
            dcc.Graph(id="graph-two", figure=fig_choropleth)
        ], span=6)
    ])
], fluid=True)


@app.callback(
    Output('filtered-data', 'children'),
    Input('filter-button', 'n_clicks'),
    Input('genre-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('director-dropdown', 'value'),
    Input('cast-dropdown', 'value'),
    Input('year-slider', 'value'),
    Input('reveue-slider', 'value'),
    Input('duration-slider', 'value')
)
def filter_data(n_clicks, genre_values, country_values, director_values, cast_values, year_range, revenue_range, duration_range):
    df_filtered = df2[
        (df2['Genre'].isin(genre_values if genre_values else df2['Genre'])) &
        (df2['Country'].isin(country_values if country_values else df2['Country'])) &
        (df2['Director'].isin(director_values if director_values else df2['Director'])) &
        (df2['Cast'].isin(cast_values if cast_values else df2['Cast'])) &
        (df2['Year'].between(year_range[0], year_range[1])) &
        (df2['Revenue'].between(revenue_range[0], revenue_range[1])) &
        (df2['Duration (min)'].between(duration_range[0], duration_range[1]))
    ]

    return df_filtered.to_json(orient='split')

@app.callback(
    Output('fig-parallel', 'figure'),
    Output('fig-bar', 'figure'),
    Output('fig-star', 'figure'),
    Output('graph-one', 'figure'),
    Input('filtered-data', 'children'),
    Input('primary-attr-dropdown', 'value'),
    Input('secondary-attr-dropdown', 'value'),
    Input('comparing-attr-dropdown', 'value')
)
def update_graph(filtered_data, primary_attr, secondary_attr, comparing_attr):
    df_filtered = pd.read_json(filtered_data, orient='split')

    fig_parallel = gen_parallel(df_filtered)
    fig_bar = gen_bar(df_filtered)
    fig_star = gen_star(df_filtered)
    if primary_attr and secondary_attr and comparing_attr:
        fig_treemap = gen_treemap(df_filtered, primary_attr, secondary_attr, comparing_attr)
    else:
        fig_treemap = fig_treemap
        
    return fig_parallel, fig_bar, fig_star, fig_treemap

#run app
if __name__ == '__main__':
    app.run(debug=True)
