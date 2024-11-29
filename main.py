import numpy as np
import pandas as pd
from dash.exceptions import PreventUpdate
from scipy.stats import norm
from dash import Dash, _dash_renderer, Input, Output, html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px
import plotly.graph_objects as go

## preprocess the dataset
df2 = pd.read_csv('./datasets/dataset.csv')
df2 = df2.assign(Genre=df2['Genre'].str.split(', ')).explode('Genre') # tokenise genre
df2 = df2.assign(Cast=df2['Cast'].str.split(', ')).explode('Cast') # tokenise cast
df2 = df2.assign(Director=df2['Director'].str.split(', ')).explode('Director') # tokenise director
df2 = df2.assign(Country=df2['Country'].str.split(', ')).explode('Country') # tokenise country
df2 = df2[df2['Revenue'] >= 0] # filter out movies with negative revenue
df2 = df2[df2['Year'] >= 2020] # filter out movies before 2020

# add country iso code
country_iso_df = pd.read_csv('./datasets/countries_with_iso_codes.csv')
df3 = (df2['Country'].value_counts()).reset_index()
df3.columns = ['Country','Number of Production']
df3 = df3.merge(country_iso_df, on='Country', how='left')
df3 = df3[['Country','ISO_Code','Number of Production']]

## data summary
# Attributes: Poster, Title, Year, Certificate, Duration (min), Genre, Rating, Metascore, Director, Cast, Votes, Description, Review Count, Review Title, Review, Revenue, Country

## funcs to gen graphs
# func to gen heatmap fig
def gen_heatmap(df_filtered):
    # Group the data by year and genre, and calculate the sum of revenue
    df_heatmap = df_filtered.groupby(['Year', 'Genre'])['Revenue'].sum().reset_index()

    # Create the heatmap
    fig_heatmap = px.density_heatmap(df_heatmap,
                                     x='Year',
                                     y='Genre',
                                     z='Revenue',
                                     color_continuous_scale='YlOrBr',
                                     title='Revenue Heatmap by Year and Genre')

    fig_heatmap.update_layout(template='plotly_dark', height=600)

    return fig_heatmap

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
    
def gen_choropleth(filtered, map_radio):
    def create_choropleth(data, color_column, title):
        fig = px.choropleth(data, locations='ISO_Code', color=color_column, hover_name='Country', color_continuous_scale='Viridis', title=title)
        fig.update_layout(template='plotly_dark')
        fig.update_coloraxes(colorbar=dict(title=color_column), colorscale='Viridis', cmin=1, cmax=data[color_column].max(), colorbar_tickformat='.0e')
        return fig

    if map_radio == "Revenue":
        data = filtered.groupby('Country')['Revenue'].sum().reset_index()
        data.columns = ['Country', 'Revenue']
        data = data.merge(country_iso_df, on='Country', how='left')
        data['Revenue'] = np.log1p(data['Revenue'])  # Apply log scale
        return create_choropleth(data, 'Revenue', 'Total Revenue by Country (Log Scale)')

    elif map_radio == 'Average Metascore':
        data = filtered.groupby('Country')['Metascore'].mean().reset_index()
        data.columns = ['Country', 'Average Metascore']
        data = data.merge(country_iso_df, on='Country', how='left')
        return create_choropleth(data, 'Average Metascore', 'Average Metascore by Country')

    elif map_radio == 'Average Votes':
        data = filtered.groupby('Country')['Votes'].mean().reset_index()
        data.columns = ['Country', 'Average Votes']
        data = data.merge(country_iso_df, on='Country', how='left')
        data['Average Votes'] = np.log1p(data['Average Votes'])  # Apply log scale
        return create_choropleth(data, 'Average Votes', 'Average Votes by Country (Log Scale)')

    elif map_radio == "Number of Production":
        data = filtered['Country'].value_counts().reset_index()
        data.columns = ['Country', 'Number of Production']
        data = data.merge(country_iso_df, on='Country', how='left')
        data['Number of Production'] = np.log1p(data['Number of Production'])  # Apply log scale
        return create_choropleth(data, 'Number of Production', 'Number of Productions by Country (Log Scale)')

def gen_treemap(df_filtered, primary_attr, secondary_attr, comparing_attr):
    # Group by primary and secondary attributes, averaging the comparing attribute
    df_treemap = df_filtered.groupby([primary_attr, secondary_attr])[comparing_attr].mean().reset_index()
    
    # Filter top 10 primary attribute groups based on the average comparing attribute
    top_primary = df_treemap.groupby(primary_attr)[comparing_attr].mean().nlargest(10).index
    df_treemap = df_treemap[df_treemap[primary_attr].isin(top_primary)]
    
    # Filter top 10 secondary attribute groups within each primary attribute group
    df_treemap = df_treemap.groupby(primary_attr).apply(lambda x: x.nlargest(10, comparing_attr)).reset_index(drop=True)
    
    # Prepare the data for the treemap
    treemap_data = []
    text_data = []  # To store text for each block
    
    for _, row in df_treemap.iterrows():
        primary = row[primary_attr]
        secondary = row[secondary_attr]
        value = row[comparing_attr]
        treemap_data.append({
            'name': secondary,  # Subcategory (secondary attribute)
            'parent': primary,   # Parent category (primary attribute)
            'value': value       # Value to size the box (comparing attribute)
        })
        text_data.append(f"{secondary}: {value:,.0f}")  # Add formatted value as text

    for primary in top_primary:
        treemap_data.append({
            'name': primary,
            'parent': '',
            'value': 0  # Top-level primary attributes act as parent nodes with no value
        })
        text_data.append(f"{primary}: {df_treemap[df_treemap[primary_attr] == primary][comparing_attr].sum():,.0f}")

    treemap_df = pd.DataFrame(treemap_data)
    treemap_df['text'] = text_data  # Add text data to the DataFrame

    # Create the Treemap figure using Plotly
    fig_treemap = px.treemap(
        treemap_df,
        path=['parent', 'name'],
        values='value',
        title=f"Average {comparing_attr} Distribution by {primary_attr} and {secondary_attr}",
        custom_data=['text'],  # Pass the custom text
    )

    # Use textinfo to display the custom text on the treemap
    fig_treemap.update_traces(
        texttemplate="%{customdata[0]}",  # Use the custom data as the text
        textinfo="label+text+value",     # Show label, custom text, and value
    )

    fig_treemap.update_layout(template='plotly_dark')

    return fig_treemap

# generate all the default graph
def gen_graph(filtered, map_radio, treemap_attrs):
    return gen_heatmap(filtered), gen_star(filtered), gen_choropleth(filtered, map_radio), gen_treemap(filtered, treemap_attrs[0], treemap_attrs[1], treemap_attrs[2])

fig_heatmap, fig_star, fig_choropleth, fig_treemap = gen_graph(df2, 'Revenue',['Genre', 'Director', 'Revenue'])

## app layout
# set the stylesheet
external_stylesheets = [
    'https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/darkly/bootstrap.min.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.1/css/all.min.css'
]

# initialize the app
app = Dash(__name__, external_stylesheets=external_stylesheets)

# create a toggle button
toggle_button = html.Div(
    children=[
        html.Button(
            children=[
                html.I(id='toggle-icon', className='fas fa-toggle-on', style={'color': 'white'})  # Use Font Awesome toggle-on icon
            ],
            id='toggle-button',
            n_clicks=0,
            style={
                'margin-bottom': '10px',
                'background-color': 'transparent',  # Transparent background
                'border': 'none'  # No border
            }
        )
    ],
    style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'}
)

# create a general filer
general_filter = html.Div(
    id='general-filter', children=[
        dbc.Col([
            dmc.Title("General Filter", size="h4", align='center', color='blue'),
            dcc.Checklist(
                id='diagram-checklist',
                options=[
                    {'label': 'Heatmap', 'value': 'heatmap'},
                    {'label': 'Star Coordinates', 'value': 'star'},
                    {'label': 'Choropleth', 'value': 'choropleth'},
                    {'label': 'Treemap', 'value': 'treemap'}
                ],
                value=['heatmap'],  # Default selected diagrams
                labelStyle={'display': 'inline-block', 'color': 'white'}
            ),
            dbc.Row([
                dbc.Col([
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
                    html.Div(id='filtered-data', style={'display': 'none'})
                ]),
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
                        html.Label('Year Range', style={'color': 'white'}),
                        dcc.RangeSlider(
                            id='year-slider',
                            min=2020,
                            max=2025,
                            step=1,
                            marks={i: str(i) for i in range(2020, 2026)},
                            value=[2020, 2025],
                        ),
                        html.Label('Revenue Range', style={'color': 'white'}),
                        dcc.RangeSlider(
                            id='revenue-slider',
                            min=df2['Revenue'].min(),
                            max=df2['Revenue'].max(),
                            step=50000000,
                            marks={i: f'{i//1000}' for i in range(0, int(df2['Revenue'].max()), 500000000)},
                            value=[0, df2['Revenue'].max()]
                        ),
                        html.Label('Duration Range', style={'color': 'white'}),
                        dcc.RangeSlider(
                            id='duration-slider',
                            min=df2['Duration (min)'].min(),
                            max=df2['Duration (min)'].max(),
                            step=50,
                            marks={i: f'{i:,}' for i in range(df2['Duration (min)'].min().astype(int), df2['Duration (min)'].max().astype(int), 50)},
                            value=[df2['Duration (min)'].min(), df2['Duration (min)'].max()],
                        )
                    ])
                )
            ])
        ])
    ]
)

# create a
# create a map fiter
map_filter = html.Div(
    id='map-filter', children=[
        dmc.Title("Map Selector", size="h4", align='center', color='blue'),
        dcc.RadioItems(
            id='map-radio',
            options=['Number of Production', 'Revenue', 'Average Metascore', 'Average Votes']
        )
    ]
)

# create a tree filter
tree_filter = html.Div(
    id='tree-filter', children=[
        dmc.Title("Tree Selector", size="h4", align='center', color='blue'),
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
            }
        )
    ]
)

# create a filter panel
filter_panel = html.Div(
    id='filter-panel', children=[
        dbc.Col(
        [
            dbc.Col([
                html.Button(id='filter-button', children=[
                    html.I(className='fas fa-filter', style={'margin-right': '5px', 'color': 'white'}),
                ], style={'background-color': 'transparent', 'border': 'none', 'justify-content': 'center'})
            ]),
            dbc.Col([
                general_filter,
                html.Hr(style={'border': '1px solid #343a40'}),
                map_filter,
                html.Hr(style={'border': '1px solid #343a40'}),
                tree_filter
            ])
        ],
        style={
            'border': '1px solid #343a40',
            'padding': '10px',
            'background-color': '#212329',
            'border-radius': '5px'
            })
    ],
    style={'display': 'none'})

# overall layout
app.layout = dmc.Container([
    dmc.Title('Movie and TV Shows Data Visualization', color="blue", size="h3"),
    dmc.Title("COMP4462 Group 8 (Daisy Har, Aatrox Deng, Lyam Tang)", size="h6"),
    toggle_button,
    filter_panel,
    dcc.Graph(id='fig-heatmap', figure=fig_heatmap, hoverData={'points': [{'customdata': ['Action']}]}, style={'display': 'block'}),
    dcc.Graph(id='fig-star', figure=fig_star, style={'display': 'block'}),
    dcc.Graph(id="fig-choropleth", figure=fig_choropleth, style={'display': 'block'}),
    dcc.Graph(id='fig-tree', figure=fig_treemap, style={'display': 'block'})
], fluid=True)

## callback funcs
# update graph
@app.callback(
    Output('fig-heatmap', 'figure'),
    Output('fig-star', 'figure'),
    Output('fig-choropleth', 'figure'),
    Output('fig-tree', 'figure'),
    Output('filter-button', 'n_clicks'),
    [
        Output('fig-heatmap', 'style'),
        Output('fig-star', 'style'),
        Output('fig-choropleth', 'style'),
        Output('fig-tree', 'style')
    ],
    Input('filter-button', 'n_clicks'),
    Input('genre-dropdown', 'value'),
    Input('country-dropdown', 'value'),
    Input('director-dropdown', 'value'),
    Input('cast-dropdown', 'value'),
    Input('year-slider', 'value'),
    Input('revenue-slider', 'value'),
    Input('duration-slider', 'value'),
    Input('map-radio', 'value'),
    Input('primary-attr-dropdown', 'value'),
    Input('secondary-attr-dropdown', 'value'),
    Input('comparing-attr-dropdown', 'value'),
    Input('diagram-checklist', 'value')
)
def update(n_clicks, genre_values, country_values, director_values, cast_values, year_range, revenue_range, duration_range, map_radio, primary_attr, secondary_attr, comparing_attr , selected_diagrams):
    # check for filter button click
    if n_clicks == 0:
        raise PreventUpdate
    else:
        # filter df2 based on the selected values
        df_filtered = df2[
            (df2['Genre'].isin(genre_values if genre_values else df2['Genre'])) &
            (df2['Country'].isin(country_values if country_values else df2['Country'])) &
            (df2['Director'].isin(director_values if director_values else df2['Director'])) &
            (df2['Cast'].isin(cast_values if cast_values else df2['Cast'])) &
            (df2['Year'].between(year_range[0], year_range[1])) &
            (df2['Revenue'].between(revenue_range[0], revenue_range[1])) &
            (df2['Duration (min)'].between(duration_range[0], duration_range[1]))
        ]

        # update the graphs
        treemap_attrs = ['Genre', 'Director', 'Revenue']
        if primary_attr and secondary_attr and comparing_attr:
            treemap_attrs = [primary_attr, secondary_attr, comparing_attr]
        updated_fig_heatmap, updated_fig_star, updated_fig_choropleth, updated_fig_treemap = gen_graph(df_filtered, map_radio, treemap_attrs)

        # update the visibilities of the graphs
        styles = {
            'fig-heatmap': {'display': 'none'},
            'fig-star': {'display': 'none'},
            'fig-choropleth': {'display': 'none'},
            'fig-tree': {'display': 'none'}
        }
        if 'heatmap' in selected_diagrams:
            styles['fig-heatmap'] = {'display': 'block'}
        if 'star' in selected_diagrams:
            styles['fig-star'] = {'display': 'block'}
        if 'choropleth' in selected_diagrams:
            styles['fig-choropleth'] = {'display': 'block'}
        if 'treemap' in selected_diagrams:
            styles['fig-tree'] = {'display': 'block'}

        return updated_fig_heatmap, updated_fig_star, updated_fig_choropleth, updated_fig_treemap, 0, styles['fig-heatmap'], styles['fig-star'], styles['fig-choropleth'], styles['fig-tree']

# update filter panel
@app.callback(
    [
        Output('filter-panel', 'style'),
        Output('toggle-icon', 'className')
    ],
        Input('toggle-button', 'n_clicks')
)
def toggle_filter_panel(n_clicks):
    if n_clicks % 2 == 1:
        return {'display': 'block'}, 'fas fa-toggle-on'
    else:
        return {'display': 'none'}, 'fas fa-toggle-off'

#run app
if __name__ == '__main__':
    app.run(debug=True)
