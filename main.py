import numpy as np
import pandas as pd
from dash import Dash, _dash_renderer, Input, Output, html, dcc, callback, dash_table
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import plotly.express as px

#read data from csv using pd
df2 = pd.read_csv('./datasets/dataset.csv')

#data summary
attributes = df2.columns.tolist()
attributes = [dmc.Text(size="sm", weight=500, children=item) for item in attributes]

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
        dmc.Title("Attributes", size="h10"),
        dmc.Container(fluid=True, children=attributes)
    ]),
    html.Div(className='row', children=[
        dmc.Title("Dataset", size="h10"),
        dash_table.DataTable(data=df2.to_dict('records'), page_size=10, style_table={'overflowX': 'auto'})
    ]),
    dcc.Graph(id="graph-three", figure= fig_three),
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

