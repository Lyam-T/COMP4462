import pandas as pd
import numpy as np

imdb_df = pd.read_csv('imdb-movies-dataset.csv')
tmdb_df = pd.read_csv('TMDB_movie_dataset_v11.csv')

# output all the attributes for each df
print("\nAttributes for imdb_df:")
print(imdb_df.columns)

print("\nAttributes for tmdb_df:")
print(tmdb_df.columns)

# merge the data
tmdf_revenue_country = tmdb_df[['title', 'revenue', 'production_countries']]
tmdf_revenue_country.rename(columns={'title': 'Title', 'revenue': 'Revenue', 'production_countries': 'Country'}, inplace=True)
df = pd.merge(imdb_df, tmdf_revenue_country, on='Title', how='left')

# output all the attributes of merged data
print("\nAttributes for merged data:")
print(df.columns)

# clean the data
df = df.dropna(subset=['Year', 'Genre', 'Rating', 'Revenue', 'Country'])
print(df.shape)
df.to_csv('dataset.csv', index=False)