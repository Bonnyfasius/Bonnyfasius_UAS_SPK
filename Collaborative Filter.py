import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

reatings = pd.read_csv("Data_movie.csv", sep= ";", index_col= 0)
reatings = reatings.fillna(0)
print(reatings)

def standardize(row): #menghitung standar devaisi dari masing" data
    new_row = (row - row.mean()) / (row.max() - row.min())
    return new_row

reatings_std = reatings.apply(standardize)
# print(reatings_std)

item_similarity = cosine_similarity(reatings_std.T)
# print(item_similarity)

item_similarity_df = pd.DataFrame(item_similarity, index= reatings.columns, columns= reatings.columns)
print(item_similarity_df)

#Rekomendasi movie
def get_similar_movies(moview_name, user_rating):
    similar_score = item_similarity_df[moview_name] * (user_rating-2.5)
    similar_score = similar_score.sort_values(ascending = False)
    return similar_score

action_lover = [("action1",5), ("action1", 1), ("action1",1)]

similar_movies = pd.DataFrame()

for movie,rating in action_lover:
    similar_movies = similar_movies.append(get_similar_movies(movie,rating), ignore_index=True)

similar_movies.head()
similar_movies.sum().sort_values(ascending=False)

print(similar_movies)
