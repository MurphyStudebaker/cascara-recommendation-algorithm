"""
A Content Based Recommendation System 
For Local Coffeehouse in Los Angeles

Visit https:/www.getcascara.com for more information

@Author: Murphy Studebaker
@Tutorial: https://www.datacamp.com/community/tutorials/recommender-systems-python
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

raw_data = pd.read_csv("./coffee-data-la.csv")

# Turn Airtable string data into suitable format for vector analysis
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]

def get_list(string):
    return list(string.split(","))

# Applying the cleaning functions in place to our data    
features = ['atmosphere','amenities','goodFor']
for feature in features:
    raw_data[feature] = raw_data[feature].apply(get_list)

for feature in features:
    raw_data[feature] = raw_data[feature].apply(clean_data)

# A method to create a "soup" of relevant words to feed into our vectorizer
def create_soup(x):
    soup = ""
    for atmosphere in x['atmosphere']:
        soup+= (atmosphere + " ")
    for amenity in x['amenities']:
        soup+= (amenity + " ")
    for item in x['goodFor']:
        soup+= (item + " ")
    return soup

raw_data['soup'] = raw_data.apply(create_soup, axis=1)

# Calculating a matrix of occurences of words throughout the soups
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(raw_data['soup'])

# Calculating a matrix of similarity using the soup between all coffeehouses
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# indexing the dataset so we can link it back up at the end
raw_data = raw_data.reset_index()
indices = pd.Series(raw_data.index, index=raw_data['name'])

# The magical method that sorts by similarity for each coffeehouse 
# and returns the top three matches (excluding index 0, which is the same)
def get_recommendations(name, cosine_sim=cosine_sim):
    # Get the index of the coffeehouse that matches the input name
    idx = indices[name]

    # Get the scores of similarity for all other cafes to this cafe
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the scores in descending order
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get only the top 3 most similar scores
    sim_scores = sim_scores[1:4]

    # Get the indices to link it all back up
    coffee_indices = [i[0] for i in sim_scores]
    
    # Return the top 3 most similar cafes
    return (raw_data['name'].iloc[coffee_indices])

print(get_recommendations("Bru Coffee Bar"))
