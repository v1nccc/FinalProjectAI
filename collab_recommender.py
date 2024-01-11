import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import ast

interactions = pd.read_csv('FinalProjectAI/interactions_train.csv')
recipes = pd.read_csv('FinalProjectAI/RAW_recipes.csv')

_all = interactions.drop(['date', 'u', 'i'], axis=1)

grouped_1 = _all.groupby(['user_id'], as_index=False, sort=False).agg({'recipe_id': 'count'}).reset_index(drop=True)
grouped_1 = grouped_1.rename(columns={'recipe_id': 'reviews_count'})
grouped_1 = grouped_1.sort_values('reviews_count', ascending=False).iloc[:7500, :]

grouped_2 = _all.groupby(['recipe_id'], as_index=False, sort=False).agg({'user_id': 'count'}).reset_index(drop=True)
grouped_2 = grouped_2.rename(columns={'user_id': 'reviews_count'})
grouped_2 = grouped_2.sort_values('reviews_count', ascending=False).iloc[:7500, :]

_part = pd.merge(_all.merge(grouped_1).drop(['reviews_count'], axis=1), grouped_2).drop(['reviews_count'], axis=1)

grouped_user = _part.groupby(['user_id'], as_index=False, sort=False).agg({'recipe_id': 'count'}).reset_index(drop=True)
grouped_user = grouped_user.rename(columns={'recipe_id': 'reviews_count'})

grouped_recipe = _part.groupby(['recipe_id'], as_index=False, sort=False).agg({'user_id': 'count'}).reset_index(
    drop=True)
grouped_recipe = grouped_recipe.rename(columns={'user_id': 'reviews_count'})

new_userID = dict(zip(list(_part['user_id'].unique()),
                      list(range(len(_part['user_id'].unique())))))
new_recipeID = dict(zip(list(_part['recipe_id'].unique()),
                        list(range(len(_part['recipe_id'].unique())))))
df = _part.replace({'user_id': new_userID, 'recipe_id': new_recipeID})

recipe = recipes[['name', 'id', 'ingredients']].merge(_part[['recipe_id']], left_on='id', right_on='recipe_id', how='right').drop(['id'], axis=1).drop_duplicates().reset_index(drop=True)

mean = df.groupby(['user_id'], as_index=False, sort=False).mean().rename(columns={'rating': 'rating_mean'})
df = df.merge(mean[['user_id', 'rating_mean']], how='left')
df.insert(2, 'rating_adjusted', df['rating'] - df['rating_mean'])

train_data, test_data = train_test_split(df, test_size=0.25)

n_users = df.user_id.unique()
n_items = df.recipe_id.unique()

train_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in train_data.itertuples():
    train_data_matrix[row[1] - 1, row[2] - 1] = row[3]

test_data_matrix = np.zeros((n_users.shape[0], n_items.shape[0]))
for row in test_data.itertuples():
    test_data_matrix[row[1] - 1, row[2] - 1] = row[3]

user_similarity = 1 - pairwise_distances(train_data_matrix, metric='cosine')

item_similarity = 1 - pairwise_distances(train_data_matrix.T, metric='cosine')


def predict(ratings, similarity, _type='user'):
    if _type == 'user':
        pred = similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=np.newaxis)])

    elif _type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    return pred


user_pred = predict(train_data_matrix, user_similarity, _type='user')

user_pred_df = pd.DataFrame(user_pred, columns=list(n_items))
user_pred_df.insert(0, 'user_id', list(n_users))

item_pred = predict(train_data_matrix, item_similarity, _type='item')

item_pred_df = pd.DataFrame(item_pred, columns=list(n_items))
item_pred_df.insert(0, 'user_id', list(n_users))


# The recommendation engine

def getRecommendations_UserBased(user_id, top_n=10):
    rated = set(df['recipe_id'].loc[df['user_id'] == user_id])

    _all = user_pred_df.loc[user_pred_df['user_id'] == user_id].copy()
    _all.drop(columns=rated, inplace=True, errors='ignore')

    _sorted = _all.iloc[:, 1:].T.nlargest(top_n, _all.index[0])

    result = []
    for rank, (predicted_recipe_id, _) in enumerate(_sorted.iterrows(), start=1):
        original_recipe_id = [orig_id for orig_id, new_id in new_recipeID.items() if new_id == predicted_recipe_id][0]

        recipe_details = recipe.loc[recipe['recipe_id'] == original_recipe_id].iloc[0]
        name = recipe_details['name']
        ingredients = ', '.join(ast.literal_eval(recipe_details['ingredients']))

        recommendation = {
            'rank': rank,
            'recipe_id': original_recipe_id,
            'name': name,
            'ingredients': ingredients
        }
        result.append(recommendation)

    return result



def getRecommendations_ItemBased(user_id, top_n=10):
    rated = set(df.loc[df['user_id'] == user_id, 'recipe_id'])

    user_predictions = item_pred_df.loc[item_pred_df['user_id'] == user_id].copy()
    user_predictions.drop(columns=rated, inplace=True, errors='ignore')

    top_predictions = user_predictions.iloc[:, 1:].T.nlargest(top_n, user_predictions.index[0])

    result = []
    for rank, (predicted_recipe_id, _) in enumerate(top_predictions.iterrows(), start=1):
        original_recipe_id = [orig_id for orig_id, new_id in new_recipeID.items() if new_id == predicted_recipe_id][0]

        recipe_details = recipe.loc[recipe['recipe_id'] == original_recipe_id].iloc[0]
        name = recipe_details['name']
        ingredients = ', '.join(ast.literal_eval(recipe_details['ingredients']))

        recommendation = {
            'rank': rank,
            'recipe_id': original_recipe_id,
            'name': name,
            'ingredients': ingredients
        }
        result.append(recommendation)

    return result
