import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

recipes = pd.read_csv('FinalProjectAI/RAW_recipes.csv')
interactions = pd.read_csv('FinalProjectAI/interactions_train.csv')

recipes['calories'] = recipes['nutrition'].apply(lambda x: ast.literal_eval(x)[0])
recipes['ingredients_str'] = recipes['ingredients'].apply(lambda x: ' '.join(ast.literal_eval(x)))
recipes['tags'] = recipes['tags'].apply(lambda x: ast.literal_eval(x))

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
count_matrix = vectorizer.fit_transform(recipes['ingredients_str'])


def get_user_input_vector(ingredients, vectorizer):
    input_str = ' '.join(ingredients)
    user_input_vector = vectorizer.transform([input_str])
    return user_input_vector


def recommend_recipes(df, user_input_vector, min_calories, max_calories, favorite_tags, top_n=10):
    cosine_sim = cosine_similarity(user_input_vector, count_matrix)
    sim_scores = list(enumerate(cosine_sim[0]))

    for i, score in sim_scores:
        if any(tag in df.iloc[i]['tags'] for tag in favorite_tags):
            sim_scores[i] = (i, score * 2)

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended = []
    count = 0
    for item in sim_scores:
        idx = item[0]
        if df.iloc[idx]['calories'] >= min_calories and df.iloc[idx]['calories'] <= max_calories:
            recommended.append(df.iloc[idx])
            count += 1
            if count >= top_n:
                break

    return pd.DataFrame(recommended)


def display_recommendations(recommended_recipes):
    result = []

    # Iterate through each recommended recipe
    for custom_idx, recipe in recommended_recipes.iterrows():
        # Process ingredients and tags for display
        ingredients = ', '.join(ast.literal_eval(recipe['ingredients']))
        tags = ', '.join(recipe['tags'])

        # Construct a dictionary for each recipe
        recipe_dict = {
            'rank': custom_idx + 1,
            'name': recipe['name'],
            'calories': recipe['calories'],
            'ingredients': ingredients,
            'tags': tags
        }

        result.append(recipe_dict)
    return result




def inputs(user_ingredients, min_calories, max_calories, favorite_tags):
    user_input_vector = get_user_input_vector(user_ingredients, vectorizer)
    recommended_recipes = recommend_recipes(recipes, user_input_vector, min_calories, max_calories, favorite_tags,
                                            top_n=5)

    return display_recommendations(recommended_recipes)
