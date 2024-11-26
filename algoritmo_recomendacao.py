import numpy as np
import pandas as pd
from surprise import Dataset, SVDpp, Reader
from surprise.model_selection import train_test_split, cross_validate, GridSearchCV
from surprise import accuracy
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json
import logging
import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Coletar dados dos filmes e classificações usando a API do The Movie Database (TMDb)
API_KEY = 'd64e12a20a79b368c0a6af27f9a7e1ad'

# Função para coletar dados dos filmes
def get_movies_from_tmdb():
    movies = []
    for page in range(1, 28):  # Coletar filmes das primeiras 7 páginas
        response = requests.get(f'https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=pt-BR&page={page}', verify=False)
        if response.status_code == 200:
            data = response.json()
            for movie in data['results']:
                # Coletar também diretores e atores principais
                movie_details = requests.get(f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={API_KEY}&language=pt-BR&append_to_response=credits", verify=False)
                if movie_details.status_code == 200:
                    details_data = movie_details.json()
                    directors = [crew_member['id'] for crew_member in details_data['credits']['crew'] if crew_member['job'] == 'Director']
                    actors = [cast_member['id'] for cast_member in details_data['credits']['cast'][:5]]  # Pegar os 5 principais atores
                    genres = [genre['id'] for genre in details_data.get('genres', [])]
                    movies.append({
                        "movieId": movie['id'],
                        "title": movie['title'],
                        "genres": genres,
                        "directors": directors,
                        "actors": actors
                    })
    return pd.DataFrame(movies)

# Função para coletar classificações dos usuários
def get_ratings_from_tmdb():
    global user_watch_history, user_preferences
    user_watch_history = {}  # Dicionário para armazenar os filmes assistidos por cada usuário
    user_preferences = {}  # Dicionário para armazenar as preferências dos usuários
    user_ids = range(1, 80)  # Usuários de 1 a 79
    movie_ids = movies_df['movieId'].tolist()  # Lista de todos os filmes coletados
    ratings = []

    for user_id in user_ids:
        rated_movies = np.random.choice(movie_ids, size=np.random.choice([40, 60]), replace=False)
        user_watch_history[user_id] = list(rated_movies)  # Armazenar os filmes assistidos por cada usuário
        user_preferences[user_id] = {
            "genres": [],
            "directors": [],
            "actors": [],
            "not_favorites": {
                "genres": [],
                "directors": [],
                "actors": []
            }
        }
        # Dicionários para manter contagem de gêneros, diretores e atores
        liked_genres_count = {}
        liked_directors_count = {}
        liked_actors_count = {}
        disliked_genres_count = {}
        disliked_directors_count = {}
        disliked_actors_count = {}
        
        # Contadores de avaliações
        liked_count = 0
        disliked_count = 0

        for movie_id in rated_movies:
            # Tornar a classificação 2 mais difícil de ser atribuída
            rating_probabilities = [0.5, 0.4, 0.1]  # Probabilidades para [0, 1, 2]
            rating = np.random.choice([0, 1, 2], p=rating_probabilities)
            ratings.append({
                "userId": user_id,
                "movieId": movie_id,
                "rating": rating  # Classificação: 0 (não gostou), 1 (gostou), 2 (gostou muito)
            })

            # Atualizar contagens de gêneros, diretores e atores
            movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
            if rating == 2:
                liked_count += 1
                for genre in movie_info["genres"]:
                    liked_genres_count[genre] = liked_genres_count.get(genre, 0) + 1
                for director in movie_info["directors"]:
                    liked_directors_count[director] = liked_directors_count.get(director, 0) + 1
                for actor in movie_info["actors"]:
                    liked_actors_count[actor] = liked_actors_count.get(actor, 0) + 1
            elif rating == 0:
                disliked_count += 1
                for genre in movie_info["genres"]:
                    disliked_genres_count[genre] = disliked_genres_count.get(genre, 0) + 1
                for director in movie_info["directors"]:
                    disliked_directors_count[director] = disliked_directors_count.get(director, 0) + 1
                for actor in movie_info["actors"]:
                    disliked_actors_count[actor] = disliked_actors_count.get(actor, 0) + 1

        # Atualizar favoritos e não favoritos com base nos critérios
        if liked_count >= 10:
            for genre, count in liked_genres_count.items():
                if count / liked_count >= 0.1:
                    user_preferences[user_id]["genres"].append(genre)
            for director, count in liked_directors_count.items():
                if count / liked_count >= 0.1:
                    user_preferences[user_id]["directors"].append(director)
            for actor, count in liked_actors_count.items():
                if count / liked_count >= 0.1:
                    user_preferences[user_id]["actors"].append(actor)

        if disliked_count >= 10:
            for genre, count in disliked_genres_count.items():
                if count / disliked_count >= 0.2:
                    user_preferences[user_id]["not_favorites"]["genres"].append(genre)
            for director, count in disliked_directors_count.items():
                if count / disliked_count >= 0.2:
                    user_preferences[user_id]["not_favorites"]["directors"].append(director)
            for actor, count in disliked_actors_count.items():
                if count / disliked_count >= 0.2:
                    user_preferences[user_id]["not_favorites"]["actors"].append(actor)

    # Remover duplicatas das preferências dos usuários
    for user_id in user_preferences:
        user_preferences[user_id]["genres"] = list(set(user_preferences[user_id]["genres"]))
        user_preferences[user_id]["directors"] = list(set(user_preferences[user_id]["directors"]))
        user_preferences[user_id]["actors"] = list(set(user_preferences[user_id]["actors"]))
        user_preferences[user_id]["not_favorites"]["genres"] = list(set(user_preferences[user_id]["not_favorites"]["genres"]))
        user_preferences[user_id]["not_favorites"]["directors"] = list(set(user_preferences[user_id]["not_favorites"]["directors"]))
        user_preferences[user_id]["not_favorites"]["actors"] = list(set(user_preferences[user_id]["not_favorites"]["actors"]))

    return pd.DataFrame(ratings)

# Carregar dados de filmes e classificações
start_time = time.time()
movies_df = get_movies_from_tmdb()
logging.info(f"Tempo para coletar dados dos filmes: {time.time() - start_time} segundos")

# Log dos filmes coletados
if movies_df.empty:
    logging.error("Erro: Nenhum dado de filmes foi coletado da API do TMDb. Verifique a chave da API e a conexão com a internet.")
    exit()
print(movies_df.head())  # Exibir amostra dos dados dos filmes coletados
ratings_df = get_ratings_from_tmdb()

# Log das classificações dos usuários
logging.info("Classificações dos usuários:")
# for _, row in ratings_df.iterrows():
#     movie_title = movies_df[movies_df['movieId'] == row['movieId']]['title'].values[0]
#     logging.info(f"Usuário ID: {row['userId']}, Filme ID: {row['movieId']}, Título: {movie_title}, Rating: {row['rating']}")
print(ratings_df.head())  # Exibir amostra dos dados das classificações coletadas

# Filtrar apenas os filmes que foram marcados como "gostou" ou "gostou muito" (rating >= 1)
positive_ratings_df = ratings_df[ratings_df['rating'] >= 1]
start_time = time.time()

# Preparar os dados para o Surprise
reader = Reader(rating_scale=(0, 2))
data = Dataset.load_from_df(positive_ratings_df[['userId', 'movieId', 'rating']], reader)

# Ajustar o modelo SVD++ com Grid Search para otimizar hiperparâmetros
logging.info("Iniciando o ajuste do modelo SVD++ com Grid Search para otimização dos hiperparâmetros...")
param_grid = {'n_epochs': [20, 30, 40], 'lr_all': [0.002, 0.005, 0.01], 'reg_all': [0.02, 0.05, 0.1]}
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=5)
gs.fit(data)

# Melhor combinação de hiperparâmetros
best_params = gs.best_params['rmse']
logging.info(f"Melhores hiperparâmetros encontrados: {best_params}")

# Treinar o modelo final com os melhores hiperparâmetros
svdpp = SVDpp(n_epochs=best_params['n_epochs'], lr_all=best_params['lr_all'], reg_all=best_params['reg_all'])
trainset = data.build_full_trainset()
svdpp.fit(trainset)
logging.info(f"Treinamento do modelo final concluído em {time.time() - start_time} segundos.")

# Avaliar o modelo final
logging.info("Avaliando o modelo final...")
cv_results = cross_validate(svdpp, data, measures=['rmse'], cv=5, verbose=True)
average_rmse = np.mean(cv_results['test_rmse'])
logging.info(f"RMSE médio do modelo final: {average_rmse}")
logging.info(f"Tempo para preparar o algoritmo: {time.time() - start_time} segundos")
start_time = time.time()
# Gerar recomendações individuais (removendo filmes já assistidos e duplicados)
logging.info("Gerando recomendações individuais...")
user_ids = positive_ratings_df['userId'].unique()
individual_recommendations = {}
for user_id in user_ids:
    user_rated_movies = user_watch_history.get(user_id, [])
    user_unrated_movies = movies_df[~movies_df['movieId'].isin(user_rated_movies)]
    user_recommendations = []
    seen_movie_ids = set()
    for movie_id in user_unrated_movies['movieId']:
        if movie_id in seen_movie_ids:
            continue
        # Verificar se o filme possui gêneros, diretores ou atores que correspondem às preferências do usuário
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        preference_score = 0
        if any(genre in user_preferences[user_id]['genres'] for genre in movie_info['genres']):
            preference_score += 1
        if any(director in user_preferences[user_id]['directors'] for director in movie_info['directors']):
            preference_score += 1
        if any(actor in user_preferences[user_id]['actors'] for actor in movie_info['actors']):
            preference_score += 1
        # Penalizar caso o filme tenha itens marcados como não favoritos
        if any(genre in user_preferences[user_id]['not_favorites']['genres'] for genre in movie_info['genres']):
            preference_score -= 1
        if any(director in user_preferences[user_id]['not_favorites']['directors'] for director in movie_info['directors']):
            preference_score -= 1
        if any(actor in user_preferences[user_id]['not_favorites']['actors'] for actor in movie_info['actors']):
            preference_score -= 1
        
        pred = svdpp.predict(user_id, movie_id)
        user_recommendations.append((movie_id, pred.est + preference_score * 0.1))
        seen_movie_ids.add(movie_id)
    user_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)
    individual_recommendations[user_id] = user_recommendations

# Aplicar o Least Misery para recomendações em grupo
logging.info("Aplicando a regra de Least Misery para recomendações em grupo...")
group_user_ids = [1, 2, 3]  # Exemplo de um grupo de usuários

# Obter filmes recomendados para cada membro do grupo
group_movies_scores = {}
for user_id in group_user_ids:
    for movie_id, score in individual_recommendations[user_id]:
        # Verificar se o filme foi avaliado negativamente por algum membro do grupo
        ratings_for_movie = ratings_df[(ratings_df['userId'] == user_id) & (ratings_df['movieId'] == movie_id)]
        if not ratings_for_movie.empty and ratings_for_movie.iloc[0]['rating'] == 0:
            continue  # Se algum membro avaliou negativamente, não incluir o filme

        if movie_id not in group_movies_scores:
            group_movies_scores[movie_id] = []
        group_movies_scores[movie_id].append(score)

# Aplicar a regra do "Least Misery" considerando avaliações negativas
group_recommendations = []
for movie_id, scores in group_movies_scores.items():
    least_misery_score = min(scores)
    movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    group_recommendations.append((movie_id, least_misery_score))

# Ordenar recomendações do grupo com base na menor miséria e remover duplicados, mantendo apenas os 20 melhores
seen_movies = set()
group_recommendations_unique = []
for rec in sorted(group_recommendations, key=lambda x: x[1], reverse=True):
    if rec[0] not in seen_movies:
        seen_movies.add(rec[0])
        group_recommendations_unique.append(rec)
    if len(group_recommendations_unique) == 20:
        break

group_recommendations = group_recommendations_unique

# Exibir os títulos dos filmes recomendados para o grupo
logging.info("Exibindo recomendações para o grupo...")
recommended_movie_ids = [movie_id for movie_id, _ in group_recommendations]
recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
recommended_movies = recommended_movies.drop_duplicates(subset='movieId')
logging.info("Recomendações para o grupo:")
print(recommended_movies[['movieId', 'title']])
