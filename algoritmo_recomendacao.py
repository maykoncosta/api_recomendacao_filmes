import numpy as np
import pandas as pd
from surprise import Dataset, SVDpp, Reader
from surprise.model_selection import cross_validate, GridSearchCV
import logging
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Inicializar o Firebase Admin SDK
cred = credentials.Certificate('C:\movie-recommender\clube-do-filme-firebase-cred.json')  # Substitua pelo caminho para o seu arquivo JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

# Configurar o logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Funções para carregar dados do Firestore
def get_movies_from_firestore():
    movies = []
    movies_ref = db.collection('movies')
    docs = movies_ref.stream()
    for doc in docs:
        data = doc.to_dict()
        movies.append({
            'movieId': data['movieId'],
            'title': data['title'],
            'genres': [genre['id'] for genre in data['genres']],        # IDs dos gêneros
            'directors': [director['id'] for director in data['directors']],  # IDs dos diretores
            'actors': [actor['id'] for actor in data['actors']]         # IDs dos atores
        })
    return pd.DataFrame(movies)

def get_user_data_from_firestore():
    global user_watch_history, user_preferences
    user_watch_history = {}
    user_preferences = {}
    ratings = []

    users_ref = db.collection('users')
    docs = users_ref.stream()
    for doc in docs:
        data = doc.to_dict()
        email = data['email']  # Usar o email como identificador
        watch_history = data.get('watch_history', [])
        user_watch_history[email] = watch_history

        # Obter as classificações do usuário
        user_ratings = data.get('ratings', {})
        for movie_id, rating in user_ratings.items():
            ratings.append({
                'email': email,
                'movieId': int(movie_id),
                'rating': rating
            })

        # Carregar as preferências do usuário com os novos nomes de campos
        user_preferences[email] = {
            "favoriteGenres": data.get('favoriteGenres', []),
            "favoriteDirectors": data.get('favoriteDirectors', []),
            "favoriteActors": data.get('favoriteActors', []),
            "favoriteMovies": data.get('favoriteMovies', []),
            "notFavorites": {
                "genres": data.get('notFavoriteGenres', []),
                "directors": data.get('notFavoriteDirectors', []),
                "actors": data.get('notFavoriteActors', []),
                "movies": data.get('notFavoriteMovies', [])
            }
        }

    return pd.DataFrame(ratings)

# Carregar dados de filmes e classificações
start_time = time.time()
movies_df = get_movies_from_firestore()
logging.info(f"Tempo para carregar dados dos filmes: {time.time() - start_time} segundos")

# Verificar se os dados dos filmes foram carregados
if movies_df.empty:
    logging.error("Erro: Nenhum dado de filmes foi carregado do Firestore.")
    exit()
print(movies_df.head())  # Exibir amostra dos dados dos filmes carregados

ratings_df = get_user_data_from_firestore()

# Verificar se as classificações dos usuários foram carregadas
if ratings_df.empty:
    logging.error("Erro: Nenhum dado de classificações foi carregado do Firestore.")
    exit()
print(ratings_df.head())  # Exibir amostra das classificações dos usuários

# Filtrar apenas os filmes que foram marcados como "gostou" ou "gostou muito" (rating >= 1)
positive_ratings_df = ratings_df[ratings_df['rating'] >= 1]
start_time = time.time()

# Preparar os dados para o Surprise
reader = Reader(rating_scale=(0, 2))
# Renomear 'email' para 'userId' para compatibilidade com o Surprise
positive_ratings_df = positive_ratings_df.rename(columns={'email': 'userId'})
data = Dataset.load_from_df(positive_ratings_df[['userId', 'movieId', 'rating']], reader)

# Ajustar o modelo SVD++ com Grid Search para otimizar hiperparâmetros
logging.info("Iniciando o ajuste do modelo SVD++ com Grid Search para otimização dos hiperparâmetros...")
param_grid = {'n_epochs': [20], 'lr_all': [0.005], 'reg_all': [0.02]}  # Valores simplificados para acelerar o treinamento
gs = GridSearchCV(SVDpp, param_grid, measures=['rmse'], cv=3)
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
cv_results = cross_validate(svdpp, data, measures=['rmse'], cv=3, verbose=True)
average_rmse = np.mean(cv_results['test_rmse'])
logging.info(f"RMSE médio do modelo final: {average_rmse}")
logging.info(f"Tempo para preparar o algoritmo: {time.time() - start_time} segundos")
start_time = time.time()

# Gerar recomendações individuais
logging.info("Gerando recomendações individuais...")
user_emails = positive_ratings_df['userId'].unique()
individual_recommendations = {}
for email in user_emails:
    user_rated_movies = user_watch_history.get(email, [])
    user_unrated_movies = movies_df[~movies_df['movieId'].isin(user_rated_movies)]
    user_recommendations = []
    seen_movie_ids = set()
    for _, movie_row in user_unrated_movies.iterrows():
        movie_id = movie_row['movieId']
        if movie_id in seen_movie_ids:
            continue
        # Verificar se o filme possui gêneros, diretores ou atores que correspondem às preferências do usuário
        movie_genres = movie_row['genres']
        movie_directors = movie_row['directors']
        movie_actors = movie_row['actors']
        preference_score = 0
        user_prefs = user_preferences[email]

        # Incrementar o score se o filme tiver gêneros favoritos
        if any(genre in user_prefs['favoriteGenres'] for genre in movie_genres):
            preference_score += 1
        # Incrementar o score se o filme tiver diretores favoritos
        if any(director in user_prefs['favoriteDirectors'] for director in movie_directors):
            preference_score += 1
        # Incrementar o score se o filme tiver atores favoritos
        if any(actor in user_prefs['favoriteActors'] for actor in movie_actors):
            preference_score += 1
        # Incrementar o score se o filme estiver na lista de filmes favoritos
        if movie_id in user_prefs['favoriteMovies']:
            preference_score += 2  # Damos um peso maior para filmes favoritos

        # Penalizar caso o filme tenha itens marcados como não favoritos
        not_favorites = user_prefs['notFavorites']
        if any(genre in not_favorites['genres'] for genre in movie_genres):
            preference_score -= 1
        if any(director in not_favorites['directors'] for director in movie_directors):
            preference_score -= 1
        if any(actor in not_favorites['actors'] for actor in movie_actors):
            preference_score -= 1
        if movie_id in not_favorites.get('movies', []):
            preference_score -= 2  # Penalidade maior para filmes não favoritos

        pred = svdpp.predict(email, movie_id)
        user_recommendations.append((movie_id, pred.est + preference_score * 0.1))
        seen_movie_ids.add(movie_id)
    user_recommendations = sorted(user_recommendations, key=lambda x: x[1], reverse=True)
    individual_recommendations[email] = user_recommendations

# Renomear a coluna 'userId' de volta para 'email'
ratings_df = ratings_df.rename(columns={'userId': 'email'})

# Aplicar o Least Misery para recomendações em grupo
logging.info("Aplicando a regra de Least Misery para recomendações em grupo...")
group_user_emails = ['usuario11@exemplo.com', 'usuario12@exemplo.com', 'usuario13@exemplo.com']  # Lista de emails dos usuários do grupo

# Obter filmes recomendados para cada membro do grupo
group_movies_scores = {}
for email in group_user_emails:
    for movie_id, score in individual_recommendations[email]:
        # Verificar se o filme foi avaliado negativamente por algum membro do grupo
        ratings_for_movie = ratings_df[(ratings_df['email'] == email) & (ratings_df['movieId'] == movie_id)]
        if not ratings_for_movie.empty and ratings_for_movie.iloc[0]['rating'] == 0:
            continue  # Se algum membro avaliou negativamente, não incluir o filme

        if movie_id not in group_movies_scores:
            group_movies_scores[movie_id] = []
        group_movies_scores[movie_id].append(score)

# Aplicar a regra do "Least Misery"
group_recommendations = []
for movie_id, scores in group_movies_scores.items():
    least_misery_score = min(scores)
    group_recommendations.append((movie_id, least_misery_score))

# Ordenar e selecionar as top 20 recomendações
group_recommendations = sorted(group_recommendations, key=lambda x: x[1], reverse=True)[:20]

# Exibir os títulos dos filmes recomendados para o grupo
logging.info("Exibindo recomendações para o grupo...")
recommended_movie_ids = [movie_id for movie_id, _ in group_recommendations]
recommended_movies = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
recommended_movies = recommended_movies.drop_duplicates(subset='movieId')
print(recommended_movies[['movieId', 'title']])
