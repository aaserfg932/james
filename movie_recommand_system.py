import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', 70)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score


model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(model_url)
print('Model Loaded')

def embed(texts):
    return model(texts)

movie_data = pd.read_csv("Top_10000_Movies.csv", engine="python")
movie_data.head()
movie_data = movie_data.drop_duplicates()
movie_data = movie_data.dropna(subset = ['overview'])
movie_data = movie_data.dropna(subset = ['original_title'])
movie_data = movie_data.dropna(subset = ['genre'])
movie_data = movie_data.dropna(subset = ['original_language'])
movie_data = movie_data.reset_index()

#特定語言的子矩陣
#movie_data = movie_data[movie_data['original_language'] == 'zh']
#movie_data = movie_data[movie_data['original_language'] == 'en']

movie_data = movie_data.reset_index(drop=True)

# 將類型列表轉換為字符串
def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        if isinstance(genres_list, list):
            return ' '.join(genres_list)
        return genres_str
    except (ValueError, SyntaxError):
        return genres_str

movie_data['genre'] = movie_data['genre'].apply(parse_genres)
movie_data['original_title_lower'] = movie_data['original_title'].str.lower()
indices = pd.Series(movie_data.index, index=movie_data['original_title_lower']).to_dict()
movie_data['original_title_lower'] = movie_data['original_title_lower'].fillna('')
movie_data['genre_lower'] = movie_data['genre'].str.lower()
indices = pd.Series(movie_data.index, index=movie_data['genre_lower']).to_dict()
movie_data['genre_lower'] = movie_data['genre_lower'].fillna('')

#TFIDF
movie_data['content'] = (movie_data['original_title_lower'] + ' ' +
                         movie_data['genre_lower']+ ' '+
                         movie_data['overview'])

#Universal-sentence-encoder
movie_data['content2'] = (movie_data['original_title'] + ' ' +
                         movie_data['genre'] + ' ' +
                         movie_data['overview'])

# TF-IDF Vectorization 
vectorizer_titandgen = TfidfVectorizer()
tfidf_matrix = vectorizer_titandgen.fit_transform(movie_data['content'])
print('The tfidf_matrix shape is:', tfidf_matrix.shape)

# Universal Sentence Encoder Vectorization 
embeddings = embed(list(movie_data['content2']))
print('The embedding shape is:', embeddings.shape)


#印出候選電影的 1.tfidf向量相似度 2.tfidf向量符合度 3.embed(uni)後向量間距離
def print_candidate(title, index):
    title_lower = [title.lower()]  # 將輸入的內容轉換為小寫
    search_vector = vectorizer_titandgen.transform(title_lower)
    search_vector.data = np.ones_like(search_vector.data)
    search_vector_ones=search_vector
    search_vector = vectorizer_titandgen.transform(title_lower)
    emb_search = embed([title])

    for i in range(len(index)):
      ind=int(index[i])
      content_vector=tfidf_matrix[ind,:]
      if search_vector.count_nonzero() != 0:
        sim_score=round(float(linear_kernel(content_vector, search_vector).item()),5)
      else:
        sim_score=0

      content_vector.data = np.ones_like(content_vector.data)
      if search_vector.count_nonzero() != 0:
        match_score=round(float(linear_kernel(search_vector_ones,content_vector).item())/search_vector.count_nonzero(),5)
      else:
        match_score=0

      emb_content=embeddings[ind,:]
      emb_content=embeddings[ind,:]
      dist=round(float(np.linalg.norm(emb_search-emb_content)),5)

      print(f"第{i+1}位候選電影({ind})的: 1.tfidf向量相似度 = {sim_score}  2.tfidf向量符合度 = {match_score}  3.embed(uni)後向量間距離 = {dist}")

      print("該電影為:", movie_data.iloc[ind]['original_title'])

def get_recommendations(title, content_matrix=tfidf_matrix, vectorizer=vectorizer_titandgen):
    title_lower = [title.lower()]  # 將輸入的內容轉換為小寫
    search_vector=vectorizer.transform(title_lower) #將輸入的內容 TF-IDF 向量化
    search_kernel=linear_kernel(content_matrix, search_vector)  #計算輸入的內容與'content'的相似度

    #計算相似度非0的content共有幾筆
    matches = np.nonzero(search_kernel)
    match_len=len(matches[0])

    # 按照相似度排序，並選擇前10部電影(也有可能相似非0筆數不足10筆)
    indexed_search_kernel = list(enumerate(search_kernel))
    sim_scores = sorted(indexed_search_kernel, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:min(10,match_len)]  #獲取最相似的前10部電影

    # 提取電影索引
    movie_indices = [index for index, value  in sim_scores]

    #相似度過低排除(主要懲罰塞一堆無關字的情況)
    search_vector.data = np.ones_like(search_vector.data)
    print("有效token數 = ",round(search_vector.sum()))
    threshold = 1/2
    movie_indices_new=[]
    print(f"\ntfidf向量化判定 符合度閥值 = {threshold}")
    print_candidate(title, movie_indices)
    for i in range(len(movie_indices)):
      ind=int(movie_indices[i])
      content_vector=content_matrix[ind,:]
      content_vector.data = np.ones_like(content_vector.data)
      if linear_kernel(search_vector,content_vector)/search_vector.count_nonzero() >= threshold:  #至少需要包含一半輸入內容有的token才算有效搜尋
        movie_indices_new.append(ind)

    movie_indices=movie_indices_new
    match_len_new=len(movie_indices)
    if   match_len_new != 0:
      print(f"\n入選的有{match_len_new}筆電影: ", end="")
      for i in range(match_len_new):
        print(f"  {movie_indices[i]}",end="")
      #print("\n")
    elif match_len != 0:
      print("\n無入選電影")
    else:
      print("無候選電影")

    #如果相似度非0的content不足10筆時，檢查是否是因為輸入標題時忘記空格(也可能是原標題沒空格)
    #if   match_len_new < 10:
    title_lower = title.lower().strip()
    matches_2 = movie_data[movie_data['original_title_lower'].str.contains(title.lower().strip())]
    idx = matches_2.index
    for i in range((len(idx))):
      if idx[i] not in movie_indices:
        movie_indices.append(idx[i])

    print("\n----------------------------------------------------------------------------------------")
    print("字串直接判定")
    if len(idx) != 0:
      print_candidate(title, idx)
    else:
      print("無候選電影")

    #movie_indices=movie_indices[:10-max(min(10,match_len)+match_len_new,9)]
    movie_indices=movie_indices[:10]

    # 返回推薦的電影信息
    #return movie_data.iloc[combined_vector][['original_title', 'overview']], len(combined_vector)
    return movie_indices, len(movie_indices)

def recommend(text):
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(embeddings)
    emb = embed([text])
    neighbors = nn.kneighbors(emb, return_distance=False)[0]

    print("\n----------------------------------------------------------------------------------------")
    print("universal-sentence-encoder後距離判定")
    print_candidate(text, neighbors)

    return neighbors

#用KNN找出距離輸入的關鍵字最近的10個單位

def main():

        title = input("請輸入電影標題/類型/概述: ")
        recommendations, num_match= get_recommendations(title, tfidf_matrix, vectorizer_titandgen)
        #print(num_match)
        #print(recommendations)
        #if  num_match < 10:
        recommendations2=recommend(title)
        if len(recommendations2) !=0:
          for i in range(10):
            if recommendations2[i] not in recommendations:#TFIDF裡面沒有的推薦但universal encode有就加進去TFIDF的清單內
              recommendations.append(recommendations2[i])
        recommendations=recommendations[:10]

        print("\n----------------------------------------------------------------------------------------")
        print('Recommended Movies:')
        print(movie_data.iloc[recommendations][['original_title', 'overview']])

        #print(f"num_match = {num_match}")


main()

