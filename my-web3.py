import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 문장 분리
def split_sentences(text):
    text = text.replace("\n", " ")
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 0] #리스트 컴프리헨션


# TF-IDF
def TF_IDF(sentences):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    return cosine_similarity(tfidf)

#cosine similarity
def cosine_similarity(matrix):
    n = matrix.shape[0] #문서 개수

    similarity_matrix = np.zeros((n, n))

    #모든 단어 비교
    for i in range(n):
        for j in range(n):

            vector1 = matrix[i].toarray()[0]#TF-IDF는 희소 행렬이라 메모리 절약되는 array로 변환, 2차원이라 [0]으로 첫행 꺼냄
            vector2 = matrix[j].toarray()[0]

            # 내적
            dot_product = np.dot(vector1, vector2)

            # 유클리드 노름
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)

            # 0 나누기 방지
            if magnitude1 == 0 or magnitude2 == 0:
                similarity = 0
            else:
                similarity = dot_product / (magnitude1 * magnitude2) #내적을 노름으로 나눠서 길이 영향 제거

            similarity_matrix[i][j] = similarity

    return similarity_matrix

#PageRank
def pagerank(sim_matrix, d=0.85, max_iter=100, tol=1e-6): #댐핑 팩터를 통해 문장 간 유사도를 얼마나 신뢰할지 결정
    n = sim_matrix.shape[0]

    # 초기 점수 (균등 분포)
    scores = np.ones(n) / n

    # 행 정규화
    row_sums = sim_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    norm_matrix = sim_matrix / row_sums[:, None]

    # 반복 계산
    for _ in range(max_iter):
        new_scores = (1 - d) / n + d * norm_matrix.T.dot(scores)

        # 수렴 체크(허용 오차 내 브레이크)
        if np.linalg.norm(new_scores - scores) < tol:
            break

        scores = new_scores

    return scores


#TextRank
def summarize(text, top_n=3):
    sentences = split_sentences(text)

    if len(sentences) <= top_n:
        return sentences

    sim_matrix = TF_IDF(sentences)

    scores = pagerank(sim_matrix)

    ranked = sorted(
        ((scores[i], s) for i, s in enumerate(sentences)),
        reverse=True
    )

    return [s for _, s in ranked[:top_n]]
#UI
st.title(":brain: TextRank 요약기 ")

text = st.text_area("텍스트 입력", height=300)

top_n = st.slider("요약 문장 개수", 1, 10, 3)

if st.button("요약"):
    if text.strip():
        result = summarize(text, top_n)

        st.subheader(":pushpin:요약 결과")
        for i, r in enumerate(result, 1):
            st.write(f"{i}. {r}")
    else:
        st.warning("텍스트를 입력하세요.")