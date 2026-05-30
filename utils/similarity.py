"""
유사도 분석 모듈

코사인 유사도를 사용한 이미지 간 유사도 계산
"""

import numpy as np
from .vector import load_vector_from_file, normalize_vector


def cosine_similarity(vec1, vec2_path):
    """
    두 벡터의 코사인 유사도 계산 (벡터 vs 파일)
    
    코사인 유사도는 두 벡터 사이의 각도를 기반으로 유사도를 계산합니다.
    범위: -1 ~ 1 (보통 0 ~ 1, 1에 가까울수록 유사)
    
    Formula:
        similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)
    
    Args:
        vec1 (np.ndarray): 첫 번째 벡터 (메모리)
        vec2_path (str): 두 번째 벡터 파일 경로
    
    Returns:
        float: 코사인 유사도 점수 (0 ~ 1 범위, 보정됨)
    
    Example:
        >>> vec1 = np.random.rand(256*256*3)
        >>> sim = cosine_similarity(vec1, './vectors/image.txt')
        >>> print(sim)
        0.95
    """
    # 파일에서 벡터 로드
    vec2 = load_vector_from_file(vec2_path)
    
    # 벡터 곱(dot product) 계산
    dot_product = np.dot(vec1, vec2)
    
    # 각 벡터의 노름(norm) 계산
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 0으로 나누는 것 방지
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    # 코사인 유사도 계산
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return float(similarity)


def cosine_similarity_2(vec1_path, vec2_path):
    """
    두 벡터의 코사인 유사도 계산 (파일 vs 파일)
    
    Args:
        vec1_path (str): 첫 번째 벡터 파일 경로
        vec2_path (str): 두 번째 벡터 파일 경로
    
    Returns:
        float: 코사인 유사도 점수
    
    Example:
        >>> sim = cosine_similarity_2('./vectors/style.txt', './vectors/product.txt')
        >>> print(sim)
        0.92
    """
    # 두 파일에서 벡터 로드
    vec1 = load_vector_from_file(vec1_path)
    vec2 = load_vector_from_file(vec2_path)
    
    # 벡터 곱 계산
    dot_product = np.dot(vec1, vec2)
    
    # 각 벡터의 노름 계산
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # 0으로 나누는 것 방지
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    # 코사인 유사도 계산
    similarity = dot_product / (norm_vec1 * norm_vec2)
    
    return float(similarity)


def find_most_similar(vec, vector_dir, top_k=1):
    """
    주어진 벡터와 가장 유사한 벡터들 찾기
    
    Args:
        vec (np.ndarray): 쿼리 벡터
        vector_dir (str): 벡터 파일 디렉토리
        top_k (int): 반환할 상위 k개 (기본값: 1)
    
    Returns:
        list: [(파일명, 유사도), ...] 유사도 높은 순서로 정렬
    
    Example:
        >>> results = find_most_similar(vec, './vectors/tops/')
        >>> print(results[0])
        ('top_123.txt', 0.97)
    """
    import os
    
    similarities = []
    
    # 디렉토리의 모든 벡터 파일과 비교
    for filename in os.listdir(vector_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(vector_dir, filename)
            try:
                sim = cosine_similarity(vec, file_path)
                similarities.append((filename, sim))
            except Exception as e:
                print(f"Warning: 파일 {filename} 처리 중 오류: {e}")
                continue
    
    # 유사도 높은 순서로 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 상위 k개 반환
    return similarities[:top_k]


def batch_similarity_analysis(query_vec, vector_files):
    """
    배치 유사도 분석
    
    Args:
        query_vec (np.ndarray): 쿼리 벡터
        vector_files (list): 비교할 벡터 파일 경로 리스트
    
    Returns:
        np.ndarray: 유사도 점수 배열
    """
    similarities = []
    
    for vec_file in vector_files:
        try:
            sim = cosine_similarity(query_vec, vec_file)
            similarities.append(sim)
        except Exception as e:
            print(f"Warning: {vec_file} 처리 중 오류: {e}")
            similarities.append(0.0)
    
    return np.array(similarities)
