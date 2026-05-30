"""
이미지 벡터화 모듈

이미지를 1D 벡터로 변환하여 유사도 분석 가능하게 함
"""

import numpy as np
from PIL import Image
from config import IMAGE_RESIZE_SIZE


def image_to_vector(image, resize_size=IMAGE_RESIZE_SIZE):
    """
    이미지를 1D 벡터로 변환
    
    이미지를 고정 크기로 리사이즈한 뒤 1D 배열로 변환하여
    유사도 계산에 사용할 수 있는 벡터로 만든다.
    
    Args:
        image (PIL.Image or np.ndarray): 입력 이미지
        resize_size (tuple): 리사이징 크기 (기본값: 256x256)
    
    Returns:
        np.ndarray: 정규화되지 않은 1D 벡터 (플로트 타입)
    
    Example:
        >>> image = Image.open('cloth.jpg')
        >>> vector = image_to_vector(image)
        >>> vector.shape
        (196608,)  # 256 * 256 * 3
    """
    # PIL Image로 변환
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.copy(image))
    
    # RGB 모드 확인 (그레이스케일을 RGB로 변환)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 지정된 크기로 리사이징
    image = image.resize(resize_size)
    
    # NumPy 배열로 변환 (float32)
    image_array = np.array(image, dtype=np.float32)
    
    # 1D 벡터로 변환 (flatten)
    image_vector = image_array.flatten()
    
    return image_vector


def load_vector_from_file(file_path):
    """
    파일에서 저장된 벡터 로드
    
    Args:
        file_path (str): 벡터 파일 경로 (txt 포맷)
    
    Returns:
        np.ndarray: 로드된 벡터
    """
    return np.loadtxt(file_path)


def save_vector_to_file(vector, file_path):
    """
    벡터를 파일에 저장
    
    Args:
        vector (np.ndarray): 저장할 벡터
        file_path (str): 저장할 파일 경로
    """
    np.savetxt(file_path, vector)


def normalize_vector(vector):
    """
    벡터를 정규화 (L2 정규화)
    
    Args:
        vector (np.ndarray): 입력 벡터
    
    Returns:
        np.ndarray: 정규화된 벡터
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
