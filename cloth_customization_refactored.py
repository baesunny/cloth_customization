"""
의류 추천 시스템 Streamlit 앱

사용자가 업로드한 의류 이미지와 상황, 카테고리를 바탕으로
트렌디한 스타일을 추천해주는 애플리케이션
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os

# 로컬 모듈 임포트
from config import (
    SITUATION_MAPPING, CLOTH_CATEGORIES, CLOTH_NAMES,
    STYLE_IMAGE_PATH, PRODUCT_IMAGE_PATH, PRODUCT_IMG_PATH,
    INTRO_IMAGE_PATH, CSV_FILES, CSV_NAME_FILES
)
from utils import (
    segment_image, extract_clothing_mask,
    crop_clothing_from_mask, image_to_vector,
    cosine_similarity, cosine_similarity_2
)


# ============================================================================
# UI 유틸리티 함수
# ============================================================================

def center_image(image_path, width=700):
    """이미지를 중앙에 표시"""
    st.markdown(
        f'<style>img {{ display: block; margin-left: auto; margin-right: auto; }} </style>',
        unsafe_allow_html=True
    )
    st.image(image_path, width=width)


# ============================================================================
# 데이터 로드 함수
# ============================================================================

@st.cache_resource
def load_product_data():
    """상품 데이터 로드 (캐싱)"""
    data = {}
    names = {}
    
    for cat in ['top', 'bottom', 'shoes', 'acc']:
        try:
            data[cat] = pd.read_csv(CSV_FILES[cat])
            names[cat] = pd.read_csv(CSV_NAME_FILES[cat])
        except Exception as e:
            st.warning(f"데이터 로드 실패: {cat} - {e}")
    
    return data, names


@st.cache_resource
def load_segmentation_model():
    """세그멘테이션 모델 로드 (캐싱)"""
    from utils.segmentation import SegmentationModel
    return SegmentationModel()


# ============================================================================
# 의류 추출 및 벡터화 함수
# ============================================================================

def process_input_image(image, input_cat):
    """
    입력 이미지를 세그멘테이션, 크롭, 벡터화
    
    Args:
        image (PIL.Image): 입력 이미지
        input_cat (str): 의류 카테고리
    
    Returns:
        np.ndarray: 이미지 벡터
    """
    # 세그멘테이션
    image_np = np.array(image)
    pred_seg = segment_image(image)
    
    # 의류 부위 추출
    masked_image = extract_clothing_mask(image_np, pred_seg, input_cat)
    
    # 객체 탐지 및 크롭
    cropped_image = crop_clothing_from_mask(masked_image)
    
    # 벡터화
    image_vector = image_to_vector(cropped_image)
    
    return image_vector


def find_matching_style(input_vector, situation, input_cat):
    """
    입력 이미지와 유사한 스타일 찾기
    
    Args:
        input_vector (np.ndarray): 입력 이미지 벡터
        situation (str): 상황
        input_cat (str): 입력 카테고리
    
    Returns:
        str: 매칭 벡터 파일명
    """
    # 스타일 이미지 디렉토리
    style_dir = STYLE_IMAGE_PATH.format(situation=situation, category=input_cat)
    
    # 모든 스타일 벡터와 비교
    similarities = []
    for filename in os.listdir(style_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(style_dir, filename)
            sim = cosine_similarity(input_vector, file_path)
            similarities.append((filename, sim))
    
    # 유사도 가장 높은 것 반환
    best_match = max(similarities, key=lambda x: x[1])[0]
    return style_dir + best_match


def find_matching_product(style_vector_path, output_cat):
    """
    스타일 벡터와 유사한 상품 찾기
    
    Args:
        style_vector_path (str): 스타일 벡터 파일 경로
        output_cat (str): 출력 카테고리
    
    Returns:
        str: 최적 상품 벡터 파일명
    """
    # 상품 벡터 디렉토리
    product_dir = PRODUCT_IMAGE_PATH.format(category=output_cat)
    
    # 모든 상품 벡터와 비교
    similarities = []
    for filename in os.listdir(product_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(product_dir, filename)
            sim = cosine_similarity_2(style_vector_path, file_path)
            similarities.append((filename, sim))
    
    # 유사도 가장 높은 것 반환
    best_match = max(similarities, key=lambda x: x[1])[0]
    return best_match


# ============================================================================
# 상품 정보 조회 함수
# ============================================================================

def get_product_info(output_name, output_cat, product_data, product_names):
    """
    상품 정보 조회
    
    Args:
        output_name (str): 상품 벡터 파일명 (예: 'bottom_1883.txt')
        output_cat (str): 상품 카테고리
        product_data (dict): 상품 데이터
        product_names (dict): 상품명 매핑
    
    Returns:
        tuple: (상품이미지경로, 상품명, 가격)
    """
    # 카테고리별 데이터프레임 선택
    if output_cat == 'bottom':
        df = product_data['bottom']
        df_name = product_names['bottom']
    elif output_cat == 'top':
        df = product_data['top']
        df_name = product_names['top']
    elif output_cat == 'shoes':
        df = product_data['shoes']
        df_name = product_names['shoes']
    else:  # hat, sunglasses, scarf, bag, belt
        df = product_data['acc']
        df_name = product_names['acc']
    
    # 상품명 추출
    output_name = output_name.split('.')[0]
    try:
        file_name = df_name[df_name['index'] == output_name].iloc[0, 1]
        final = df[df['id'] == file_name]
        
        name = final['name'].values[0].split('\n')[-1]
        price = final['price'].values[0]
        image_path = PRODUCT_IMG_PATH + output_cat + '/' + file_name
        
        return image_path, name, price
    except Exception as e:
        st.error(f"상품 정보 조회 실패: {e}")
        return None, None, None


# ============================================================================
# 메인 앱
# ============================================================================

def main():
    """Streamlit 앱 메인 함수"""
    
    # 페이지 설정
    st.set_page_config(page_title="👚 오늘 뭐입지?!", layout="wide")
    
    # ========================================================================
    # 1. 소개 섹션
    # ========================================================================
    st.header('👚 오늘 뭐입지?! 👕')
    st.markdown('💬 : 🚨 **설마 너 지금.. 그렇게 입고 나가게?** 🚨')
    st.markdown(
        '**패션센스가 2% 부족한 당신을 위해 준비했습니다!** '
        '사진 이미지만 입력하면, 요즘 트렌디한 스타일과 여러분의 TPO를 고려하여 '
        '코디를 추천해드립니다. 무신사와 온더룩의 패셔니스타들의 코디를 지금 바로 참고해보세요!'
    )
    center_image(os.path.join(INTRO_IMAGE_PATH, 'fashionista.jpg'))
    
    st.markdown('---')
    st.subheader('PROCESS')
    center_image(os.path.join(INTRO_IMAGE_PATH, 'process.png'))
    st.markdown('---')
    
    # ========================================================================
    # 2. 입력 섹션
    # ========================================================================
    st.subheader('✅ 의류 이미지 업로드')
    input_image = st.file_uploader(
        "**의류 이미지를 업로드하세요. (배경이 깔끔한 사진이라면 더 좋습니다!)**",
        type=['png', 'jpg', 'jpeg']
    )
    
    if not input_image:
        st.stop()
    
    center_image(input_image, 400)
    st.markdown('---')
    
    # 입력 카테고리 선택
    st.subheader('✅ 업로드한 의류 이미지 카테고리 선택')
    input_cat_selected = st.radio(
        "**귀하가 업로드한 의류 이미지의 카테고리를 골라주세요.**",
        CLOTH_CATEGORIES,
        index=None,
        horizontal=True
    )
    
    if not input_cat_selected:
        st.stop()
    
    input_cat = input_cat_selected[:-1]  # 이모지 제거
    st.write('✓ Selected:', input_cat_selected)
    st.markdown('---')
    
    # 출력 카테고리 선택
    st.subheader('✅ 추천받고 싶은 의류 카테고리 선택')
    output_cat_selected = st.radio(
        '**추천받고 싶은 의류 카테고리를 선택해주세요.**',
        CLOTH_CATEGORIES,
        index=None,
        horizontal=True
    )
    
    if not output_cat_selected:
        st.write('🚫 주의: 업로드한 의류 카테고리와 다른 카테고리를 선택해주세요.')
        st.stop()
    
    output_cat = output_cat_selected[:-1]  # 이모지 제거
    st.write('✓ Selected:', output_cat_selected)
    st.markdown('---')
    
    # 상황 선택
    st.subheader('✅ 상황 카테고리 선택')
    situation_selected = st.radio(
        "**상황 카테고리를 선택해주세요.**",
        list(SITUATION_MAPPING.keys()),
        captions=['(바다, 여행)', '(카페, 데일리)', '(데이트, 결혼식)', '(캠퍼스, 출근)', '(추위)', '(운동)'],
        index=None,
        horizontal=True
    )
    
    if not situation_selected:
        st.stop()
    
    situation = SITUATION_MAPPING[situation_selected]
    st.write('✓ Selected:', situation_selected)
    st.markdown('---')
    
    # ========================================================================
    # 3. 처리 및 추천 섹션
    # ========================================================================
    st.subheader('⏳ 처리 중...')
    
    # 데이터 로드
    product_data, product_names = load_product_data()
    
    with st.spinner('이미지를 분석하고 최적의 상품을 찾고 있습니다...'):
        # 입력 이미지 처리
        input_pil_image = Image.open(input_image)
        input_vector = process_input_image(input_pil_image, input_cat)
        
        # 매칭 스타일 찾기
        style_vector_path = find_matching_style(input_vector, situation, input_cat)
        
        # 최적 상품 찾기
        output_name = find_matching_product(style_vector_path, output_cat)
        
        # 상품 정보 조회
        image_path, product_name, price = get_product_info(
            output_name, output_cat, product_data, product_names
        )
    
    # ========================================================================
    # 4. 결과 표시
    # ========================================================================
    st.subheader('OUTPUT ✨')
    
    if image_path and os.path.exists(image_path):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            try:
                result_image = Image.open(image_path)
                st.image(result_image, width=400)
            except Exception as e:
                st.error(f"이미지 로드 실패: {e}")
        
        with col3:
            st.caption(f"**상품명**: {product_name}")
            st.caption(f"**가격**: {price}")
    else:
        st.error("상품 이미지를 찾을 수 없습니다.")


if __name__ == "__main__":
    main()
