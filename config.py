"""
Configuration and Constants for Cloth Customization System

세그멘테이션 마스크 인덱스, 카테고리 매핑, 경로 설정 등을 중앙화
"""

# ============================================================================
# 상황(Situation) 매핑
# ============================================================================
SITUATION_MAPPING = {
    '여행🌊': 'travel',
    '카페☕️': 'cafe',
    '전시회🖼️': 'exhibit',
    '캠퍼스🏫 & 출근💼': 'campus_work',
    '급추위🤧': 'cold',
    '운동💪': 'exercise'
}

# ============================================================================
# 의류 카테고리
# ============================================================================
CLOTH_CATEGORIES = ['top👕', 'bottom👖', 'shoes👞', 'hat🧢', 'sunglasses🕶️', 'scarf🧣', 'bag👜']
CLOTH_NAMES = ['top', 'bottom', 'shoes', 'hat', 'sunglasses', 'scarf', 'bag']

# ============================================================================
# Segformer 세그멘테이션 마스크 인덱스
# Segformer_b2_clothes 모델의 출력 인덱스 정의
# ============================================================================
SEGMENTATION_INDICES = {
    'hat': [1],                          # 모자
    'sunglasses': [3],                   # 안경
    'top': [4],                          # 상의
    'bottom': [5, 6, 7],                # 하의 (바지, 치마, 드레스)
    'belt': [8],                         # 벨트
    'shoes': [9, 10],                    # 신발
    'bag': [16],                         # 가방
    'scarf': [17]                        # 목도리
}

# ============================================================================
# 카테고리별 데이터베이스 파일 매핑
# ============================================================================
CSV_FILES = {
    'top': 'top.csv',
    'bottom': 'bottom.csv',
    'shoes': 'shoes.csv',
    'acc': 'acc.csv'
}

CSV_NAME_FILES = {
    'top': 'top_name.csv',
    'bottom': 'bottom_name.csv',
    'shoes': 'shoes_name.csv',
    'acc': 'acc_name.csv'
}

# ============================================================================
# 경로 설정
# ============================================================================
STYLE_IMAGE_PATH = './style/{situation}/{category}/'  # 스타일 이미지 폴더
PRODUCT_IMAGE_PATH = './product/{category}/'           # 상품 이미지 벡터 폴더
PRODUCT_IMG_PATH = './product/img/'                    # 상품 실제 이미지 폴더
INTRO_IMAGE_PATH = './intro_img/'                      # 소개 이미지

# ============================================================================
# 이미지 처리 파라미터
# ============================================================================
IMAGE_RESIZE_SIZE = (256, 256)           # 벡터화 시 이미지 리사이즈 크기
OBJECT_DETECTION_START_THRESHOLD = 1.0   # Object Detection 초기 임계값
OBJECT_DETECTION_MIN_THRESHOLD = 0.0     # 최소 임계값
OBJECT_DETECTION_STEP = -0.05            # 임계값 감소 스텝
