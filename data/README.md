# 데이터 안내

본 저장소에는 **크롤링 원본·상품 이미지·벡터 파일은 포함하지 않는다.** 무신사·온더룩 등 외부 사이트에서 수집한 데이터는 용량·저작권 문제로 GitHub에 올리지 않았다.

Streamlit 앱(`cloth_customization_refactored.py`)을 실행하려면 아래 구조로 `data/` 디렉터리를 준비해야 한다.

```text
data/
├── top.csv, bottom.csv, shoes.csv, acc.csv          # 상품 메타데이터
├── top_name.csv, bottom_name.csv, shoes_name.csv, acc_name.csv
├── style/{situation}/{category}/*.txt               # TPO·카테고리별 스타일 벡터
└── product/{category}/*.txt                         # 카테고리별 상품 벡터
    product/img/{category}/*                         # 상품 썸네일 이미지
```

- `situation`: `travel`, `cafe`, `exhibit`, `campus_work`, `cold`, `exercise`
- `category`: `top`, `bottom`, `shoes`, `hat`, `sunglasses`, `scarf`, `bag`

벡터 생성·크롤링 절차는 `notebooks/03_web_crawling_onthelook.ipynb`, `notebooks/02_segmentation_to_similarity.ipynb`를 참고한다.
