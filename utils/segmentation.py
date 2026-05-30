"""
이미지 세그멘테이션 모듈

Segformer 모델을 사용한 의류 세그멘테이션
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from config import SEGMENTATION_INDICES


class SegmentationModel:
    """Segformer를 사용한 의류 세그멘테이션 모델"""
    
    def __init__(self):
        """
        사전 학습된 Segformer 모델 초기화
        
        Model: mattmdjaga/segformer_b2_clothes
        - 의류 세그멘테이션 특화 모델
        - 18개 카테고리 분류 (hat, sunglasses, top, bottom 등)
        """
        self.processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
        self.model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    
    def segment(self, image):
        """
        이미지 세그멘테이션 수행
        
        Args:
            image (PIL.Image): 입력 이미지
        
        Returns:
            torch.Tensor: 세그멘테이션 결과 (H x W 크기의 마스크)
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits.cpu()
        
        # 로짓을 입력 이미지 크기로 업샘플링
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        # 각 픽셀별 가장 확률 높은 클래스 선택
        pred_seg = upsampled_logits.argmax(dim=1)[0]
        return pred_seg


def segment_image(image):
    """
    편의 함수: 이미지를 세그멘테이션
    
    Args:
        image (PIL.Image or np.ndarray): 입력 이미지
    
    Returns:
        torch.Tensor: 세그멘테이션 마스크
    """
    # 그레이스케일 이미지를 RGB로 변환
    if isinstance(image, np.ndarray) and len(image.shape) == 2:
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    
    model = SegmentationModel()
    return model.segment(image)


def extract_clothing_mask(image, pred_seg, clothing_type):
    """
    세그멘테이션 결과에서 특정 의류 부위의 마스크 추출
    
    Args:
        image (np.ndarray): 원본 이미지
        pred_seg (torch.Tensor): 세그멘테이션 결과
        clothing_type (str): 의류 종류 ('top', 'bottom', 'shoes' 등)
    
    Returns:
        np.ndarray: 의류 부위만 추출된 이미지
    """
    # 해당 의류 부위의 인덱스 가져오기
    indices = SEGMENTATION_INDICES.get(clothing_type, [])
    
    # 마스크 생성: 해당 인덱스와 일치하는 부분
    mask = torch.zeros_like(pred_seg, dtype=torch.bool)
    for idx in indices:
        mask = mask | (pred_seg == torch.tensor(idx))
    
    # NumPy 배열로 변환 (0-255 범위)
    mask_np = (mask * 255).numpy().astype(np.uint8)
    
    # 원본 이미지에 마스크 적용
    import cv2
    result = cv2.bitwise_and(
        image.astype(np.uint8),
        image.astype(np.uint8),
        mask=mask_np
    )
    
    return result
