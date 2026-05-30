"""
객체 탐지 및 크롭 모듈

DETR(Detection Transformer) 모델을 사용한 객체 탐지 및 이미지 크롭
"""

import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, DetrForObjectDetection
from config import OBJECT_DETECTION_START_THRESHOLD, OBJECT_DETECTION_MIN_THRESHOLD, OBJECT_DETECTION_STEP


class ObjectDetectionModel:
    """DETR를 사용한 객체 탐지 모델"""
    
    def __init__(self):
        """
        사전 학습된 DETR 모델 초기화
        
        Model: facebook/detr-resnet-50
        - 일반적인 객체 탐지 모델
        - 제너릭 객체 탐지에 사용
        """
        self.processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    def detect(self, image, threshold=0.5):
        """
        이미지에서 객체 탐지
        
        Args:
            image (PIL.Image): 입력 이미지
            threshold (float): 신뢰도 임계값 (0~1)
        
        Returns:
            dict: 탐지된 객체 정보
                - 'scores': 신뢰도 점수
                - 'labels': 클래스 레이블
                - 'boxes': 바운딩 박스 좌표
        """
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # 결과를 Pascal VOC 형식(xmin, ymin, xmax, ymax)으로 변환
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )[0]
        
        return results


def crop_clothing_from_mask(image, start_threshold=OBJECT_DETECTION_START_THRESHOLD,
                            min_threshold=OBJECT_DETECTION_MIN_THRESHOLD,
                            step=OBJECT_DETECTION_STEP):
    """
    마스크된 의류 이미지에서 객체를 탐지하고 크롭
    
    Args:
        image (np.ndarray): 마스크가 적용된 이미지
        start_threshold (float): 초기 신뢰도 임계값
        min_threshold (float): 최소 신뢰도 임계값
        step (float): 임계값 감소 스텝 (음수)
    
    Returns:
        PIL.Image: 크롭된 의류 이미지
    """
    image_pil = Image.fromarray(image)
    model = ObjectDetectionModel()
    
    # 임계값을 점진적으로 낮추면서 객체 탐지
    for threshold in np.arange(start_threshold, min_threshold, step):
        try:
            results = model.detect(image_pil, threshold=threshold)
            
            # 탐지된 객체가 있으면 가장 큰 객체 크롭
            if len(results['boxes']) > 0:
                # 바운딩 박스 가져오기 (가장 큰 객체)
                boxes = results['boxes']
                # 면적 계산: (xmax-xmin) * (ymax-ymin)
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                max_idx = torch.argmax(areas)
                
                box = boxes[max_idx].tolist()
                xmin, ymin, xmax, ymax = [round(i, 2) for i in box]
                
                # 이미지 크롭
                cropped = image_pil.crop((xmin, ymin, xmax, ymax))
                return cropped
        
        except Exception as e:
            # 현재 임계값에서 오류 발생 시 낮은 임계값 시도
            continue
    
    # 모든 임계값에서 객체를 찾지 못한 경우 원본 반환
    return image_pil


def crop_clothing_simple(image):
    """
    간단한 크롭 함수 (기본 임계값 사용)
    
    Args:
        image (np.ndarray): 입력 이미지
    
    Returns:
        PIL.Image: 크롭된 이미지
    """
    return crop_clothing_from_mask(image)
