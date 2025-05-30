import torch
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV1

class AestheticScorer:
    def __init__(self, model_id="shunk031/aesthetics-predictor-v1-vit-large-patch14", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AestheticsPredictorV1.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

    def score_image(self, image: Image.Image) -> float:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        score = outputs.logits.item()
        return score