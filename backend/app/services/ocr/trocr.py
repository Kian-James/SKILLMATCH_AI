from app.services.ocr.base import BaseOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

class TrOCROCR(BaseOCR):
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    def extract_text(self, file_bytes: bytes) -> str:
        # Convert bytes to image
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        # Preprocess
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        # Generate text
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
