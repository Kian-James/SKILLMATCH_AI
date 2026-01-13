from google.cloud import vision
from app.utils.text_cleaner import clean_text

client = vision.ImageAnnotatorClient()

def extract_text(file_bytes: bytes) -> str:
    image = vision.Image(content=file_bytes)
    response = client.text_detection(image=image)

    if not response.text_annotations:
        return ""

    return clean_text(response.text_annotations[0].description)
