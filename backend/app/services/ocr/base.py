from abc import ABC, abstractmethod

class BaseOCR(ABC):
    @abstractmethod
    def extract_text(self, file_bytes: bytes) -> str:
        """
        Extract text from a PDF or image in-memory.
        :param file_bytes: Raw bytes of the uploaded file
        :return: Extracted text as string
        """
        pass
