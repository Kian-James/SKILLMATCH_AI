# backend/app/core/security.py
import re

def wipe_bytes(byte_data: bytearray):
    """
    Overwrite byte array to ensure resume is erased from memory
    """
    for i in range(len(byte_data)):
        byte_data[i] = 0

def scrub_pii(text: str) -> str:
    """
    Remove personal info from resume text: emails, phones, addresses
    """
    # Remove emails
    text = re.sub(r'\S+@\S+', '[email]', text)
    # Remove phone numbers
    text = re.sub(r'\+?\d[\d -]{7,}\d', '[phone]', text)
    # Remove simple addresses (street numbers + street name)
    text = re.sub(r'\d{1,5}\s\w+(\s\w+){0,3}', '[address]', text)
    return text

class EphemeralContext:
    """
    Context manager to enforce in-memory ephemeral handling
    Usage:
        with EphemeralContext(file_bytes) as data:
            # process data
        # automatically wipes bytes on exit
    """
    def __init__(self, byte_data: bytearray):
        self.byte_data = byte_data

    def __enter__(self):
        return self.byte_data

    def __exit__(self, exc_type, exc_value, traceback):
        wipe_bytes(self.byte_data)
        self.byte_data = None
