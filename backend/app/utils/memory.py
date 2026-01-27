# backend/app/utils/memory.py
from contextlib import contextmanager

def wipe_bytes(byte_data: bytearray):
    """
    Securely overwrite bytes in memory.
    """
    for i in range(len(byte_data)):
        byte_data[i] = 0

@contextmanager
def ephemeral_bytes(byte_data: bytearray):
    """
    Context manager to ensure resume bytes are wiped after use.
    Usage:
        with ephemeral_bytes(file_bytes) as data:
            # process data
    """
    try:
        yield byte_data
    finally:
        wipe_bytes(byte_data)
        byte_data = None
