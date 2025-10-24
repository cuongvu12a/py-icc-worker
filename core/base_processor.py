from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class ImageProcessor(ABC):
    @abstractmethod
    def clone(self): pass
    @abstractmethod
    def erase_by_mask(self, mask_path): pass
    @abstractmethod
    def resize(self, width, height): pass
    @abstractmethod
    def rotate(self, angle): pass
    @abstractmethod
    def crop(self, left, top, width, height, auto): pass
    @abstractmethod
    def composite(self, other, x, y): pass
    @abstractmethod
    def save(self, path, preview): pass
    @abstractmethod
    def load_layout(self, path): pass
