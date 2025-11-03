from wand.image import Image
from wand.color import Color
import numpy as np

from core import ImageProcessor

ALLOWED_MODES = ['CMYK', 'RGBA', 'RGB']

class WandProcessor(ImageProcessor):
    image = None
    icc_profile = None
    color_space = None
    
    def __init__(self, image, icc_profile, color_space):
        self.image = image
        self.icc_profile = icc_profile
        self.color_space = color_space
        
    @classmethod
    def load(cls, path):
        img = Image(filename=path)
        img.alpha_channel = 'activate'
        alpha_np = np.array(img.channel_images['alpha'])
        if alpha_np.max() == 0 and alpha_np.min() == 0:
            full_opaque = Image.from_array(np.full_like(alpha_np, 255, dtype=np.uint8))
            img.composite_channel('alpha', full_opaque, 'copy_alpha')
        
        if img.colorspace.lower() not in ['cmyk', 'srgb']:
            raise ValueError(f"Ảnh input không phải CMYK (colorspace={img.colorspace})!")
        
        return cls(img, img.profiles.get('icc'), img.colorspace)

    def clone(self):
        return WandProcessor(self.image.clone(), self.icc_profile, self.color_space)

    def erase_by_mask(self, mask_path):
        base_clone = self.clone()
        mask = Image(filename=mask_path)
        if 'alpha' not in mask.channel_images:
            mask.alpha_channel = True
        base_alpha = np.array(base_clone.image.channel_images['alpha'], dtype=np.int16)
        mask_alpha = np.array(mask.channel_images['alpha'], dtype=np.int16)

        new_alpha = np.clip(base_alpha - mask_alpha, 0, 255).astype(np.uint8)
        img_alpha = Image.from_array(new_alpha)
        img_alpha.type = 'grayscale'
        img_alpha.colorspace = 'gray'

        self.image.composite_channel('alpha', img_alpha, 'copy_alpha')

        return self
    
    def resize(self, width, height):
        self.image.resize(width, height)
        return self
    
    def rotate(self, angle):
        self.image.rotate(angle)
        return self
    
    def crop(self, left=0, top=0, width=0, height=0, auto=False):
        if auto:
            self.image.trim()
        else:
            self.image.crop(left=left, top=top, width=width, height=height)
        return self
    
    def composite(self, other: 'WandProcessor', x, y):
        self.image.composite(other.image, left=x, top=y)
        return self

    def save(self, path, preview=False):
        self.image.profiles['icc'] = self.icc_profile
        self.image.units = 'pixelsperinch'
        self.image.resolution = (100, 100)
        self.image.compression = 'zip'
        self.image.save(filename=path)
        
    def load_layout(self, path):
        layout = Image(filename=path)
        layout.transform_colorspace(self.color_space)
        layout.alpha_channel = True

        canvas = Image(width=layout.width, height=layout.height, background=Color("transparent"))
        canvas.colorspace = self.color_space

        canvas_proc = WandProcessor(canvas, self.icc_profile, self.color_space)
        layout_proc = WandProcessor(layout, self.icc_profile, self.color_space)
        
        return canvas_proc, layout_proc