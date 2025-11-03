import cv2
import numpy as np
from PIL import Image
import os
from wand.image import Image as WandImage
from wand.color import Color

from core import ImageProcessor

ALLOWED_MODES = ['CMYK', 'RGBA', 'RGB']

class OpenCVProcessor(ImageProcessor):
    image = None
    icc_profile = None
    mode = None
    
    def __init__(self, image, icc_profile, mode):
        self.image = image
        self.icc_profile = icc_profile
        self.mode = mode

    @classmethod
    def load(cls, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        im = Image.open(path)
        icc_profile = im.info.get('icc_profile')
        mode = im.mode.upper()
        if ALLOWED_MODES and mode not in ALLOWED_MODES:
            raise ValueError(f"Unsupported image mode: {mode}. Allowed modes: {ALLOWED_MODES}")

        np_img = np.array(im)
        if mode == 'CMYK':
            alpha = np.full((np_img.shape[0], np_img.shape[1], 1), 255, dtype=np.uint8)
            np_img = np.concatenate([np_img, alpha], axis=2)
            mode = 'CMYKA'
        elif mode == 'RGB':
            alpha = np.full((np_img.shape[0], np_img.shape[1], 1), 255, dtype=np.uint8)
            np_img = np.concatenate([np_img, alpha], axis=2)
            mode = 'RGBA'
        elif mode == 'RGBA':
            pass  # already has alpha channel

        return cls(np_img, icc_profile, mode)

    def clone(self):
        return OpenCVProcessor(self.image.copy(), self.icc_profile, self.mode)

    def erase_by_mask(self, mask_path):
        mask_alpha = np.array(Image.open(mask_path).convert("RGBA"))[:, :, 3]

        alpha = self.clone().image[:, :, -1].astype(np.int16)
        mask_alpha = mask_alpha.astype(np.int16)

        new_alpha = np.clip(alpha - mask_alpha, 0, 255).astype(np.uint8)

        self.image[:, :, -1] = new_alpha

        return self
    
    def resize(self, width, height):
        channels, alpha = self.image[:, :, :-1], self.image[:, :, -1]
        channels_resized = cv2.resize(channels, (width, height), interpolation=cv2.INTER_LANCZOS4)
        alpha_resized = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_NEAREST)
        self.image = np.dstack((channels_resized, alpha_resized))

        return self

    def rotate(self, angle):
        """
        Xoay ảnh RGBA hoặc CMYKA theo kiểu Wand:
        - Không crop
        - Giữ alpha trong suốt
        - Cân giữa canvas mới
        """
        image = self.image
        (h, w) = image.shape[:2]
        num_channels = image.shape[2]

        # Tâm xoay
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)

        # Tính kích thước canvas mới
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # Dịch chuyển để giữ tâm ở giữa canvas mới
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        # Tách kênh và xoay từng phần
        rotated_channels = []
        for i in range(num_channels):
            ch = image[..., i]
            rotated = cv2.warpAffine(
                ch,
                M,
                (new_w, new_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,  # trong suốt ở alpha
            )
            rotated_channels.append(rotated)

        # Ghép lại
        rotated = cv2.merge(rotated_channels)
        self.image = rotated

    def crop(self, left=0, top=0, width=0, height=0, auto=False):
        x1, y1, x2, y2 = left, top, left + width, top + height
        if auto:
            alpha = self.image[:, :, -1]
            ys, xs = np.where(alpha > 0)
            top, bottom = ys.min(), ys.max()
            left, right = xs.min(), xs.max()
            self.image = self.image[top:bottom + 1, left:right + 1]
        else:
            self.image = self.image[y1:y2, x1:x2]

        return self

    def composite(self, other: 'OpenCVProcessor', x, y):
        h, w, _ = other.image.shape

        target_region = self.image[y:y+h, x:x+w, :]

        alpha_mask = other.image[:, :, -1] > 0
        alpha_mask = alpha_mask[:, :, None]

        target_region = np.where(alpha_mask, other.image, target_region)

        self.image[y:y+h, x:x+w, :] = target_region

        return self

    def save(self, path, preview=False):
        channels, alpha = self.image[:, :, :-1], self.image[:, :, -1]
        h, w, _ = self.image.shape

        channels_bytes = channels.tobytes()
        alpha_bytes = alpha.tobytes()
        if self.mode == 'CMYKA':
            channel_map = 'cmyk'
            colorspace = 'cmyk'
        else:
            channel_map = 'rgb'
            colorspace = 'rgb'

        with WandImage(width=w, height=h, depth=8, background=Color('transparent')) as img:
            img.format = 'tiff'
            img.colorspace = colorspace
            img.alpha_channel = 'activate'
            img.units = 'pixelsperinch'
            img.resolution = (100, 100)

            # write CMYK channels
            img.import_pixels(channel_map=channel_map, data=channels_bytes)

            # write alpha
            img.import_pixels(channel_map='a', data=alpha_bytes)

            img.profiles['icc'] = self.icc_profile

            img.compression = 'zip'
            # save file
            img.save(filename=path)
         
    def load_layout(self, path):
        input_image = Image.open(path).convert("RGBA")
        if self.mode == 'CMYKA':
            cmyk = input_image.convert("CMYK")
            np_cmyk = np.array(cmyk)  # (H, W, 4)
            alpha = np.array(input_image)[:, :, 3][:, :, None]  # (H,W,1)
            np_layout = np.dstack([np_cmyk, alpha]).astype(np.uint8)  # (H,W,5)
            h, w = np_layout.shape[:2]
            np_canvas = np.zeros((h, w, 5), dtype=np.uint8)
        else:
            np_layout = np.array(input_image)  # (H, W, 4)
            h, w = np_layout.shape[:2]
            np_canvas = np.zeros((h, w, 4), dtype=np.uint8)

        canvas_proc = OpenCVProcessor(np_canvas, self.icc_profile, self.mode)
        layout_proc = OpenCVProcessor(np_layout, self.icc_profile, self.mode)
        return canvas_proc, layout_proc