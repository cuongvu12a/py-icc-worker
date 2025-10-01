from typing import Optional, Tuple, Union
from PIL import Image
import numpy as np
import os
import cv2
import json
import time

try:  # type: ignore
    import tifffile  # noqa: F401
except Exception:
    raise ImportError("tifffile module is required but not installed.")

OUTPUT_PATH = os.path.join('_output')
OUTPUT_PIECES_PATH = os.path.join(OUTPUT_PATH, 'pieces')

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(OUTPUT_PIECES_PATH, exist_ok=True)

def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)

def read_cmyka_image(file_path: str) -> Image.Image:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    im = Image.open(file_path)
    icc_profile = im.info.get('icc_profile')
    mode = im.mode
    if mode != 'CMYK':
        raise ValueError(f"Chế độ hình ảnh không phải CMYK: {mode}")
    
    np_img = np.array(im)
    alpha = np.full((np_img.shape[0], np_img.shape[1], 1), 255, dtype=np.uint8)

    return np.concatenate([np_img, alpha], axis=2), icc_profile

def apply_mask(np_img: np.ndarray, mask_path: str) -> np.ndarray:
    # mở mask RGBA và lấy kênh alpha
    mask = Image.open(mask_path).convert("RGBA")
    alpha = np.array(mask)[:, :, 3]  # (H, W)
    
    # core_mask = True ở chỗ alpha = 0
    core_mask = (alpha == 0)  # (H, W)
    
    # broadcast sang (H, W, 1) để áp lên 5 kênh
    core_mask = core_mask[:, :, None]  # (H, W, 1)
    
    # tạo output giữ nguyên pixel tại core_mask, còn lại set 0
    result = np.where(core_mask, np_img, 0)
    
    return result

def crop_img(np_img: np.ndarray, crop_box: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    return np_img[y1:y2, x1:x2]

def resize_img(np_img: np.ndarray, new_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(np_img, new_size, interpolation=cv2.INTER_LANCZOS4)

def rotate_img(np_img: np.ndarray, angle: float) -> np.ndarray:
    (h, w) = np_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(np_img, M, (w, h))

def save_tiff_cmyka(np_img: np.ndarray, file_path: str, icc_profile: Optional[bytes] = None, preview: bool = False):
    stacked = np_img.astype(np.uint8)
    c, m, y, k, a = [stacked[:, :, i] for i in range(5)]
    mask = (a == 0)
    c[mask] = 0
    m[mask] = 0
    y[mask] = 0
    k[mask] = 0
    stacked = np.dstack([c, m, y, k, a]).astype(np.uint8)
        
    extratags = [
        (338, 1, 1, b'\x02', False)  # ExtraSamples: Associated Alpha
    ]
    if icc_profile:
        extratags.append((34675, 7, len(icc_profile), icc_profile, False))
    tifffile.imwrite(
        file_path,
        stacked,
        photometric='separated',
        planarconfig='contig',
        extratags=extratags,
        compression='lzw' if preview else 'deflate',
        metadata=None
    )
                
def main(debug: bool = False):
    file_path = 'mzdzl7z73fd.jpeg'
    config_path = 'L/config.json'
    layout_path_cmyka = 'L/layout_cmyka.tif'
    
    try:
        origin_array, icc_profile = read_cmyka_image(file_path)
        layout_array = tifffile.imread(layout_path_cmyka)
        print("CMYK Array Shape:", origin_array.shape)
        if icc_profile:
            print("ICC Profile found.")
        else:
            print("No ICC Profile found.")
            
        config = load_config(config_path)
        partials = config['partials']
        
        pieces = []
        for partial in partials:
            id = partial['id']
            steps = partial['steps']
            location= partial['location']
            cmyk_array = origin_array.copy()
            
            for step in steps:
                action = step['action']
                data = step['data']
                if action == 'mask':
                    mask_path = os.path.join('L', data)
                    cmyk_array = apply_mask(cmyk_array, mask_path)
                    print(f"Applied mask: {mask_path}, New shape: {cmyk_array.shape}")
                elif action == 'crop':
                    top = data['top']
                    left = data['left']
                    width = data['width']
                    height = data['height']
                    box = (left, top, left + width, top + height)
                    cmyk_array = crop_img(cmyk_array, box)
                    print(f"Cropped to box: {box}, New shape: {cmyk_array.shape}")
                elif action == 'resize':
                    width = data['width']
                    height = data['height']
                    size = (width, height)
                    cmyk_array = resize_img(cmyk_array, size)
                    print(f"Resized to: {size}, New shape: {cmyk_array.shape}")
                elif action == 'rotate':
                    angle = step['data']
                    cmyk_array = rotate_img(cmyk_array, angle)
                    print(f"Rotated by: {angle} degrees, New shape: {cmyk_array.shape}")
                else:
                    print(f"Unknown action: {action}")

                if debug:
                    debug_path = os.path.join(OUTPUT_PIECES_PATH, f"{id}_{action}.tif")
                    save_tiff_cmyka(cmyk_array, debug_path, icc_profile)
                    print(f"Saved debug image: {debug_path}")

            pieces.append((cmyk_array, location['top'], location['left']))
        
        result = layout_array.copy()
        for piece in pieces:
            piece_array, y0, x0 = piece
            h, w, _ = piece_array.shape
            
            target_region = result[y0:y0+h, x0:x0+w, :]
            
            alpha_mask = piece_array[:, :, 4] > 0
            alpha_mask = alpha_mask[:, :, None]
            
            target_region = np.where(alpha_mask, piece_array, target_region)
            
            result[y0:y0+h, x0:x0+w, :] = target_region

        save_tiff_cmyka(result, os.path.join(OUTPUT_PATH, "final_result.tif"), icc_profile, preview=False)

    except Exception as e:
        # print(f"Error: {e}")
        raise e
        
if __name__ == "__main__":
    start = time.time()
    main(debug=False)
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")