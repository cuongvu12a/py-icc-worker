import os
import json
import time
import cv2
import numpy as np
from icc_utils import load_image_with_icc, save_image_with_icc
from PIL import Image

OUTPUT_PATH = os.path.join('_output')
MASK_INVERT = False  # Đặt True = đảo mask (giống logic cũ). Đặt False = dùng mask nguyên gốc.
CMYK_ENABLE_ALPHA = True  # Thêm kênh alpha ảo cho CMYK (CMYKA). Lưu TIFF 5 kênh nếu có tifffile.

try:
    import tifffile  # type: ignore
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'pieces'), exist_ok=True)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)


def apply_mask(image, mask_path):
    if not os.path.exists(mask_path):
        print(f"[WARN] mask missing: {mask_path}")
        return image
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"[WARN] cannot read mask: {mask_path}")
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif len(image.shape) == 2:
        tmp = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        tmp[:, :, :3] = np.stack([image]*3, axis=2)
        tmp[:, :, 3] = 255
        image = tmp

    if len(mask.shape) == 3 and mask.shape[2] == 4:
        mask_alpha = mask[:, :, 3]
    elif len(mask.shape) == 3 and mask.shape[2] == 3:
        mask_alpha = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    elif len(mask.shape) == 2:
        mask_alpha = mask
    else:
        return image

    # Nếu muốn đảo chiều (vùng trắng thành trong suốt) thì invert tại đây
    if MASK_INVERT:
        mask_alpha = 255 - mask_alpha

    if mask_alpha.shape[:2] != image.shape[:2]:
        mask_alpha = cv2.resize(mask_alpha, (image.shape[1], image.shape[0]))

    alpha_norm = mask_alpha.astype(np.float32) / 255.0
    for c in range(3):
        image[:, :, c] = (image[:, :, c].astype(np.float32) * alpha_norm).astype(np.uint8)
    image[:, :, 3] = (image[:, :, 3].astype(np.float32) * alpha_norm).astype(np.uint8)
    return image


def crop_image(image, top, left, width, height):
    h, w = image.shape[:2]
    top = max(0, min(top, h))
    left = max(0, min(left, w))
    bottom = min(h, top + height)
    right = min(w, left + width)
    return image[top:bottom, left:right]


def resize_image(image, width, height):
    return cv2.resize(image, (width, height))


def rotate_image(image, angle):
    if angle == 0:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    if len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return cv2.warpAffine(image, M, (new_w, new_h))


def process_piece(input_image, partial, base_path, debug=False):
    pid = partial['id']
    steps = partial['steps']
    cur = input_image.copy()
    print(f"Process piece: {pid}")
    for idx, step in enumerate(steps):
        action = step['action']
        data = step['data']
        if action == 'mask':
            cur = apply_mask(cur, os.path.join(base_path, data))
        elif action == 'crop':
            cur = crop_image(cur, data['top'], data['left'], data['width'], data['height'])
        elif action == 'resize':
            cur = resize_image(cur, data['width'], data['height'])
        elif action == 'rotate':
            cur = rotate_image(cur, data)
        if debug:
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'pieces', f"{pid}_step_{idx+1}_{action}.png"), cur)
    if debug:
        cv2.imwrite(os.path.join(OUTPUT_PATH, 'pieces', f"{pid}_final.png"), cur)
    return cur


def place_piece(canvas, piece, location):
    top = location['top']; left = location['left']
    ph, pw = piece.shape[:2]
    ch, cw = canvas.shape[:2]
    end_top = min(ch, top + ph)
    end_left = min(cw, left + pw)
    h = end_top - top; w = end_left - left
    if h <= 0 or w <= 0:
        print('[WARN] piece outside canvas')
        return canvas
    roi = canvas[top:end_top, left:end_left]
    sub = piece[:h, :w]
    if sub.shape[2] == 4:
        if roi.shape[2] == 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
            roi = canvas[top:end_top, left:end_left]
        alpha = sub[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (roi[:, :, c] * (1 - alpha) + sub[:, :, c] * alpha)
        roi[:, :, 3] = np.maximum(roi[:, :, 3], sub[:, :, 3])
    else:
        if roi.shape[2] == 4 and sub.shape[2] == 3:
            tmp = cv2.cvtColor(sub, cv2.COLOR_BGR2BGRA)
            tmp[:, :, 3] = 255
            sub = tmp
        roi[:,:] = sub
    canvas[top:end_top, left:end_left] = roi
    return canvas


def main(debug=False):
    input_image_path = 'mzdzl7z73fd.jpeg'
    config_path = 'L/config.json'
    layout_path_rgb = 'L/layout.png'
    layout_path_cmyka = 'L/layout_cmyka.tif'
    base_path = 'L'
    output_path = os.path.join(OUTPUT_PATH, 'final_output.tif')

    # 1. Load input with ICC metadata
    print('[INFO] Load input with ICC ...')
    input_icc = load_image_with_icc(input_image_path, as_numpy=True)
    if input_icc.icc_profile:
        print('[INFO] ICC found, length:', len(input_icc.icc_profile))
    print('[INFO] Mode:', input_icc.mode, 'Size:', input_icc.size)

    # 2. Read config
    config = load_config(config_path)
    partials = config['partials']

    # 3. Load layout content (not just size) depending on colorspace
    if input_icc.mode == 'CMYK':
        if os.path.exists(layout_path_cmyka):
            print(f'[INFO] Loading CMYK layout base: {layout_path_cmyka}')
            base_cmyk_img = None
            alpha_canvas = None
            if _HAS_TIFFFILE:
                try:
                    arr = tifffile.imread(layout_path_cmyka)
                    if arr.ndim == 3 and arr.shape[2] in (4,5):
                        if arr.shape[2] == 5 and CMYK_ENABLE_ALPHA:
                            c_arr, m_arr, y_arr, k_arr, a_arr = [arr[:,:,i].astype('uint8') for i in range(5)]
                            base_cmyk_img = Image.merge('CMYK', (
                                Image.fromarray(c_arr,'L'),
                                Image.fromarray(m_arr,'L'),
                                Image.fromarray(y_arr,'L'),
                                Image.fromarray(k_arr,'L')
                            ))
                            alpha_canvas = a_arr.copy()
                        else:
                            # Assume 4 channels CMYK
                            c_arr, m_arr, y_arr, k_arr = [arr[:,:,i].astype('uint8') for i in range(4)]
                            base_cmyk_img = Image.merge('CMYK', (
                                Image.fromarray(c_arr,'L'),
                                Image.fromarray(m_arr,'L'),
                                Image.fromarray(y_arr,'L'),
                                Image.fromarray(k_arr,'L')
                            ))
                    else:
                        print('[WARN] Unexpected layout_cmyka shape, fallback Pillow open.')
                except Exception as e:
                    print('[WARN] tifffile read failed, fallback Pillow open:', e)
            if base_cmyk_img is None:
                try:
                    tmp = Image.open(layout_path_cmyka)
                    tmp.load()
                    if tmp.mode != 'CMYK':
                        tmp = tmp.convert('CMYK')
                    base_cmyk_img = tmp
                except Exception as e:
                    print('[WARN] Cannot open CMYKA base, will fallback blank:', e)
            # Derive layout size
            if base_cmyk_img is not None:
                layout_w, layout_h = base_cmyk_img.size
            else:
                layout_w, layout_h = input_icc.size
        else:
            print(f'[WARN] layout_cmyka.tif missing, fallback blank canvas.')
            base_cmyk_img = None
            alpha_canvas = None
            layout_w, layout_h = input_icc.size
        print(f'[INFO] Canvas size: {(layout_w, layout_h)}')
    else:
        if os.path.exists(layout_path_rgb):
            layout_img = cv2.imread(layout_path_rgb, cv2.IMREAD_UNCHANGED)
            if layout_img is None:
                print('[WARN] Cannot read layout.png, fallback input size')
                layout_h, layout_w = input_icc.size[1], input_icc.size[0]
                layout_img = None
            else:
                layout_h, layout_w = layout_img.shape[:2]
        else:
            print('[WARN] layout.png missing, fallback blank')
            layout_img = None
            layout_w, layout_h = input_icc.size
        print(f'[INFO] Canvas size: {(layout_w, layout_h)}')

    # Nếu người dùng muốn giữ nguyên CMYK tuyệt đối: dùng đường xử lý Pillow thuần cho CMYK
    if input_icc.mode == 'CMYK':
        print('[INFO] CMYK preserve path: xử lý không convert sang RGB.')
        # Sử dụng layout_cmyka.tif nếu có, nếu không tạo trống
        if 'base_cmyk_img' in locals() and base_cmyk_img is not None:
            canvas = base_cmyk_img.copy()
            if CMYK_ENABLE_ALPHA:
                if 'alpha_canvas' not in locals() or alpha_canvas is None:
                    alpha_canvas = np.zeros((layout_h, layout_w), dtype=np.uint8)
        else:
            canvas = Image.new('CMYK', (layout_w, layout_h), (0,0,0,0))
            alpha_canvas = np.zeros((layout_h, layout_w), dtype=np.uint8) if CMYK_ENABLE_ALPHA else None

        def pillow_process_piece(pil_src: Image.Image, partial):
            cur = pil_src.copy()
            for step in partial['steps']:
                action = step['action']
                data = step['data']
                if action == 'mask':
                    mask_path = os.path.join(base_path, data)
                    if os.path.exists(mask_path):
                        mask_img = Image.open(mask_path)
                        mask_img.load()
                        if 'A' in mask_img.mode:
                            mask_l = mask_img.split()[-1]
                        else:
                            mask_l = mask_img.convert('L')
                        if MASK_INVERT:
                            mask_l = Image.eval(mask_l, lambda v: 255 - v)
                        c, m, y, k = cur.split()
                        # inv_mask = vùng cần clear = 0 (giữ) hay 255? Ta clear nơi mask_l == 0
                        inv_mask = Image.eval(mask_l, lambda v: 255 if v == 0 else 0)
                        zero = Image.new('L', cur.size, 0)
                        c = Image.composite(c, zero, inv_mask)
                        m = Image.composite(m, zero, inv_mask)
                        y = Image.composite(y, zero, inv_mask)
                        k = Image.composite(k, zero, inv_mask)
                        cur = Image.merge('CMYK', (c, m, y, k))
                elif action == 'crop':
                    box = (data['left'], data['top'], data['left'] + data['width'], data['top'] + data['height'])
                    cur = cur.crop(box)
                elif action == 'resize':
                    cur = cur.resize((data['width'], data['height']), Image.Resampling.LANCZOS)
                elif action == 'rotate':
                    cur = cur.rotate(data, expand=True)
            return cur

        pieces = []
        for partial in partials:
            piece_img = pillow_process_piece(input_icc.pil_image, partial)
            pieces.append((piece_img, partial['location']))

        for i, (pimg, loc) in enumerate(pieces):
            x = loc['left']; y = loc['top']
            if x >= canvas.width or y >= canvas.height:
                continue
            c, m, yk, k = pimg.split()
            arr_sum = np.array(c, dtype=np.uint16) + np.array(m, dtype=np.uint16) + np.array(yk, dtype=np.uint16) + np.array(k, dtype=np.uint16)
            mask_bin_np = (arr_sum > 0).astype('uint8') * 255
            mask_bin = Image.fromarray(mask_bin_np, mode='L')
            canvas.paste(pimg, (x, y), mask_bin)
            if alpha_canvas is not None:
                # Ghi alpha 255 nơi có mực (mask_bin=255)
                h_p, w_p = pimg.size[1], pimg.size[0]
                # region area in alpha canvas
                end_x = min(alpha_canvas.shape[1], x + w_p)
                end_y = min(alpha_canvas.shape[0], y + h_p)
                sub_h = end_y - y; sub_w = end_x - x
                if sub_h > 0 and sub_w > 0:
                    alpha_slice = mask_bin_np[:sub_h, :sub_w]
                    alpha_canvas[y:end_y, x:end_x] = np.maximum(alpha_canvas[y:end_y, x:end_x], alpha_slice[:sub_h, :sub_w])

        # Lưu output
        if CMYK_ENABLE_ALPHA and alpha_canvas is not None and _HAS_TIFFFILE:
            print('[INFO] Saving CMYK+Alpha (5 kênh) TIFF với tifffile.')
            try:
                c, m, yk, k = canvas.split()
                c_arr = np.array(c, dtype=np.uint8)
                m_arr = np.array(m, dtype=np.uint8)
                y_arr = np.array(yk, dtype=np.uint8)
                k_arr = np.array(k, dtype=np.uint8)
                a_arr = alpha_canvas.astype(np.uint8)
                stacked = np.dstack([c_arr, m_arr, y_arr, k_arr, a_arr])  # H W 5
                extratags = [
                    (338, 1, 1, b'\x02', False)  # ExtraSamples: Associated Alpha
                ]
                # Nhúng ICC profile bằng tag 34675 (ICC Profile) kiểu UNDEFINED (7)
                if input_icc.icc_profile:
                    icc_bytes = input_icc.icc_profile
                    extratags.append((34675, 7, len(icc_bytes), icc_bytes, False))
                tifffile.imwrite(
                    output_path,
                    stacked,
                    photometric='separated',
                    planarconfig='contig',
                    extratags=extratags,
                    compression='deflate',
                    metadata=None  # tránh serialize bytes -> JSON
                )
                print('[DONE] Saved CMYK+Alpha 5-channel:', output_path)
            except Exception as e:
                print('[WARN] Ghi 5-kênh thất bại, fallback CMYK + alpha riêng:', e)
                base, ext = os.path.splitext(output_path)
                save_image_with_icc(
                    canvas,
                    base + '_cmyk.tif',
                    icc_profile=input_icc.icc_profile,
                    mode='CMYK',
                    prefer_format='TIFF'
                )
                Image.fromarray(alpha_canvas, mode='L').save(base + '_alpha.png')
                print('[DONE] Saved fallback:', base + '_cmyk.tif', 'and', base + '_alpha.png')
        else:
            print('[INFO] Saving CMYK only (alpha riêng nếu bật).')
            save_image_with_icc(
                canvas,
                output_path,
                icc_profile=input_icc.icc_profile,
                mode='CMYK',
                prefer_format='TIFF'
            )
            if CMYK_ENABLE_ALPHA and alpha_canvas is not None:
                base, ext = os.path.splitext(output_path)
                alpha_path = base + '_alpha.png'
                Image.fromarray(alpha_canvas, mode='L').save(alpha_path)
                print('[INFO] Alpha mask saved:', alpha_path)
            print('[DONE] Saved at', output_path)
        return

    # ---- NHÁNH MẶC ĐỊNH (RGB/RGBA) như cũ ----
    # 4. Create blank canvas according to input mode (preserve channels). Use BGRA or BGR for OpenCV processing.
    if input_icc.mode in ('RGBA',):
        channels = 4
    elif input_icc.mode == 'RGB':
        channels = 3
    else:
        channels = 4  # fallback
    # Use layout image as base if available; else blank
    if 'layout_img' in locals() and layout_img is not None:
        # Ensure channel count matches
        if channels == 4 and (layout_img.ndim == 3 and layout_img.shape[2] == 3):
            blank = cv2.cvtColor(layout_img, cv2.COLOR_BGR2BGRA)
            blank[:, :, 3] = 255  # Opaque base
        else:
            blank = layout_img.copy()
    else:
        blank = np.zeros((layout_h, layout_w, channels), dtype=np.uint8)
        if channels == 4:
            blank[:, :, 3] = 0

    # 5. Prepare input image for OpenCV
    src_arr = input_icc.np_array
    if input_icc.mode == 'RGBA':
        bgr_input = cv2.cvtColor(src_arr, cv2.COLOR_RGBA2BGRA)
    elif input_icc.mode == 'RGB':
        bgr_input = cv2.cvtColor(src_arr, cv2.COLOR_RGB2BGR)
    else:
        # Convert any other to RGB then to BGR(A)
        pil_rgb = input_icc.pil_image.convert('RGBA') if input_icc.mode not in ('RGB', 'RGBA') else input_icc.pil_image
        arr = np.array(pil_rgb)
        if pil_rgb.mode == 'RGBA':
            bgr_input = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        else:
            bgr_input = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    processed = []
    for partial in partials:
        piece = process_piece(bgr_input, partial, base_path, debug)
        processed.append((piece, partial['location']))

    canvas = blank
    for i, (piece, loc) in enumerate(processed):
        print(f"Place piece {i+1}: top={loc['top']} left={loc['left']}")
        canvas = place_piece(canvas, piece, loc)

    if canvas.shape[2] == 4:
        rgba = cv2.cvtColor(canvas, cv2.COLOR_BGRA2RGBA)
        save_arr = rgba
        target_mode = 'RGBA'
    else:
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        save_arr = rgb
        target_mode = 'RGB'

    print('[INFO] Saving final output with ICC...')
    save_image_with_icc(
        save_arr,
        output_path,
        icc_profile=input_icc.icc_profile,
        mode=target_mode,
        prefer_format='TIFF'
    )
    print('[DONE] Saved at', output_path)

if __name__ == "__main__":
    start = time.time()
    main(debug=False)
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")
