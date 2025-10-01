import os
import json
import time
import numpy as np
from PIL import Image
from icc_utils import load_image_with_icc, save_image_with_icc

# Các hằng số giống trong main.py
OUTPUT_PATH = os.path.join('_output')
MASK_INVERT = False          # Đảo mask hay không
CMYK_ENABLE_ALPHA = True     # Bật kênh alpha tổng hợp cho CMYK (CMYKA) - ép luôn True

try:  # type: ignore
    import tifffile  # noqa: F401
except Exception:
    raise ImportError("tifffile module is required but not installed.")

_HAS_TIFFFILE = True  # ép luôn True theo yêu cầu

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, 'pieces'), exist_ok=True)


def load_config(config_path: str):
    with open(config_path, 'r') as f:
        return json.load(f)


def pillow_process_piece(pil_src: Image.Image, partial: dict, base_path: str) -> Image.Image:
    """Thực thi các step (mask / crop / resize / rotate) trên ảnh CMYK PIL.
    Giữ nguyên dữ liệu kênh CMYK, không chuyển sang RGB.
    """
    cur = pil_src.copy()
    for step in partial['steps']:
        action = step['action']
        data = step['data']
        if action == 'mask':
            mask_path = os.path.join(base_path, data)
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path)
                    mask_img.load()
                except Exception as e:
                    print(f"[WARN] Không đọc được mask {mask_path}: {e}")
                    continue
                if 'A' in mask_img.mode:
                    mask_l = mask_img.split()[-1]
                else:
                    mask_l = mask_img.convert('L')
                if MASK_INVERT:
                    mask_l = Image.eval(mask_l, lambda v: 255 - v)
                c, m, y, k = cur.split()
                # Clear vùng nơi mask_l == 0
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
        else:
            print(f"[WARN] Step không hỗ trợ: {action}")
    return cur


def save_cmyk_or_cmyka(canvas: Image.Image,
                        alpha_canvas: np.ndarray,
                        output_path: str,
                        icc_profile: bytes):
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
    if icc_profile:
        extratags.append((34675, 7, len(icc_profile), icc_profile, False))
    import tifffile  # local import để chắc chắn đã có
    tifffile.imwrite(
        output_path,
        stacked,
        photometric='separated',
        planarconfig='contig',
        extratags=extratags,
        compression='deflate',
        metadata=None
    )
    print('[DONE] Đã lưu 5-kênh:', output_path)
    return


def main(debug: bool = False):
    """Chạy riêng logic cho ảnh input CMYK.
    Nếu ảnh đầu vào không phải CMYK -> thoát và cảnh báo.
    Các đường dẫn giữ nguyên giống "main.py" để dễ test.
    """
    input_image_path = 'mzdzl7z73fd.jpeg'
    config_path = 'L/config.json'
    layout_path_cmyka = 'L/layout_cmyka.tif'
    base_path = 'L'
    output_path = os.path.join(OUTPUT_PATH, 'final_output.tif')

    # 1. Load input + ICC
    print('[INFO] Load input with ICC ...')
    input_icc = load_image_with_icc(input_image_path, as_numpy=True)
    print('[INFO] Mode:', input_icc.mode, 'Size:', input_icc.size)
    if input_icc.icc_profile:
        print('[INFO] ICC length:', len(input_icc.icc_profile))

    if input_icc.mode != 'CMYK':
        print('[EXIT] Ảnh không phải CMYK -> file này chỉ dành cho xử lý CMYK.')
        return

    # 2. Read config
    config = load_config(config_path)
    partials = config['partials']

    # 3. Load base layout CMYK(A)
    if os.path.exists(layout_path_cmyka):
        print(f'[INFO] Load layout CMYKA base: {layout_path_cmyka}')
        base_cmyk_img = None
        alpha_canvas = None
        try:
            import tifffile
            arr = tifffile.imread(layout_path_cmyka)
            if arr.ndim == 3 and arr.shape[2] in (4, 5):
                if arr.shape[2] == 5 and CMYK_ENABLE_ALPHA:
                    c_arr, m_arr, y_arr, k_arr, a_arr = [arr[:, :, i].astype('uint8') for i in range(5)]
                    base_cmyk_img = Image.merge('CMYK', (
                        Image.fromarray(c_arr, 'L'),
                        Image.fromarray(m_arr, 'L'),
                        Image.fromarray(y_arr, 'L'),
                        Image.fromarray(k_arr, 'L')
                    ))
                    alpha_canvas = a_arr.copy()
                else:
                    raise Exception("Layout CMYKA không có 5 kênh hoặc CMYK_ENABLE_ALPHA tắt.")
                    # c_arr, m_arr, y_arr, k_arr = [arr[:, :, i].astype('uint8') for i in range(4)]
                    # base_cmyk_img = Image.merge('CMYK', (
                    #     Image.fromarray(c_arr, 'L'),
                    #     Image.fromarray(m_arr, 'L'),
                    #     Image.fromarray(y_arr, 'L'),
                    #     Image.fromarray(k_arr, 'L')
                    # ))
            else:
                print('[WARN] layout_cmyka shape lạ, fallback Pillow open.')
        except Exception as e:
            print('[WARN] tifffile read fail, fallback Pillow open:', e)
        if base_cmyk_img is None:
            try:
                tmp = Image.open(layout_path_cmyka)
                tmp.load()
                if tmp.mode != 'CMYK':
                    tmp = tmp.convert('CMYK')
                base_cmyk_img = tmp
            except Exception as e:
                print('[WARN] Không mở được layout CMYKA, dùng canvas trống:', e)
                base_cmyk_img = None
        if base_cmyk_img is not None:
            layout_w, layout_h = base_cmyk_img.size
        else:
            layout_w, layout_h = input_icc.size
    else:
        print('[WARN] layout_cmyka.tif không tồn tại, dùng canvas trống.')
        base_cmyk_img = None
        alpha_canvas = None
        layout_w, layout_h = input_icc.size

    print(f'[INFO] Canvas size: {(layout_w, layout_h)}')

    # 4. Canvas khởi tạo
    if base_cmyk_img is not None:
        canvas = base_cmyk_img.copy()
        if CMYK_ENABLE_ALPHA:
            if 'alpha_canvas' not in locals() or alpha_canvas is None:
                alpha_canvas = np.zeros((layout_h, layout_w), dtype=np.uint8)
    else:
        canvas = Image.new('CMYK', (layout_w, layout_h), (0, 0, 0, 0))
        alpha_canvas = np.zeros((layout_h, layout_w), dtype=np.uint8) if CMYK_ENABLE_ALPHA else None

    # 5. Process pieces
    pieces = []
    for partial in partials:
        piece_img = pillow_process_piece(input_icc.pil_image, partial, base_path)
        pieces.append((piece_img, partial['location']))

    # 6. Paste pieces lên canvas
    for i, (pimg, loc) in enumerate(pieces):
        x = loc['left']; y = loc['top']
        if x >= canvas.width or y >= canvas.height:
            print(f'[WARN] Piece {i} ngoài canvas, bỏ qua.')
            continue
        c, m, yk, k = pimg.split()
        arr_sum = (np.array(c, dtype=np.uint16) +
                   np.array(m, dtype=np.uint16) +
                   np.array(yk, dtype=np.uint16) +
                   np.array(k, dtype=np.uint16))
        mask_bin_np = (arr_sum > 0).astype('uint8') * 255
        mask_bin = Image.fromarray(mask_bin_np, mode='L')
        canvas.paste(pimg, (x, y), mask_bin)
        if alpha_canvas is not None:
            h_p, w_p = pimg.size[1], pimg.size[0]
            end_x = min(alpha_canvas.shape[1], x + w_p)
            end_y = min(alpha_canvas.shape[0], y + h_p)
            sub_h = end_y - y; sub_w = end_x - x
            if sub_h > 0 and sub_w > 0:
                alpha_slice = mask_bin_np[:sub_h, :sub_w]
                alpha_canvas[y:end_y, x:end_x] = np.maximum(alpha_canvas[y:end_y, x:end_x], alpha_slice[:sub_h, :sub_w])

    # 7. Save
    save_cmyk_or_cmyka(canvas, alpha_canvas, output_path, input_icc.icc_profile)


def run():
    start = time.time()
    main(debug=False)
    end = time.time()
    print(f"Total processing time (CMYK): {end - start:.2f} seconds")


if __name__ == '__main__':
    run()
