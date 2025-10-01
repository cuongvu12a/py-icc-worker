"""compare_image.py

Script so sánh hai ảnh dựa trên:
 1. ICC profile (bytes so sánh md5 + length)
 2. Colorspace / mode (Pillow mode)
 3. Kích thước (width, height)
 4. Dữ liệu điểm ảnh: tất cả pixel & kênh phải giống (hoặc trong ngưỡng sai số cho phép)

Hỗ trợ trường hợp:
 - Ảnh ở CMYK và ảnh còn lại là bản convert RGB (mở rộng kênh) -> dùng chuyển đổi có ICC để so sánh trong không gian RGB.
 - Cả hai đều CMYK -> so sánh trực tiếp từng kênh.
 - Palette (P) -> chuyển RGBA trước khi so sánh.

CLI:
    python3 compare_image.py img1.tif img2.tif \
        --allow-convert-cmyk --tolerance 0 --max-mismatches 20 --json

Các tham số:
    --allow-convert-cmyk: Cho phép convert CMYK -> RGB khi mode khác nhau hoặc khi so sánh mong muốn ở RGB.
    --tolerance: Sai số tối đa mỗi kênh (0 = phải tuyệt đối giống). Ví dụ 1 hoặc 2 để bỏ qua rounding.
    --max-mismatches: Giới hạn số pixel mismatch in ra (để không spam output).
    --json: In báo cáo JSON ngắn gọn.

Exit code:
    0 nếu pass (tất cả tiêu chí) else 1.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image, ImageCms
import numpy as np
import io


@dataclass
class ImageMeta:
    path: str
    mode: str
    size: Tuple[int, int]
    icc_profile: Optional[bytes]
    icc_md5: Optional[str]
    icc_len: int
    pil_image: Image.Image


def load_image(path: str) -> ImageMeta:
    im = Image.open(path)
    im.load()
    icc = im.info.get('icc_profile')
    icc_md5 = hashlib.md5(icc).hexdigest() if icc else None
    icc_len = len(icc) if icc else 0
    return ImageMeta(
        path=path,
        mode=im.mode,
        size=im.size,
        icc_profile=icc,
        icc_md5=icc_md5,
        icc_len=icc_len,
        pil_image=im
    )


def _ensure_rgb_with_profile(im: Image.Image, embedded_icc: Optional[bytes]) -> Image.Image:
    """Convert image to RGB using ICC aware transform when possible.

    If embedded_icc exists -> profileToProfile to sRGB. Else fallback im.convert('RGB').
    For CMYK, this avoids naive Pillow conversion differences (still uses lcms internally).
    """
    if im.mode == 'RGB':
        return im
    try:
        src_prof = None
        if embedded_icc:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(embedded_icc))
            except Exception:
                src_prof = None
        if src_prof is None:
            if im.mode == 'CMYK':
                # Fallback generic CMYK -> sRGB
                return im.convert('RGB')
            return im.convert('RGB')
        dst_prof = ImageCms.createProfile('sRGB')
        return ImageCms.profileToProfile(im, src_prof, dst_prof, outputMode='RGB')
    except Exception:
        return im.convert('RGB')


def normalize_for_comparison(meta: ImageMeta, allow_convert_cmyk: bool) -> np.ndarray:
    im = meta.pil_image
    mode = im.mode
    if mode == 'P':
        im = im.convert('RGBA')
        mode = im.mode
    if allow_convert_cmyk:
        if mode == 'CMYK':
            im = _ensure_rgb_with_profile(im, meta.icc_profile)
            mode = im.mode
    # Standard direct modes
    if mode in ('RGB', 'RGBA', 'CMYK', 'L'):
        arr = np.array(im)
        # Ensure 2D grayscale becomes HxW x1 for uniformity
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr
    # Fallback convert to RGBA
    arr = np.array(im.convert('RGBA'))
    return arr


def compare_arrays(a: np.ndarray, b: np.ndarray, tolerance: int) -> Dict[str, Any]:
    if a.shape != b.shape:
        return {
            'same_shape': False,
            'shape_a': a.shape,
            'shape_b': b.shape,
            'mismatch_pixels': None,
            'max_diff': None,
            'mean_diff': None
        }
    diff = a.astype(np.int16) - b.astype(np.int16)
    abs_diff = np.abs(diff)
    max_diff = int(abs_diff.max()) if abs_diff.size else 0
    ok_mask = abs_diff <= tolerance
    all_ok = ok_mask.all()
    mean_diff = float(abs_diff.mean()) if abs_diff.size else 0.0
    # Count pixels with ANY channel over tolerance
    per_pixel_over = (~np.all(ok_mask, axis=2)) if ok_mask.ndim == 3 else (~ok_mask)
    mismatch_count = int(per_pixel_over.sum())
    return {
        'same_shape': True,
        'all_within_tolerance': all_ok,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'mismatch_pixels': mismatch_count
    }


def compare_images(
    path_a: str,
    path_b: str,
    allow_convert_cmyk: bool = False,
    tolerance: int = 0,
    max_mismatches: int = 20
) -> Dict[str, Any]:
    meta_a = load_image(path_a)
    meta_b = load_image(path_b)

    report: Dict[str, Any] = {
        'path_a': path_a,
        'path_b': path_b,
        'dimensions_equal': meta_a.size == meta_b.size,
        'size_a': meta_a.size,
        'size_b': meta_b.size,
        'mode_a': meta_a.mode,
        'mode_b': meta_b.mode,
        'modes_equal': meta_a.mode == meta_b.mode,
        'icc_present_a': meta_a.icc_profile is not None,
        'icc_present_b': meta_b.icc_profile is not None,
        'icc_md5_a': meta_a.icc_md5,
        'icc_md5_b': meta_b.icc_md5,
        'icc_len_a': meta_a.icc_len,
        'icc_len_b': meta_b.icc_len,
        'icc_equal': (meta_a.icc_md5 == meta_b.icc_md5) if (meta_a.icc_md5 and meta_b.icc_md5) else (meta_a.icc_profile is None and meta_b.icc_profile is None),
    }

    # Prepare arrays for pixel comparison
    arr_a = normalize_for_comparison(meta_a, allow_convert_cmyk)
    arr_b = normalize_for_comparison(meta_b, allow_convert_cmyk)
    arr_cmp = compare_arrays(arr_a, arr_b, tolerance)
    report.update(arr_cmp)

    # Collect mismatch samples if needed
    mismatch_samples: List[Dict[str, Any]] = []
    if arr_cmp.get('same_shape') and not arr_cmp.get('all_within_tolerance') and max_mismatches > 0:
        # Build boolean mask of over-tolerance
        diff = arr_a.astype(np.int16) - arr_b.astype(np.int16)
        abs_diff = np.abs(diff)
        if abs_diff.ndim == 3:
            over = np.any(abs_diff > tolerance, axis=2)
        else:
            over = abs_diff > tolerance
        ys, xs = np.where(over)
        for y, x in zip(ys[:max_mismatches], xs[:max_mismatches]):
            val_a = arr_a[y, x].tolist() if arr_a.ndim == 3 else int(arr_a[y, x])
            val_b = arr_b[y, x].tolist() if arr_b.ndim == 3 else int(arr_b[y, x])
            mismatch_samples.append({'x': int(x), 'y': int(y), 'a': val_a, 'b': val_b})
    report['mismatch_samples'] = mismatch_samples

    # Overall pass criteria
    report['pass'] = (
        report['dimensions_equal'] and
        report['icc_equal'] and
        report['same_shape'] and
        report['all_within_tolerance']
    )
    return report


def print_human(report: Dict[str, Any]):
    print("=== Image Comparison Report ===")
    print(f"A: {report['path_a']}")
    print(f"B: {report['path_b']}")
    print(f"Size A: {report['size_a']} | Size B: {report['size_b']} -> {'OK' if report['dimensions_equal'] else 'DIFF'}")
    print(f"Mode A: {report['mode_a']} | Mode B: {report['mode_b']} -> {'OK' if report['modes_equal'] else 'DIFF'}")
    print(f"ICC A: {report['icc_md5_a']} (len {report['icc_len_a']})")
    print(f"ICC B: {report['icc_md5_b']} (len {report['icc_len_b']}) -> {'OK' if report['icc_equal'] else 'DIFF'}")
    if not report['same_shape']:
        print(f"Pixel array shape khác: {report.get('shape_a')} vs {report.get('shape_b')}")
    else:
        print(f"Pixel max diff: {report['max_diff']} | mean diff: {report['mean_diff']:.4f}")
        print(f"Mismatch pixels: {report['mismatch_pixels']} | Within tolerance: {report['all_within_tolerance']}")
        if report['mismatch_samples']:
            print("-- Sample mismatches --")
            for s in report['mismatch_samples']:
                print(f"(x={s['x']}, y={s['y']}) A={s['a']} B={s['b']}")
    print(f"RESULT: {'PASS' if report['pass'] else 'FAIL'}")


def parse_args():
    p = argparse.ArgumentParser(description="So sánh hai ảnh: ICC, mode, kích thước, pixel.")
    # p.add_argument('image_a')
    # p.add_argument('image_b')
    p.add_argument('--allow-convert-cmyk', action='store_true', help='Cho phép chuyển CMYK -> RGB để so sánh trong RGB')
    p.add_argument('--tolerance', type=int, default=0, help='Sai số tối đa mỗi kênh (mặc định 0)')
    p.add_argument('--max-mismatches', type=int, default=20, help='Giới hạn số mismatch in ra')
    p.add_argument('--json', action='store_true', help='In kết quả dạng JSON')
    return p.parse_args()


def main():
    args = parse_args()
    report = compare_images(
        # args.image_a,
        # args.image_b,
        "_output/final_output.tif",
        # "_output/final_output.tif",
        "_output/final_result.tif",
        # "mzdzl7z73fd.jpeg",
        allow_convert_cmyk=args.allow_convert_cmyk,
        tolerance=args.tolerance,
        max_mismatches=args.max_mismatches
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_human(report)
    sys.exit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
