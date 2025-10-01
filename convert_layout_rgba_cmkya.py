"""convert_layout_rgba_cmkya.py

Chức năng: Mở một file RGBA, loại bỏ (bỏ qua) toàn bộ pixel alpha = 0, chuyển màu phần còn lại sang CMYK
và xuất ra TIFF CMYK + kênh alpha (CMYKA) nếu có `tifffile`, hoặc fallback: TIFF CMYK + file alpha riêng.

Tóm tắt pipeline:
 1. Load ảnh RGBA (PNG/TIFF/...) bằng Pillow.
 2. Lấy alpha channel -> tạo mask (alpha > threshold => 255, còn lại 0).
 3. Chuyển RGB sang CMYK thông qua ICC transform (nếu có profile nguồn / đích) hoặc Pillow chuyển đổi mặc định.
 4. Thiết lập các kênh C,M,Y,K = 0 tại vùng alpha == 0 để "loại bỏ" pixel không hiển thị.
 5. Ghi TIFF 5 kênh (C,M,Y,K,A) với tag ICC nếu có `tifffile`; nếu không, lưu CMYK + alpha.png.

Sử dụng:
    python3 convert_layout_rgba_cmkya.py \
        --input layout_rgba.png \
        --output layout_cmyka.tif \
        [--target-icc path/to/target.icc] \
        [--source-icc path/to/source.icc] \
        [--alpha-threshold 0]

Ghi chú:
 - Pixel alpha <= threshold sẽ bị coi là trong suốt và đặt C=M=Y=K=0, A=0.
 - Nếu không có ICC nguồn: giả định sRGB.
 - Nếu không có ICC đích: tạo profile CMYK mặc định (ImageCms.createProfile('CMYK')).
 - Chuyển đổi CMYK sử dụng LittleCMS qua Pillow ImageCms.
 - Lưu kênh alpha dưới dạng ExtraSamples (Associated Alpha) nếu có tifffile.
 - Với ảnh rất lớn, có thể thêm tuỳ chọn xử lý theo block (chưa triển khai ở đây).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional
from PIL import Image, ImageCms
import numpy as np
import io

try:
    import tifffile  # type: ignore
    _HAS_TIFFFILE = True
except Exception:
    _HAS_TIFFFILE = False


def load_rgba(path: str) -> Image.Image:
    im = Image.open(path)
    im.load()
    if im.mode not in ("RGBA", "RGB", "LA", "P"):
        # Chuyển sang RGBA để unify pipeline
        im = im.convert("RGBA")
    elif im.mode == 'RGB':
        # Thêm alpha full 255
        im = im.convert('RGBA')
    elif im.mode == 'P':
        im = im.convert('RGBA')
    return im


def build_profiles(src_img: Image.Image, source_icc_path: Optional[str], target_icc_path: Optional[str]):
    """Trả về (src_prof, dst_prof_or_none, use_profile_transform).

    Ghi chú: Pillow ImageCms.createProfile không hỗ trợ 'CMYK' -> tránh gọi createProfile('CMYK').
    Nếu không có ICC CMYK đích, ta sẽ fallback dùng rgb_img.convert('CMYK') (không quản lý màu ICC chính xác nhưng tránh crash).
    """
    # Source profile
    if source_icc_path and os.path.exists(source_icc_path):
        try:
            src_prof = ImageCms.ImageCmsProfile(source_icc_path)
        except Exception as e:
            print(f"[WARN] Không load được source ICC ({e}) -> fallback sRGB")
            src_prof = ImageCms.createProfile('sRGB')
    else:
        embedded = src_img.info.get('icc_profile')
        if embedded:
            try:
                src_prof = ImageCms.ImageCmsProfile(io.BytesIO(embedded))
            except Exception:
                src_prof = ImageCms.createProfile('sRGB')
        else:
            src_prof = ImageCms.createProfile('sRGB')

    # Target profile
    dst_prof = None
    if target_icc_path and os.path.exists(target_icc_path):
        try:
            dst_prof = ImageCms.ImageCmsProfile(target_icc_path)
        except Exception as e:
            print(f"[WARN] Không load được target ICC ({e}) -> fallback convert('CMYK') không ICC")
            dst_prof = None
    else:
        # Không có file ICC đích -> dùng convert('CMYK') fallback
        dst_prof = None

    use_transform = dst_prof is not None
    return src_prof, dst_prof, use_transform


def convert_rgba_to_cmyk_with_alpha(
    rgba_img: Image.Image,
    src_prof: ImageCms.ImageCmsProfile,
    dst_prof: Optional[ImageCms.ImageCmsProfile],
    use_transform: bool,
    alpha_threshold: int = 0,
    rendering_intent: int = 0,
    black_point_compensation: bool = True,
):
    # RGBA -> tách
    r, g, b, a = rgba_img.split()
    alpha_np = np.array(a, dtype=np.uint8)
    # Mask alpha giữ (alpha > threshold)
    keep_mask = (alpha_np > alpha_threshold)

    # Chuẩn bị ảnh RGB cho chuyển màu
    rgb_img = Image.merge('RGB', (r, g, b))

    # Xây transform
    flags_val = 0
    if black_point_compensation and hasattr(ImageCms, 'FLAGS') and 'BLACKPOINTCOMPENSATION' in ImageCms.FLAGS:
        flags_val |= ImageCms.FLAGS['BLACKPOINTCOMPENSATION']

    if use_transform:
        try:
            transform = ImageCms.buildTransform(
                src_prof,
                dst_prof,
                'RGB',
                'CMYK',
                renderingIntent=rendering_intent,
                flags=flags_val
            )
            cmyk_img = ImageCms.applyTransform(rgb_img, transform)
        except Exception as e:
            print(f"[WARN] Transform CMYK ICC thất bại ({e}) -> fallback convert('CMYK')")
            cmyk_img = rgb_img.convert('CMYK')
    else:
        cmyk_img = rgb_img.convert('CMYK')
    c, m, y, k = cmyk_img.split()

    # Áp mask: chỗ alpha=0 thì set C=M=Y=K=0, alpha=0
    def mask_channel(ch: Image.Image) -> Image.Image:
        arr = np.array(ch, dtype=np.uint8)
        arr[~keep_mask] = 0
        return Image.fromarray(arr, mode='L')

    c = mask_channel(c)
    m = mask_channel(m)
    y = mask_channel(y)
    k = mask_channel(k)

    # Alpha output: 255 nơi keep, 0 không keep
    out_alpha = np.where(keep_mask, 255, 0).astype(np.uint8)

    return (c, m, y, k, out_alpha)


def write_cmyka_tiff(
    output_path: str,
    channels: tuple,
    icc_profile_bytes: Optional[bytes],
    compress: str = 'deflate'
):
    c, m, y, k, a = channels
    c_arr = np.array(c, dtype=np.uint8)
    m_arr = np.array(m, dtype=np.uint8)
    y_arr = np.array(y, dtype=np.uint8)
    k_arr = np.array(k, dtype=np.uint8)
    a_arr = np.array(a, dtype=np.uint8)
    stacked = np.dstack([c_arr, m_arr, y_arr, k_arr, a_arr])  # H W 5
    extratags = [
        (338, 1, 1, b'\x02', False)  # ExtraSamples: Associated Alpha
    ]
    if icc_profile_bytes:
        extratags.append((34675, 7, len(icc_profile_bytes), icc_profile_bytes, False))
    tifffile.imwrite(
        output_path,
        stacked,
        photometric='separated',
        planarconfig='contig',
        extratags=extratags,
        compression=compress,
        metadata=None
    )


def fallback_save(
    output_path: str,
    channels: tuple,
    icc_profile_bytes: Optional[bytes]
):
    # Lưu CMYK thường + alpha riêng
    c, m, y, k, a = channels
    from icc_utils import save_image_with_icc
    cmyk_img = Image.merge('CMYK', (c, m, y, k))
    save_image_with_icc(
        cmyk_img,
        output_path,
        icc_profile=icc_profile_bytes,
        mode='CMYK',
        prefer_format='TIFF'
    )
    base, ext = os.path.splitext(output_path)
    Image.fromarray(np.array(a, dtype=np.uint8), mode='L').save(base + '_alpha.png')
    print('[INFO] Fallback saved CMYK:', output_path, 'Alpha:', base + '_alpha.png')


def parse_args():
    p = argparse.ArgumentParser(description='Convert RGBA -> CMYKA (giữ alpha, bỏ pixel alpha=0)')
    p.add_argument('--input', '-i', required=True, help='Input RGBA image path')
    p.add_argument('--output', '-o', required=True, help='Output CMYKA TIFF path')
    p.add_argument('--source-icc', help='Optional source ICC override')
    p.add_argument('--target-icc', help='Optional target CMYK ICC profile')
    p.add_argument('--alpha-threshold', type=int, default=0, help='Ngưỡng alpha coi là trong suốt (mặc định 0)')
    p.add_argument('--intent', type=int, default=0, choices=[0,1,2,3], help='Rendering intent')
    p.add_argument('--no-bpc', action='store_true', help='Tắt black point compensation')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print('[ERR] Input file không tồn tại:', args.input)
        return 1

    rgba = load_rgba(args.input)
    print('[INFO] Loaded input:', args.input, 'Mode:', rgba.mode, 'Size:', rgba.size)
    src_prof, dst_prof, use_transform = build_profiles(rgba, args.source_icc, args.target_icc)

    channels = convert_rgba_to_cmyk_with_alpha(
        rgba,
        src_prof,
        dst_prof,
        use_transform,
        alpha_threshold=args.alpha_threshold,
        rendering_intent=args.intent,
        black_point_compensation=not args.no_bpc
    )

    embedded_icc_bytes = rgba.info.get('icc_profile')
    icc_bytes = None
    try:
        # Dùng target ICC nếu có; nếu không, ưu tiên ICC nguồn
        if args.target_icc and os.path.exists(args.target_icc):
            with open(args.target_icc, 'rb') as f:
                icc_bytes = f.read()
        elif embedded_icc_bytes:
            icc_bytes = embedded_icc_bytes
    except Exception:
        icc_bytes = embedded_icc_bytes

    if _HAS_TIFFFILE:
        try:
            write_cmyka_tiff(args.output, channels, icc_bytes)
            print('[OK] Saved CMYKA TIFF:', args.output)
            return 0
        except Exception as e:
            print('[WARN] Ghi CMYKA trực tiếp thất bại:', e)
    # Fallback
    fallback_save(args.output, channels, icc_bytes)
    return 0


if __name__ == '__main__':
    sys.exit(main())
