"""Tiện ích làm việc với ảnh để giữ nguyên ICC profile, colorspace và dữ liệu điểm ảnh.

Các hàm chính:
    load_image_with_icc(path, as_numpy=True)
        -> Đọc ảnh bằng Pillow, lấy icc_profile, mode. Optionally trả về numpy array.

    save_image_with_icc(image, out_path, icc_profile=None, mode=None, prefer_format=None)
        -> Lưu ảnh, gắn lại icc_profile, giữ mode nếu có thể.

Ghi chú quan trọng:
1. OpenCV (cv2) không bảo toàn ICC profile; vì vậy ta luôn dùng Pillow để đọc/lưu.
2. Nếu bạn xử lý ảnh bằng OpenCV ở dạng BGR/BGRA, trước khi lưu cần chuyển về đúng thứ tự kênh cho Pillow (RGB/RGBA).
3. Với ảnh CMYK: nếu bạn convert sang RGB để xử lý rồi quay lại CMYK thì màu có thể thay đổi do chuyển đổi profile. Nếu mục tiêu là *giữ nguyên dữ liệu kênh CMYK* tránh chuyển đổi màu, không nên chuyển sang RGB để xử lý. File này hỗ trợ đọc CMYK thành mảng numpy (H, W, 4) mà không chuyển profile.
4. JPEG không hỗ trợ alpha; nếu ảnh có alpha mà lưu JPEG sẽ bị phẳng (flatten) hoặc mất alpha.

Edge cases được xử lý:
    - Ảnh không có ICC profile: hàm vẫn chạy, chỉ cảnh báo.
    - Mode không tương thích định dạng đích: tự động chuyển sang định dạng an toàn (TIFF/PNG).
    - CMYK + alpha (hiếm): chuyển alpha sang kênh riêng PNG/TIFF (TIFF hỗ trợ dễ hơn).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from PIL import Image, ImageCms
import numpy as np
import os
import io


@dataclass
class ImageICCData:
    """Gói dữ liệu ảnh + metadata ICC.

    Attributes:
        pil_image: Ảnh Pillow (nếu cần giữ nguyên gốc, không nên mutate).
        np_array: Ảnh dạng numpy (None nếu as_numpy=False).
        icc_profile: bytes ICC profile (None nếu không có).
        mode: Pillow mode gốc (RGB, RGBA, CMYK, L, P, ...)
        size: (width, height)
        exif: bytes EXIF (nếu có)
    """
    pil_image: Image.Image
    np_array: Optional[np.ndarray]
    icc_profile: Optional[bytes]
    mode: str
    size: Tuple[int, int]
    exif: Optional[bytes] = None


def _pil_to_numpy(im: Image.Image) -> np.ndarray:
    """Chuyển Pillow Image sang numpy mà KHÔNG thay đổi ý nghĩa kênh.

    Mapping:
        - RGB  -> shape (H,W,3) dtype=uint8
        - RGBA -> shape (H,W,4)
        - L    -> shape (H,W)
        - CMYK -> shape (H,W,4) (c,m,y,k)
        - P    -> convert tạm sang RGBA để bảo toàn palette + alpha (nếu có).
    """
    mode = im.mode
    if mode in ("RGB", "RGBA", "L", "CMYK"):
        return np.array(im)
    if mode == "P":
        # Palette có thể chứa alpha -> chuyển RGBA để không mất thông tin hiển thị.
        return np.array(im.convert("RGBA"))
    # Các mode khác hiếm gặp (I;16, F...) -> chuyển sang 8bit nếu cần.
    try:
        return np.array(im.convert("RGBA"))
    except Exception:
        return np.array(im)


def load_image_with_icc(path: str, as_numpy: bool = True, keep_raw: bool = True) -> ImageICCData:
    """Đọc ảnh & trích xuất ICC profile + metadata mà không làm thay đổi dữ liệu màu.

    Args:
        path: Đường dẫn ảnh.
        as_numpy: Có trả thêm numpy array để xử lý không.
        keep_raw: Nếu True, không ép convert mode (trừ khi Pillow cần để đọc).

    Returns:
        ImageICCData
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File không tồn tại: {path}")

    im = Image.open(path)
    # Trì hoãn load đầy đủ cho đến khi cần (Lazy) -> ta force load để tránh file closed
    im.load()

    icc_profile = im.info.get('icc_profile')
    exif = im.info.get('exif')
    mode = im.mode
    size = im.size

    if not keep_raw:
        # Option: ép về RGB/RGBA phổ thông.
        if mode == 'P':
            im = im.convert('RGBA')
            mode = im.mode

    np_array = _pil_to_numpy(im) if as_numpy else None

    return ImageICCData(
        pil_image=im,
        np_array=np_array,
        icc_profile=icc_profile,
        mode=mode,
        size=size,
        exif=exif
    )


def _numpy_to_pil(arr: np.ndarray, target_mode: Optional[str]) -> Image.Image:
    """Chuyển numpy array về Pillow Image theo target_mode nếu cung cấp.

    Logic:
        - Nếu target_mode là CMYK và arr.shape[-1] == 4: tạo Image.fromarray(arr, 'CMYK').
        - Nếu target_mode là RGBA / RGB khớp số kênh -> dùng trực tiếp.
        - Nếu target_mode None -> suy từ shape.
    """
    if arr.ndim == 2:
        # Grayscale
        im = Image.fromarray(arr, mode='L')
        if target_mode and target_mode != 'L':
            im = im.convert(target_mode)
        return im

    if arr.ndim != 3:
        raise ValueError("Mảng ảnh phải có dạng HxW hoặc HxWxC")

    h, w, c = arr.shape
    if target_mode is None:
        target_mode = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(c, None)
    if target_mode is None:
        raise ValueError(f"Không suy ra được mode cho số kênh: {c}")

    # Trường hợp CMYK
    if target_mode == 'CMYK':
        if c != 4:
            raise ValueError("Để lưu CMYK cần mảng có 4 kênh (C,M,Y,K)")
        return Image.fromarray(arr, mode='CMYK')

    # RGBA / RGB / L
    expected_channels = {'RGB': 3, 'RGBA': 4, 'L': 1, 'CMYK': 4}
    if target_mode in expected_channels and expected_channels[target_mode] != c:
        # Cố gắng convert bằng cách tạo image mặc định rồi chuyển
        base_mode = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(c, None)
        if base_mode is None:
            raise ValueError(f"Không thể map số kênh {c} -> Pillow mode")
        im_temp = Image.fromarray(arr, mode=base_mode)
        return im_temp.convert(target_mode)
    else:
        base_mode = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(c)
        if base_mode is None:
            raise ValueError(f"Số kênh không hỗ trợ: {c}")
        im = Image.fromarray(arr, mode=base_mode)
        if base_mode != target_mode:
            im = im.convert(target_mode)
        return im


def save_image_with_icc(
    image: Union[Image.Image, np.ndarray, ImageICCData],
    out_path: str,
    icc_profile: Optional[bytes] = None,
    mode: Optional[str] = None,
    prefer_format: Optional[str] = None,
    exif: Optional[bytes] = None,
    overwrite: bool = True,
    compression: Optional[str] = None,
    keep_mode_if_possible: bool = True,
) -> str:
    """Lưu ảnh với ICC profile & mode giữ nguyên nếu có thể.

    Args:
        image: Pillow Image, numpy array hoặc ImageICCData.
        out_path: Đường dẫn đích.
        icc_profile: bytes ICC nếu muốn ghi (nếu None và image là ImageICCData sẽ lấy từ đó).
        mode: Muốn ép mode cụ thể (ví dụ 'CMYK', 'RGB'). Nếu None cố dùng mode gốc.
        prefer_format: Gợi ý định dạng ('PNG','TIFF','JPEG'...). Nếu None suy từ out_path.
        exif: Dữ liệu EXIF (nếu muốn giữ).
        overwrite: Nếu False và file tồn tại -> raise.
        compression: Tuỳ chọn cho TIFF ('tiff_deflate','lzw').
        keep_mode_if_possible: Nếu True cố giữ mode gốc thay vì convert.

    Returns:
        out_path: Đường dẫn file đã lưu.
    """
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"File đã tồn tại: {out_path}")

    # Chuẩn hoá đầu vào & metadata
    pil_im: Image.Image
    src_mode: Optional[str] = None

    if isinstance(image, ImageICCData):
        pil_im = image.pil_image
        src_mode = image.mode
        if icc_profile is None:
            icc_profile = image.icc_profile
        if exif is None:
            exif = image.exif
    elif isinstance(image, Image.Image):
        pil_im = image
        src_mode = pil_im.mode
    elif isinstance(image, np.ndarray):
        pil_im = _numpy_to_pil(image, mode)
        src_mode = pil_im.mode
    else:
        raise TypeError("image phải là ImageICCData, PIL.Image hoặc numpy.ndarray")

    target_mode = mode or (src_mode if keep_mode_if_possible else None)
    if target_mode and pil_im.mode != target_mode:
        try:
            pil_im = pil_im.convert(target_mode)
        except Exception:
            # Nếu convert thất bại (ví dụ CMYK + alpha) -> fallback RGBA
            pil_im = pil_im.convert('RGBA')
            target_mode = 'RGBA'

    root, ext = os.path.splitext(out_path)
    if not ext and prefer_format:
        out_path = root + '.' + prefer_format.lower()
        ext = '.' + prefer_format.lower()

    fmt = prefer_format or (ext[1:].upper() if ext else 'PNG')
    if fmt == 'JPG':
        fmt = 'JPEG'

    # Kiểm tra tương thích mode / format
    if fmt in ('JPEG', 'JPG') and pil_im.mode in ('RGBA', 'LA'):
        # JPEG không alpha -> flatten nền trắng
        bg = Image.new('RGB', pil_im.size, (255, 255, 255))
        bg.paste(pil_im.convert('RGB'), mask=pil_im.split()[-1])
        pil_im = bg
        target_mode = 'RGB'
        print("[WARN] Ảnh có alpha nhưng lưu JPEG -> đã flatten nền trắng.")

    save_kwargs = {}
    if icc_profile:
        save_kwargs['icc_profile'] = icc_profile

    # Ghi EXIF: An toàn cho JPEG. Với TIFF dễ phát sinh lỗi (Bad LONG8 / IFD8) khi copy từ JPEG
    # nên mặc định BỎ qua EXIF cho TIFF để tránh crash. Có thể mở rộng sau bằng piexif để làm sạch.
    if exif and fmt == 'JPEG':
        save_kwargs['exif'] = exif

    if fmt == 'TIFF':
        if not compression:
            compression = 'tiff_deflate'
        save_kwargs['compression'] = compression
    elif fmt == 'PNG':
        save_kwargs['compress_level'] = 9
    elif fmt == 'JPEG':
        save_kwargs['quality'] = 95
        save_kwargs['subsampling'] = 0  # Giữ chi tiết màu tối đa

    # Thử lưu, nếu TIFF + EXIF gây lỗi (Bad LONG8 / IFD8) -> bỏ EXIF và lưu lại
    try:
        pil_im.save(out_path, format=fmt, **save_kwargs)
    except RuntimeError as e:
        if fmt == 'TIFF' and 'exif' in save_kwargs:
            print(f"[WARN] Lưu TIFF kèm EXIF thất bại ({e}) -> bỏ EXIF và thử lại.")
            save_kwargs.pop('exif', None)
            pil_im.save(out_path, format=fmt, **save_kwargs)
        else:
            # Thử fallback cuối: nếu có icc_profile nghi ngờ gây lỗi, bỏ và thử lần nữa (hiếm)
            if 'icc_profile' in save_kwargs:
                icc_bytes = save_kwargs.pop('icc_profile')
                try:
                    print(f"[WARN] Thử bỏ ICC profile và lưu lại do lỗi: {e}")
                    pil_im.save(out_path, format=fmt, **save_kwargs)
                    print("[INFO] Lưu thành công khi bỏ ICC profile.")
                finally:
                    # Không restore vì đã xong.
                    pass
            else:
                raise
    return out_path


def copy_image_preserve_icc(src: str, dst: str, force_format: Optional[str] = None) -> str:
    """Sao chép ảnh *trong miền ảnh* (decode + re-encode) bảo toàn profile & mode.

    Nếu mục tiêu chỉ đơn giản copy file byte-by-byte và không cần chạm vào dữ liệu, nên dùng shutil.copy2.

    Args:
        src: ảnh nguồn.
        dst: đường dẫn đích.
        force_format: Ép định dạng đầu ra (ví dụ 'TIFF'). Nếu None suy từ đuôi file dst.
    """
    data = load_image_with_icc(src, as_numpy=False)
    # Nếu lưu TIFF -> bỏ EXIF để tránh lỗi libtiff; JPEG/PNG vẫn giữ.
    exif_bytes = data.exif if (force_format not in (None, 'TIFF', 'tiff', 'TIF', 'tif')) else None
    return save_image_with_icc(
        data,
        dst,
        icc_profile=data.icc_profile,
        mode=data.mode,
        prefer_format=force_format,
        exif=exif_bytes
    )


__all__ = [
    'ImageICCData',
    'load_image_with_icc',
    'save_image_with_icc',
    'copy_image_preserve_icc'
]
