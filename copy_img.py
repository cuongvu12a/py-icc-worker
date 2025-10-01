import time
from icc_utils import load_image_with_icc, save_image_with_icc, copy_image_preserve_icc

def main():
    src = "mzdzl7z73fd.jpeg"
    dst = "final_output.tif"  # Lưu dạng TIFF để giữ ICC & không mất dữ liệu

    # 1. Đọc ảnh và lấy ICC + mode
    data = load_image_with_icc(src, as_numpy=True)
    print(f"Mode gốc: {data.mode}, Kích thước: {data.size}, ICC: {'Có' if data.icc_profile else 'Không'}")

    # 2. (Ví dụ) Thao tác nhẹ trên numpy (không đổi màu) -> ở đây chỉ đọc rồi lưu
    # Nếu cần vẫn có thể truy cập data.np_array

    # 3. Lưu lại với ICC + mode gốc
    out_path = save_image_with_icc(
        data.np_array,  # hoặc dùng trực tiếp data.pil_image
        dst,
        icc_profile=data.icc_profile,
        mode=data.mode,  # Giữ mode gốc nếu có thể
        prefer_format='TIFF'
    )
    print(f"Đã lưu: {out_path}")

    # 4. Ví dụ copy nhanh (decode/encode) sang bản khác
    copy_path = "final_output_copy.tif"
    copy_image_preserve_icc(src, copy_path, force_format='TIFF')
    print(f"Đã sao chép bảo toàn ICC: {copy_path}")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")
