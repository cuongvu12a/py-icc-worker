import cv2
import numpy as np
import json
import os
from pathlib import Path
import time
from PIL import Image, ImageCms
import io

OUTPUT_PATH = os.path.join(".output")
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "pieces"), exist_ok=True)

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def apply_mask(image, mask_path):
    """Apply mask to image - keep non-transparent parts, cut transparent parts"""
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file {mask_path} not found")
        return image
    
    # Load mask with alpha channel
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        print(f"Error: Could not load mask {mask_path}")
        return image
    
    # Ensure image has alpha channel
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    elif len(image.shape) == 2:
        # Grayscale input image
        temp_image = np.zeros((*image.shape, 4), dtype=np.uint8)
        temp_image[:, :, :3] = np.stack([image, image, image], axis=2)
        temp_image[:, :, 3] = 255
        image = temp_image
    
    # Process mask to get alpha values
    if len(mask.shape) == 3 and mask.shape[2] == 4:
        # Mask has alpha channel - use it directly (inverted)
        mask_alpha = 255 - mask[:, :, 3]  # Invert: transparent becomes opaque, opaque becomes transparent
    elif len(mask.shape) == 3 and mask.shape[2] == 3:
        # RGB mask - convert to grayscale and invert
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_alpha = 255 - gray  # Invert: black becomes white, white becomes black
    elif len(mask.shape) == 2:
        # Grayscale mask - invert it
        mask_alpha = 255 - mask
    else:
        print(f"Warning: Unsupported mask format for {mask_path}")
        return image
    
    # Resize mask to match image size if needed
    if mask_alpha.shape[:2] != image.shape[:2]:
        mask_alpha = cv2.resize(mask_alpha, (image.shape[1], image.shape[0]))
    
    # Apply inverted mask - where original mask was opaque (255), keep image; where transparent (0), make transparent
    alpha_normalized = mask_alpha.astype(np.float32) / 255.0
    
    # Apply mask to RGB channels and alpha channel
    for c in range(3):
        image[:, :, c] = (image[:, :, c].astype(np.float32) * alpha_normalized).astype(np.uint8)
    image[:, :, 3] = (image[:, :, 3].astype(np.float32) * alpha_normalized).astype(np.uint8)
    
    return image

def crop_image(image, top, left, width, height):
    """Crop image according to coordinates"""
    h, w = image.shape[:2]
    
    # Ensure coordinates are within image bounds
    top = max(0, min(top, h))
    left = max(0, min(left, w))
    bottom = min(h, top + height)
    right = min(w, left + width)
    
    return image[top:bottom, left:right]

def resize_image(image, width, height):
    """Resize image to specified dimensions"""
    return cv2.resize(image, (width, height))

def rotate_image(image, angle):
    """Rotate image by specified angle"""
    if angle == 0:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation
    if len(image.shape) == 3 and image.shape[2] == 4:
        # Handle alpha channel
        rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    else:
        rotated = cv2.warpAffine(image, M, (new_w, new_h))
    
    return rotated

def process_piece(input_image, partial_config, base_path, debug=False):
    """Process a single piece according to its configuration"""
    piece_id = partial_config['id']
    steps = partial_config['steps']
    
    # Start with a copy of the input image
    current_image = input_image.copy()
    
    print(f"Processing piece: {piece_id}")
    
    for i, step in enumerate(steps):
        action = step['action']
        data = step['data']
        
        print(f"  Step {i+1}: {action}")
        
        if action == 'mask':
            mask_path = os.path.join(base_path, data)
            current_image = apply_mask(current_image, mask_path)
        elif action == 'crop':
            current_image = crop_image(current_image, data['top'], data['left'], data['width'], data['height'])
        elif action == 'resize':
            current_image = resize_image(current_image, data['width'], data['height'])
        elif action == 'rotate':
            current_image = rotate_image(current_image, data)
        
        if debug:
            # Save intermediate step for debugging
            step_filename = os.path.join(OUTPUT_PATH, "pieces", f"{piece_id}_step_{i+1}_{action}.png")
            cv2.imwrite(step_filename, current_image)
            print(f"    Saved intermediate result: {step_filename}")
    
    if debug:
        # Save final piece
        piece_filename = os.path.join(OUTPUT_PATH, "pieces", f"{piece_id}_final.png")
        cv2.imwrite(piece_filename, current_image)
        print(f"  Saved final piece: {piece_filename}")
    
    return current_image

def place_piece_on_layout(layout, piece_image, location):
    """Place a processed piece on the layout at specified location"""
    top = location['top']
    left = location['left']
    
    piece_h, piece_w = piece_image.shape[:2]
    layout_h, layout_w = layout.shape[:2]
    
    # Calculate actual placement bounds
    end_top = min(layout_h, top + piece_h)
    end_left = min(layout_w, left + piece_w)
    actual_piece_h = end_top - top
    actual_piece_w = end_left - left
    
    if actual_piece_h <= 0 or actual_piece_w <= 0:
        print(f"Warning: Piece placement is outside layout bounds")
        return layout
    
    # Handle alpha blending if piece has alpha channel
    if len(piece_image.shape) == 3 and piece_image.shape[2] == 4:
        # BGRA image
        piece_rgb = piece_image[:actual_piece_h, :actual_piece_w, :3]
        piece_alpha = piece_image[:actual_piece_h, :actual_piece_w, 3] / 255.0
        
        # Ensure layout has same number of channels
        if len(layout.shape) == 3 and layout.shape[2] == 3:
            layout = cv2.cvtColor(layout, cv2.COLOR_BGR2BGRA)
        
        # Alpha blending
        for c in range(3):
            layout[top:end_top, left:end_left, c] = (
                layout[top:end_top, left:end_left, c] * (1 - piece_alpha) +
                piece_rgb[:, :, c] * piece_alpha
            )
        
        # Update layout alpha
        layout[top:end_top, left:end_left, 3] = np.maximum(
            layout[top:end_top, left:end_left, 3],
            piece_image[:actual_piece_h, :actual_piece_w, 3]
        )
    else:
        # Regular RGB image - direct placement
        if len(layout.shape) == 3 and layout.shape[2] == 4:
            # Convert piece to BGRA if layout has alpha
            if len(piece_image.shape) == 3 and piece_image.shape[2] == 3:
                piece_bgra = cv2.cvtColor(piece_image[:actual_piece_h, :actual_piece_w], cv2.COLOR_BGR2BGRA)
                piece_bgra[:, :, 3] = 255  # Full opacity
                layout[top:end_top, left:end_left] = piece_bgra
            else:
                layout[top:end_top, left:end_left] = piece_image[:actual_piece_h, :actual_piece_w]
        else:
            layout[top:end_top, left:end_left] = piece_image[:actual_piece_h, :actual_piece_w]
    
    return layout

def main(debug=False):
    # Configuration
    input_image_path = "mzdzl7z73fd.jpeg"
    config_path = "L/config.json"
    layout_path = "L/layout.png"
    output_path = os.path.join(OUTPUT_PATH, "final_output.tif")  # Changed to TIFF
    base_path = "L"
    
    # Load input image with PIL to get ICC profile
    print("Loading input image and ICC profile...")
    pil_input = Image.open(input_image_path)
    icc_profile = pil_input.info.get('icc_profile')
    colorspace = pil_input.mode
    print(f"Input colorspace: {colorspace}")
    if icc_profile:
        print("ICC profile found and will be preserved")
    else:
        print("No ICC profile found in input image")
    
    # Load with OpenCV for processing
    input_image = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)
    if input_image is None:
        print(f"Error: Could not load input image {input_image_path}")
        return
    
    print(f"Input image shape: {input_image.shape}")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)
    partials = config['partials']
    
    # Load layout
    print("Loading layout...")
    layout = cv2.imread(layout_path, cv2.IMREAD_UNCHANGED)
    if layout is None:
        print(f"Error: Could not load layout {layout_path}")
        return
    
    print(f"Layout shape: {layout.shape}")
    
    # Process each piece
    processed_pieces = []
    
    for partial in partials:
        piece_image = process_piece(input_image, partial, base_path, debug)
        processed_pieces.append((piece_image, partial['location']))
    
    # Place all pieces on layout
    print("\nPlacing pieces on layout...")
    final_layout = layout.copy()
    
    for i, (piece_image, location) in enumerate(processed_pieces):
        print(f"Placing piece {i+1} at location top={location['top']}, left={location['left']}")
        final_layout = place_piece_on_layout(final_layout, piece_image, location)
    
    # Save final output as TIFF with ICC profile preservation
    print("Saving final output with ICC profile...")
    
    # Convert OpenCV image (BGR/BGRA) to PIL format
    if len(final_layout.shape) == 3:
        if final_layout.shape[2] == 4:
            # BGRA -> RGBA for PIL
            final_layout_pil = cv2.cvtColor(final_layout, cv2.COLOR_BGRA2RGBA)
            pil_mode = 'RGBA'
        else:
            # BGR -> RGB for PIL
            final_layout_pil = cv2.cvtColor(final_layout, cv2.COLOR_BGR2RGB)
            pil_mode = 'RGB'
    else:
        # Grayscale
        final_layout_pil = final_layout
        pil_mode = 'L'
    
    # Create PIL image
    final_pil_image = Image.fromarray(final_layout_pil, mode=pil_mode)
    
    # If we have an ICC profile, attach it to the output
    save_kwargs = {
        'format': 'TIFF',
        'compression': 'lzw',  # Lossless compression
        'tiffinfo': {}
    }
    
    if icc_profile:
        save_kwargs['icc_profile'] = icc_profile
        print("ICC profile attached to output")
    
    # Save the image
    final_pil_image.save(output_path, **save_kwargs)
    print(f"Final output saved to: {output_path} (TIFF format, colorspace preserved)")
    
    print("\nProcessing completed!")

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Total processing time: {end - start:.2f} seconds")
