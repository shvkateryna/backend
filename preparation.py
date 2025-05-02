import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import cv2
import facer

def cut_element_from_image(image_path, mask_np):
    image = Image.open(image_path).convert('RGBA')
    if mask_np.shape != image.size[::-1]:
        raise ValueError("Provided mask array must match the image size.")
    image_np = np.array(image)
    alpha = (mask_np > 0).astype(np.uint8) * 255
    result = image_np.copy()
    result[..., 3] = alpha
    return Image.fromarray(result)

def get_average_color_from_masked_image(image_pil):
    image_np = np.array(image_pil.convert("RGB"))
    if image_np.shape[2] == 4:
        alpha = image_np[..., 3]
        mask = alpha > 0
        if not mask.any():
            return np.array([0, 0, 0])
        image_np = image_np[..., :3]
    else:
        mask = np.ones(image_np.shape[:2], dtype=bool)

    masked_pixels = image_np[mask]
    image_hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    avg_color = np.mean(image_hsv.reshape(-1, 3), axis=0)
    return avg_color

def main(image_path):
    sys.path.append('..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_parser = facer.face_parser('farl/celebm/448', device=device)

    img_tensor = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    with torch.inference_mode():
        faces = face_detector(img_tensor)

    if len(faces) == 0:
        print(f"No faces detected in {image_path}, skipping.")
        return None

    with torch.inference_mode():
        faces = face_parser(img_tensor, faces)

    warped_images, grid, inv_grid = face_parser.warp_images(img_tensor / 255.0, faces)
    seg_logits, seg_preds, label_names = face_parser.forward_warped(warped_images)

    target_parts = {
        'skin': 'face',
        'hair': 'hair',
        'l_eye': 'le',
        'r_eye': 're'
    }

    result = {
        'file_name': os.path.splitext(os.path.basename(image_path))[0]
    }

    for face_idx in range(len(seg_preds)):
        seg_pred = seg_preds[face_idx]
        for part, label in target_parts.items():
            if label in label_names:
                label_idx = label_names.index(label)
                warped_mask = (seg_pred == label_idx).float().unsqueeze(0).unsqueeze(0)

                unwarped_mask = F.grid_sample(
                    warped_mask,
                    inv_grid[face_idx:face_idx+1],
                    mode='nearest',
                    align_corners=False
                )

                mask_np = (unwarped_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
                cut_img = cut_element_from_image(image_path, mask_np)

                # --- Change background to black ---
                if cut_img.mode != "RGBA":
                    cut_img = cut_img.convert("RGBA")
                black_bg = Image.new("RGBA", cut_img.size, (0, 0, 0, 255))
                cut_img = Image.alpha_composite(black_bg, cut_img).convert("RGB")
                # -----------------------------------

                avg_color = get_average_color_from_masked_image(cut_img)
                result[f"{part}_H"] = avg_color[0]
                result[f"{part}_S"] = avg_color[1]
                result[f"{part}_V"] = avg_color[2]
            else:
                print(f"Label '{label}' not found in label_names.")
            
        print(result)
        eyes_color = np.mean([[result["l_eye_H"], result["l_eye_S"], result["l_eye_V"]], [result["r_eye_H"], result["r_eye_S"], result["r_eye_V"]]], axis = 0)
        result["eyes_H"] = eyes_color[0]
        result["eyes_S"] = eyes_color[1]
        result["eyes_V"] = eyes_color[2]

    # Add contrast features
    try:
        skin_V = result['face_V']
        hair_V = result['hair_V']
        eyes_V = result.get('l_eye_V', 0)
        result['contrast_score'] = max(skin_V, hair_V, eyes_V) - min(skin_V, hair_V, eyes_V)

        skin_S = result['face_S']
        hair_S = result['hair_S']
        eyes_S = result.get('l_eye_S', 0)
        result['saturation_contrast'] = max(skin_S, hair_S, eyes_S) - min(skin_S, hair_S, eyes_S)
    except KeyError as e:
        print(f"Missing value for contrast calculation in {image_path}: {e}")
        result['contrast_score'] = 0
        result['saturation_contrast'] = 0
    
    result.pop('r_eye_H')
    result.pop('r_eye_S')
    result.pop('r_eye_V')
    result.pop('l_eye_H')
    result.pop('l_eye_S')
    result.pop('l_eye_V')

    print(f"Finished processing {image_path}")
    print(result)
    return result

# на самому кінці preparation.py
def extract_features(image_path: str) -> dict:
    return main(image_path)