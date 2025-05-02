def main(file_like_obj):
    sys.path.append('..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_parser = facer.face_parser('farl/celebm/448', device=device)

    # 💡 читаємо зображення напряму з потоку, без збереження на диск
    image_pil = Image.open(file_like_obj)
    image_np = np.array(image_pil.convert("RGB"))
    img_tensor = facer.hwc2bchw(image_np).to(device=device)

    with torch.inference_mode():
        faces = face_detector(img_tensor)

    if len(faces) == 0:
        print("No faces detected, skipping.")
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

    result = {}

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
                cut_img = cut_element_from_image(image_pil, mask_np)

                # --- Зміна фону на чорний ---
                if cut_img.mode != "RGBA":
                    cut_img = cut_img.convert("RGBA")
                black_bg = Image.new("RGBA", cut_img.size, (0, 0, 0, 255))
                cut_img = Image.alpha_composite(black_bg, cut_img).convert("RGB")
                # ----------------------------

                avg_color = get_average_color_from_masked_image(cut_img)
                result[f"{part}_H"] = avg_color[0]
                result[f"{part}_S"] = avg_color[1]
                result[f"{part}_V"] = avg_color[2]
            else:
                print(f"Label '{label}' not found in label_names.")
        
        eyes_color = np.mean([[result["l_eye_H"], result["l_eye_S"], result["l_eye_V"]],
                              [result["r_eye_H"], result["r_eye_S"], result["r_eye_V"]]], axis=0)
        result["eyes_H"], result["eyes_S"], result["eyes_V"] = eyes_color

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
        print(f"Missing value for contrast calculation: {e}")
        result['contrast_score'] = 0
        result['saturation_contrast'] = 0

    # Видаляємо проміжні ключі очей
    for k in ['r_eye_H', 'r_eye_S', 'r_eye_V', 'l_eye_H', 'l_eye_S', 'l_eye_V']:
        result.pop(k, None)

    return result
