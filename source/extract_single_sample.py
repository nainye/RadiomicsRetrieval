"""Run RadiomicsRetrieval (NSCLC) inference on a single sample.

Given an image (.nii.gz), tumor seg (.nii.gz), and APE (.npy), this script:
  1. extracts 72 PyRadiomics features from image+seg
  2. crops a 128^3 patch around the tumor (or a user-given point)
  3. runs the image+APE encoder and the radiomics encoder
  4. saves embeddings, predicted mask, GT mask, cropped image, and class probs

Usage:
    python extract_single_sample.py --id LUNG1-001_1
    python extract_single_sample.py --id LUNG1-001_1 --point 168 168 134
    python extract_single_sample.py \\
        --id LUNG1-001_1 \\
        --image /path/to/img.nii.gz --seg /path/to/seg.nii.gz --ape /path/to/ape.npy
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

warnings.simplefilter('ignore', DeprecationWarning)
import radiomics
from radiomics import shape as rad_shape, firstorder, glcm, glszm
radiomics.logging.getLogger("radiomics").setLevel(radiomics.logging.ERROR)

from radiomicsRetrieval_withAPE import build_RadiomicsRetireval
import transtab

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CLASS_NAMES = ['SCC', 'LCC', 'ADC', 'NOS']
PATCH_SIZE = 128


def extract_radiomics_features(img_sitk, seg_sitk):
    seg_sitk = seg_sitk != 0
    settings = {'binCount': 128, 'interpolator': None, 'verbose': True}
    feats = {}
    for prefix, cls, kw in [
        ('Shape', rad_shape.RadiomicsShape, {}),
        ('Hist', firstorder.RadiomicsFirstOrder, settings),
        ('GLCM', glcm.RadiomicsGLCM, settings),
        ('GLSZM', glszm.RadiomicsGLSZM, settings),
    ]:
        ext = cls(img_sitk, seg_sitk, **kw)
        ext.enableAllFeatures()
        ext.execute()
        for k, v in ext.featureValues.items():
            feats[f'{prefix}_{k}'] = float(v)
    return feats


def normalize_radiomics(features, min_max):
    names = list(min_max.keys())
    values = []
    for name in names:
        if name not in features:
            raise KeyError(f"Missing radiomics feature: {name}")
        mn, mx = min_max[name]
        values.append((features[name] - mn) / (mx - mn))
    if any(np.isnan(v) for v in values):
        raise ValueError("NaN in radiomics features after normalization")
    return names, values


def load_nii_xyz(path):
    """Read .nii.gz and return (sitk_image, np_array_in_xyz)."""
    s = sitk.ReadImage(path)
    arr = np.transpose(sitk.GetArrayFromImage(s), (2, 1, 0))
    return s, arr


def compute_crop_start(image_shape, seg_np, center_point, patch_size=PATCH_SIZE):
    """Pick a (xs, ys, zs) start so the 128^3 patch covers the tumor (or the point)."""
    shape_x, shape_y, shape_z = image_shape

    if seg_np is not None and seg_np.sum() > 0:
        nz = np.nonzero(seg_np)
        x_min, x_max = nz[0].min(), nz[0].max()
        y_min, y_max = nz[1].min(), nz[1].max()
        z_min, z_max = nz[2].min(), nz[2].max()
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        cz = 0.5 * (z_min + z_max)
    else:
        cx, cy, cz = center_point
        x_min = x_max = int(cx)
        y_min = y_max = int(cy)
        z_min = z_max = int(cz)

    def get_start_range(mn, mx, dim):
        return max(0, mx - (patch_size - 1)), min(dim - patch_size, mn)

    def pick(center, smin, smax):
        if smin > smax:
            print(f'Warning: start_min > start_max, {smin} > {smax}')
            return smin
        ideal = int(np.floor(center - patch_size / 2))
        return max(smin, min(smax, ideal))

    xs = pick(cx, *get_start_range(x_min, x_max, shape_x))
    ys = pick(cy, *get_start_range(y_min, y_max, shape_y))
    zs = pick(cz, *get_start_range(z_min, z_max, shape_z))
    return xs, ys, zs


def build_seg_x_list(df):
    cols = df.columns.tolist()
    sub = [df.copy()]
    for prefix in ['Shape', 'Hist', 'GLCM', 'GLSZM']:
        sub.append(df[[c for c in cols if c.startswith(prefix)]])
    return sub


def label_to_int(label):
    return {'SCC': 0, 'LCC': 1, 'ADC': 2, 'NOS': 3}.get(label, -1)


def main():
    p = argparse.ArgumentParser(description="Single-sample RadiomicsRetrieval inference.")
    p.add_argument('--id', required=True, help="Sample id (used for output filenames), e.g., LUNG1-001_1")
    p.add_argument('--data-root', default='/workspace/RadiomicsRetrieval/data/NSCLC',
                   help="Root that contains images/, labels/, apes_npy/")
    p.add_argument('--image', default=None, help="Override image .nii.gz path")
    p.add_argument('--seg', default=None, help="Override tumor seg .nii.gz path")
    p.add_argument('--ape', default=None, help="Override APE .npy path")
    p.add_argument('--point', type=int, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                   help="Voxel coordinate (x y z) used as the prompt point. If omitted, the seg bbox center is used.")
    p.add_argument('--label', default=None, choices=[*CLASS_NAMES, 'NaN'],
                   help="Optional GT label to record alongside outputs.")
    p.add_argument('--result-dir', default='/workspace/RadiomicsRetrieval/results',
                   help="Root that contains the model checkpoint folder.")
    p.add_argument('--model-name', default='checkpoint-lung',
                   help="Checkpoint folder name under --result-dir.")
    p.add_argument('--ckpt-dir', default=None,
                   help="Full checkpoint dir; overrides --result-dir/--model-name when set.")
    p.add_argument('--min-max-json', default=None,
                   help="Path to radiomics_features_min_max.json (default: <data-root>/radiomics_features_min_max.json)")
    p.add_argument('--output-dir', default=None,
                   help="Where to write outputs (default: <ckpt_dir>/single_sample_results/<id>)")
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    image_path = args.image or os.path.join(args.data_root, 'images', f'{args.id}.nii.gz')
    seg_path = args.seg or os.path.join(args.data_root, 'labels', f'{args.id}.nii.gz')
    ape_path = args.ape or os.path.join(args.data_root, 'apes_npy', f'{args.id}.npy')
    min_max_path = args.min_max_json or os.path.join(args.data_root, 'radiomics_features_min_max.json')
    ckpt_dir = args.ckpt_dir or os.path.join(args.result_dir, args.model_name)
    output_dir = args.output_dir or os.path.join(ckpt_dir, 'single_sample_results', args.id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[paths]\n  image: {image_path}\n  seg  : {seg_path}\n  ape  : {ape_path}\n  ckpt : {ckpt_dir}\n  out  : {output_dir}")

    img_sitk, image_np = load_nii_xyz(image_path)
    seg_sitk, seg_np = load_nii_xyz(seg_path)
    ape_np = np.load(ape_path)

    if image_np.shape != seg_np.shape:
        raise ValueError(f"image/seg shape mismatch: {image_np.shape} vs {seg_np.shape}")
    if ape_np.shape[1:] != image_np.shape:
        raise ValueError(f"ape shape mismatch: {ape_np.shape[1:]} vs {image_np.shape}")

    print("[radiomics] extracting 72 features with PyRadiomics...")
    feats = extract_radiomics_features(img_sitk, seg_sitk)
    with open(min_max_path, 'r') as f:
        min_max = json.load(f)
    feature_names, normalized_values = normalize_radiomics(feats, min_max)

    xs, ys, zs = compute_crop_start(image_np.shape, seg_np, args.point)
    print(f"[crop] start=({xs},{ys},{zs}) size={PATCH_SIZE}")

    sl = (slice(xs, xs + PATCH_SIZE), slice(ys, ys + PATCH_SIZE), slice(zs, zs + PATCH_SIZE))
    image_c = image_np[sl].astype(np.float32)
    seg_c = seg_np[sl].astype(np.uint8)
    ape_c = ape_np[:, sl[0], sl[1], sl[2]]

    if image_c.shape != (PATCH_SIZE,) * 3:
        raise ValueError(f"cropped image has shape {image_c.shape}, expected {(PATCH_SIZE,)*3}")

    image_c = (image_c - image_c.mean()) / image_c.std()

    if args.point is not None:
        px, py, pz = args.point
        local_pt = (px - xs, py - ys, pz - zs)
        if any(c < 0 or c >= PATCH_SIZE for c in local_pt):
            raise ValueError(f"point {args.point} falls outside the 128^3 crop starting at ({xs},{ys},{zs})")
    else:
        seg_nz = np.stack(np.nonzero(seg_c), axis=1)
        if len(seg_nz) == 0:
            raise ValueError("Seg is empty inside the crop; pass --point to specify a prompt.")
        cand = np.mean(seg_nz, axis=0).astype(int)
        if seg_c[tuple(cand)] == 0:
            cand = seg_nz[np.random.randint(len(seg_nz))]
        local_pt = tuple(int(v) for v in cand)
    print(f"[prompt] local point (in crop): {local_pt}")

    images = torch.from_numpy(image_c).float().unsqueeze(0).unsqueeze(0).to(device)
    segs = torch.from_numpy(seg_c).long().unsqueeze(0).unsqueeze(0).to(device)
    apes = torch.from_numpy(ape_c).float().unsqueeze(0).to(device)
    point_coords = torch.tensor([[list(local_pt)]], dtype=torch.long, device=device)
    point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

    rad_df = pd.DataFrame([normalized_values], columns=feature_names)
    ape_at_pt = ape_c[:, local_pt[0], local_pt[1], local_pt[2]].astype(np.float32)
    ape_df = pd.DataFrame(
        [ape_at_pt.tolist()],
        columns=[f'Anatomical_Positional_Embedding_1_{a}' for a in ['x', 'y', 'z']],
    )

    print("[model] loading image model...")
    model_args_path = os.path.join(ckpt_dir, 'image_model', 'params.json')
    image_ckpt_path = os.path.join(ckpt_dir, 'image_model', 'model.pth')
    with open(model_args_path, 'r') as f:
        _ = json.load(f)
    model = build_RadiomicsRetireval(
        pretrained=True,
        checkpoint_path=image_ckpt_path,
        num_multimask_outputs=1,
        num_class=4,
    ).to(device).eval()

    print("[model] loading transtab radiomics model...")
    transtab_dir = os.path.join(ckpt_dir, 'transtab')
    with open(os.path.join(transtab_dir, 'transtab_params.json'), 'r') as f:
        rad_args = json.load(f)
    model_radiomics = transtab.build_radiomics_learner(
        checkpoint=transtab_dir,
        numerical_columns=rad_args['numerical_columns'],
        num_class=4,
        hidden_dim=128,
        num_layer=2,
        hidden_dropout_prob=0.1,
        projection_dim=384,
        activation='leakyrelu',
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        ape_drop_rate=0.0,
        device=device,
    ).to(device).eval()

    print("[infer] running forward passes...")
    with torch.no_grad():
        sub_x_list = build_seg_x_list(rad_df)
        rad_emb_no_ape, logits_no_ape = model_radiomics.forward_withSubX(sub_x_list, None)
        rad_emb, logits = model_radiomics.forward_withSubX(sub_x_list, ape_df)

        image_embeddings = model.image_encoder(images)
        sparse_emb, dense_emb, ape_down = model.prompt_encoder(
            points=[point_coords, point_labels],
            ape_map=apes,
            masks=None,
        )
        image_pe = model.prompt_encoder.get_dense_pe().expand_as(image_embeddings)
        pos_src = image_pe + ape_down if ape_down is not None else image_pe

        low_res_masks, _, _, _, radiomics_token_out, cls_token_out = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=pos_src,
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        image_patch_embeddings = model.projection_head(radiomics_token_out)

        prev_masks = torch.sigmoid(F.interpolate(
            low_res_masks, size=images.shape[-3:], mode='trilinear', align_corners=False
        ))

        img_logits = model.classifier(cls_token_out)
        img_probs = F.softmax(img_logits, dim=1)[0].cpu().numpy()
        rad_probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        rad_probs_no_ape = F.softmax(logits_no_ape, dim=1)[0].cpu().numpy()

    img_probs_dict = {CLASS_NAMES[i]: float(img_probs[i]) for i in range(4)}
    rad_probs_dict = {CLASS_NAMES[i]: float(rad_probs[i]) for i in range(4)}
    rad_probs_no_ape_dict = {CLASS_NAMES[i]: float(rad_probs_no_ape[i]) for i in range(4)}

    print("[save] writing outputs...")
    img_emb_np = image_patch_embeddings.cpu().numpy().squeeze(0)
    rad_emb_np = rad_emb.cpu().numpy().squeeze(0)
    rad_emb_no_ape_np = rad_emb_no_ape.cpu().numpy().squeeze(0)

    np.save(os.path.join(output_dir, f'{args.id}_img_embedding.npy'), img_emb_np)
    np.save(os.path.join(output_dir, f'{args.id}_rad_embedding.npy'), rad_emb_np)
    np.save(os.path.join(output_dir, f'{args.id}_rad_embedding_withoutAPE.npy'), rad_emb_no_ape_np)

    def write_nii(arr_xyz, path, dtype, spacing=(1.5, 1.5, 1.5)):
        arr = np.transpose(arr_xyz.astype(dtype), (2, 1, 0))
        s = sitk.GetImageFromArray(arr)
        s.SetSpacing(spacing)
        sitk.WriteImage(s, path)

    write_nii(image_c, os.path.join(output_dir, f'{args.id}_image.nii.gz'), np.float32)
    write_nii(seg_c, os.path.join(output_dir, f'{args.id}_gt_mask.nii.gz'), np.uint8)
    write_nii(prev_masks.cpu().numpy().squeeze(0).squeeze(0),
              os.path.join(output_dir, f'{args.id}_pred_mask.nii.gz'), np.float32)

    summary = {
        'id': args.id,
        'gt_label': args.label,
        'gt_label_int': label_to_int(args.label) if args.label else None,
        'crop_start_xyz': [int(xs), int(ys), int(zs)],
        'patch_size': PATCH_SIZE,
        'prompt_point_local_xyz': list(local_pt),
        'prompt_point_global_xyz': [int(local_pt[0] + xs), int(local_pt[1] + ys), int(local_pt[2] + zs)],
        'image_pred_probs': img_probs_dict,
        'radiomics_pred_probs': rad_probs_dict,
        'radiomics_pred_probs_withoutAPE': rad_probs_no_ape_dict,
        'radiomics_features_raw': feats,
        'ape_at_prompt_xyz': ape_at_pt.tolist(),
    }
    with open(os.path.join(output_dir, f'{args.id}_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[done] outputs saved to {output_dir}")
    print(f"  image probs        : {img_probs_dict}")
    print(f"  radiomics probs    : {rad_probs_dict}")
    print(f"  rad probs (no APE) : {rad_probs_no_ape_dict}")


if __name__ == '__main__':
    main()
