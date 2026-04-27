"""Build retrieval DB: image embedding per sample + normalized radiomics + labels.

For every sample in --jsonl, this:
  1. crops a 128^3 patch around the tumor (seg bbox center)
  2. forwards it through the image encoder + mask decoder + projection head
  3. saves the resulting image embedding to <output-dir>/img_embeddings/<id>.npy
  4. records the precomputed (normalized) radiomics features and label

Outputs (under --output-dir):
    img_embeddings/<id>.npy      # (D,) image embedding
    radiomics_normalized.json    # {<id>: {feature_name: normalized_value}}
    feature_names.json           # ordered list of 72 feature names
    gt_labels.json               # {<id>: 'SCC' | 'LCC' | 'ADC' | 'NOS' | 'NaN'}
    radiomics_features_min_max.json   # copied for retrieve.py to reuse
    skipped.json                 # {<id>: reason} (only if any were skipped)
"""

import argparse
import json
import os
import shutil

import numpy as np
import torch
from tqdm import tqdm

from radiomicsRetrieval_withAPE import build_RadiomicsRetireval
from extract_single_sample import (
    PATCH_SIZE,
    load_nii_xyz, compute_crop_start, normalize_radiomics,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    p = argparse.ArgumentParser(description="Build image-embedding retrieval DB.")
    p.add_argument('--jsonl', default='/workspace/RadiomicsRetrieval/data/NSCLC/total.jsonl')
    p.add_argument('--data-root', default='/workspace/RadiomicsRetrieval/data/NSCLC',
                   help="Root containing images/, labels/, apes_npy/")
    p.add_argument('--result-dir', default='/workspace/RadiomicsRetrieval/results')
    p.add_argument('--model-name', default='checkpoint-lung')
    p.add_argument('--ckpt-dir', default=None,
                   help="Full ckpt dir; overrides --result-dir/--model-name when set.")
    p.add_argument('--min-max-json', default=None)
    p.add_argument('--output-dir', default=None,
                   help="Default: <ckpt_dir>/db")
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None, help="Process only the first N items (debug).")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ckpt_dir = args.ckpt_dir or os.path.join(args.result_dir, args.model_name)
    output_dir = args.output_dir or os.path.join(ckpt_dir, 'db')
    min_max_path = args.min_max_json or os.path.join(args.data_root, 'radiomics_features_min_max.json')

    img_emb_dir = os.path.join(output_dir, 'img_embeddings')
    os.makedirs(img_emb_dir, exist_ok=True)

    print(f"[paths]\n  jsonl : {args.jsonl}\n  data  : {args.data_root}\n  ckpt  : {ckpt_dir}\n  out   : {output_dir}")

    with open(min_max_path) as f:
        min_max = json.load(f)
    feature_names = list(min_max.keys())

    print("[model] loading image model...")
    model = build_RadiomicsRetireval(
        pretrained=True,
        checkpoint_path=os.path.join(ckpt_dir, 'image_model', 'model.pth'),
        num_multimask_outputs=1, num_class=4,
    ).to(device).eval()

    items = []
    with open(args.jsonl) as f:
        for line in f:
            items.append(json.loads(line))
    if args.limit is not None:
        items = items[:args.limit]
    print(f"[loop] {len(items)} samples")

    radiomics_db = {}
    gt_labels = {}
    skipped = []

    for item in tqdm(items, desc="building DB"):
        img_id = item['id']
        image_path = os.path.join(args.data_root, 'images', f'{img_id}.nii.gz')
        seg_path = os.path.join(args.data_root, 'labels', f'{img_id}.nii.gz')
        ape_path = os.path.join(args.data_root, 'apes_npy', f'{img_id}.npy')
        if not (os.path.exists(image_path) and os.path.exists(seg_path) and os.path.exists(ape_path)):
            skipped.append((img_id, 'missing files')); continue

        try:
            _, image_np = load_nii_xyz(image_path)
            _, seg_np = load_nii_xyz(seg_path)
            ape_np = np.load(ape_path)
            if image_np.shape != seg_np.shape or ape_np.shape[1:] != image_np.shape:
                skipped.append((img_id, 'shape mismatch')); continue

            _, normalized = normalize_radiomics(item['radiomics'], min_max)

            xs, ys, zs = compute_crop_start(image_np.shape, seg_np, None)
            sl = (slice(xs, xs + PATCH_SIZE), slice(ys, ys + PATCH_SIZE), slice(zs, zs + PATCH_SIZE))
            image_c = image_np[sl].astype(np.float32)
            seg_c = seg_np[sl].astype(np.uint8)
            ape_c = ape_np[:, sl[0], sl[1], sl[2]]
            if image_c.shape != (PATCH_SIZE,) * 3:
                skipped.append((img_id, f'bad crop {image_c.shape}')); continue
            image_c = (image_c - image_c.mean()) / image_c.std()

            seg_nz = np.stack(np.nonzero(seg_c), axis=1)
            if len(seg_nz) == 0:
                skipped.append((img_id, 'empty seg in crop')); continue
            cand = np.mean(seg_nz, axis=0).astype(int)
            if seg_c[tuple(cand)] == 0:
                cand = seg_nz[np.random.randint(len(seg_nz))]
            local_pt = tuple(int(v) for v in cand)

            images = torch.from_numpy(image_c).float().unsqueeze(0).unsqueeze(0).to(device)
            apes = torch.from_numpy(ape_c).float().unsqueeze(0).to(device)
            point_coords = torch.tensor([[list(local_pt)]], dtype=torch.long, device=device)
            point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

            with torch.no_grad():
                image_embeddings = model.image_encoder(images)
                sparse_emb, dense_emb, ape_down = model.prompt_encoder(
                    points=[point_coords, point_labels], ape_map=apes, masks=None,
                )
                image_pe = model.prompt_encoder.get_dense_pe().expand_as(image_embeddings)
                pos_src = image_pe + ape_down if ape_down is not None else image_pe
                _, _, _, _, radiomics_token_out, _ = model.mask_decoder(
                    image_embeddings=image_embeddings, image_pe=pos_src,
                    sparse_prompt_embeddings=sparse_emb,
                    dense_prompt_embeddings=dense_emb,
                    multimask_output=False,
                )
                img_emb = model.projection_head(radiomics_token_out).cpu().numpy().squeeze(0)
        except Exception as e:
            skipped.append((img_id, repr(e))); continue

        np.save(os.path.join(img_emb_dir, f'{img_id}.npy'), img_emb)
        radiomics_db[img_id] = {n: float(v) for n, v in zip(feature_names, normalized)}
        gt_labels[img_id] = item.get('label')

    with open(os.path.join(output_dir, 'radiomics_normalized.json'), 'w') as f:
        json.dump(radiomics_db, f)
    with open(os.path.join(output_dir, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f, indent=2)
    with open(os.path.join(output_dir, 'gt_labels.json'), 'w') as f:
        json.dump(gt_labels, f, indent=4)
    shutil.copy(min_max_path, os.path.join(output_dir, 'radiomics_features_min_max.json'))

    n_total = len(items)
    n_ok = len(radiomics_db)
    n_skip = len(skipped)

    if skipped:
        with open(os.path.join(output_dir, 'skipped.json'), 'w') as f:
            json.dump(dict(skipped), f, indent=2)

    print(f"\n[summary] {n_ok}/{n_total} succeeded, {n_skip} skipped")
    if skipped:
        from collections import Counter
        by_reason = Counter()
        for _, reason in skipped:
            # group long error reprs by their first line / type only
            key = reason.split('\n')[0][:80]
            by_reason[key] += 1
        for reason, count in by_reason.most_common():
            print(f"           {count:4d}  {reason}")
        print(f"           (full list in {os.path.join(output_dir, 'skipped.json')})")
    print(f"[done] wrote DB to {output_dir}")


if __name__ == '__main__':
    main()
