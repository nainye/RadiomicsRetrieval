"""Retrieve top-k similar samples from a RadiomicsRetrieval DB.

Modes (--mode):
    img         image embedding cosine                    (uses db/img_embeddings/)
    rad         all 72 radiomics → transtab embedding cosine
    shape       14 Shape features
    firstorder  18 First-order features                   (alias: hist)
    texture     40 GLCM ∪ GLSZM features
    glcm        24 GLCM features
    glszm       16 GLSZM features
    feature     1 specific feature, --feature-name FOO

For non-img modes, the chosen feature subset is forwarded through transtab
(which was trained with random column-subset augmentation including a single
column, so 1-feature inputs still work — just less reliably). The DB is
embedded once and cached at <db-dir>/_cache_<mode>[_<feature>].npy.

Query is one of:
    --query-id ID                              # sample already in the DB
    --image PATH --seg PATH [--ape PATH] [--point X Y Z]
                                               # --ape only required for --mode img

Examples:
    python retrieve.py --mode img --query-id LUNG1-001_1
    python retrieve.py --mode shape --query-id LUNG1-001_1
    python retrieve.py --mode feature --feature-name Shape_Elongation --query-id LUNG1-001_1
    python retrieve.py --mode rad --image x.nii.gz --seg y.nii.gz
    python retrieve.py --mode img --image x.nii.gz --seg y.nii.gz --ape z.npy
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from extract_single_sample import (
    PATCH_SIZE,
    load_nii_xyz, compute_crop_start,
    extract_radiomics_features, normalize_radiomics,
)
from radiomicsRetrieval_withAPE import build_RadiomicsRetireval
import transtab

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

GROUP_PREFIXES = {
    'rad':        ('Shape', 'Hist', 'GLCM', 'GLSZM'),
    'shape':      ('Shape',),
    'firstorder': ('Hist',),
    'hist':       ('Hist',),
    'texture':    ('GLCM', 'GLSZM'),
    'glcm':       ('GLCM',),
    'glszm':      ('GLSZM',),
}
ALL_MODES = ['img'] + list(GROUP_PREFIXES.keys()) + ['feature']


def cosine_topk(query, embs, top_k, exclude_idx=None):
    """query: (D,), embs: (N, D)."""
    q = query / (np.linalg.norm(query) + 1e-12)
    e = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sim = e @ q
    if exclude_idx is not None:
        sim[exclude_idx] = -np.inf
    order = np.argsort(-sim)
    order = order[np.isfinite(sim[order])][:top_k]
    return order, sim[order]


def load_img_db(db_dir):
    emb_dir = os.path.join(db_dir, 'img_embeddings')
    if not os.path.isdir(emb_dir):
        raise SystemExit(f"DB folder not found: {emb_dir}. Run build_db.py first.")
    files = sorted(f for f in os.listdir(emb_dir) if f.endswith('.npy'))
    if not files:
        raise SystemExit(f"No .npy files in {emb_dir}")
    ids = [f[:-4] for f in files]
    embs = np.stack([np.load(os.path.join(emb_dir, f)) for f in files], axis=0)
    return ids, embs


def load_radiomics_db(db_dir):
    with open(os.path.join(db_dir, 'radiomics_normalized.json')) as f:
        rad = json.load(f)
    with open(os.path.join(db_dir, 'feature_names.json')) as f:
        names = json.load(f)
    ids = sorted(rad.keys())
    arr = np.array([[rad[i][n] for n in names] for i in ids], dtype=np.float32)
    return ids, names, arr


def select_feature_columns(names, mode, feature_name):
    if mode == 'feature':
        if feature_name not in names:
            raise SystemExit(
                f"--feature-name {feature_name} not found. "
                f"Examples: {names[:3]} ... {names[-3:]}"
            )
        return [names.index(feature_name)]
    prefixes = GROUP_PREFIXES[mode]
    cols = [i for i, n in enumerate(names) if any(n.startswith(p) for p in prefixes)]
    if not cols:
        raise SystemExit(f"No features matched prefixes {prefixes}")
    return cols


def load_transtab(ckpt_dir, device):
    transtab_dir = os.path.join(ckpt_dir, 'transtab')
    with open(os.path.join(transtab_dir, 'transtab_params.json'), 'r') as f:
        rad_args = json.load(f)
    return transtab.build_radiomics_learner(
        checkpoint=transtab_dir,
        numerical_columns=rad_args['numerical_columns'],
        num_class=4, hidden_dim=128, num_layer=2,
        hidden_dropout_prob=0.1, projection_dim=384,
        activation='leakyrelu',
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        ape_drop_rate=0.0, device=device,
    ).to(device).eval()


def transtab_embed(model_rad, df_subset, batch_size):
    """Forward (N, F) DataFrame through transtab and return (N, D) embeddings."""
    out = []
    n = len(df_subset)
    for start in range(0, n, batch_size):
        chunk = df_subset.iloc[start:start + batch_size].reset_index(drop=True)
        with torch.no_grad():
            multi, _ = model_rad.forward_withSubX([chunk], None)
            out.append(multi[:, 0, :].cpu().numpy())
    return np.concatenate(out, axis=0)


def cache_path(db_dir, mode, feature_name):
    name = f'_cache_{mode}'
    if mode == 'feature':
        name += f'_{feature_name}'
    return os.path.join(db_dir, f'{name}.npy')


def compute_query_img_embedding(args, device):
    _, image_np = load_nii_xyz(args.image)
    _, seg_np = load_nii_xyz(args.seg)
    ape_np = np.load(args.ape)
    if image_np.shape != seg_np.shape or ape_np.shape[1:] != image_np.shape:
        raise SystemExit(f"shape mismatch: img={image_np.shape}, seg={seg_np.shape}, ape={ape_np.shape}")

    xs, ys, zs = compute_crop_start(image_np.shape, seg_np, args.point)
    sl = (slice(xs, xs+PATCH_SIZE), slice(ys, ys+PATCH_SIZE), slice(zs, zs+PATCH_SIZE))
    image_c = image_np[sl].astype(np.float32)
    seg_c = seg_np[sl].astype(np.uint8)
    ape_c = ape_np[:, sl[0], sl[1], sl[2]]
    if image_c.shape != (PATCH_SIZE,) * 3:
        raise SystemExit(f"bad crop shape {image_c.shape}")
    image_c = (image_c - image_c.mean()) / image_c.std()

    if args.point is not None:
        local_pt = (args.point[0]-xs, args.point[1]-ys, args.point[2]-zs)
        if any(c < 0 or c >= PATCH_SIZE for c in local_pt):
            raise SystemExit("--point falls outside the crop")
    else:
        seg_nz = np.stack(np.nonzero(seg_c), axis=1)
        if len(seg_nz) == 0:
            raise SystemExit("Seg empty in crop; pass --point.")
        cand = np.mean(seg_nz, axis=0).astype(int)
        if seg_c[tuple(cand)] == 0:
            cand = seg_nz[np.random.randint(len(seg_nz))]
        local_pt = tuple(int(v) for v in cand)
    print(f"[query] crop=({xs},{ys},{zs}), local point={local_pt}")

    images = torch.from_numpy(image_c).float().unsqueeze(0).unsqueeze(0).to(device)
    apes = torch.from_numpy(ape_c).float().unsqueeze(0).to(device)
    point_coords = torch.tensor([[list(local_pt)]], dtype=torch.long, device=device)
    point_labels = torch.ones(1, 1, dtype=torch.long, device=device)

    model = build_RadiomicsRetireval(
        pretrained=True,
        checkpoint_path=os.path.join(args.ckpt_dir, 'image_model', 'model.pth'),
        num_multimask_outputs=1, num_class=4,
    ).to(device).eval()

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
        emb = model.projection_head(radiomics_token_out).cpu().numpy().squeeze(0)
    return emb


def compute_query_radiomics_normalized(args):
    img_sitk, _ = load_nii_xyz(args.image)
    seg_sitk, _ = load_nii_xyz(args.seg)
    print('[query] extracting radiomics features with PyRadiomics...')
    feats = extract_radiomics_features(img_sitk, seg_sitk)
    with open(args.min_max_json) as f:
        min_max = json.load(f)
    _, normalized = normalize_radiomics(feats, min_max)
    return np.array(normalized, dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="Retrieve top-k similar samples from a RadiomicsRetrieval DB.")
    p.add_argument('--mode', required=True, choices=ALL_MODES,
                   help="Search space.")
    p.add_argument('--feature-name', default=None,
                   help="Required when --mode feature, e.g., Shape_Elongation")
    p.add_argument('--db-dir', default='/workspace/RadiomicsRetrieval/results/checkpoint-lung/db',
                   help="Directory produced by build_db.py")
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--query-id', default=None)
    p.add_argument('--image', default=None)
    p.add_argument('--seg', default=None)
    p.add_argument('--ape', default=None)
    p.add_argument('--point', type=int, nargs=3, default=None, metavar=('X', 'Y', 'Z'))
    p.add_argument('--ckpt-dir', default='/workspace/RadiomicsRetrieval/results/checkpoint-lung')
    p.add_argument('--min-max-json', default=None,
                   help="Default: <db-dir>/radiomics_features_min_max.json")
    p.add_argument('--gt-labels-json', default=None,
                   help="Default: <db-dir>/gt_labels.json")
    p.add_argument('--output-json', default=None)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=64,
                   help="Batch size when forwarding the DB through transtab.")
    p.add_argument('--no-cache', action='store_true',
                   help="Do not read or write the per-mode embedding cache.")
    p.add_argument('--rebuild-cache', action='store_true',
                   help="Force re-forwarding the DB even if a cache exists.")
    p.add_argument('--query-label', default=None, choices=['SCC', 'LCC', 'ADC', 'NOS', 'NaN'],
                   help="Optional ground-truth label for an external query (for label_match column).")
    p.add_argument('--no-aux', action='store_true',
                   help="Skip auxiliary raw-radiomics-cosine and label-match columns.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    if args.mode == 'feature' and not args.feature_name:
        raise SystemExit('--mode feature requires --feature-name')
    if args.min_max_json is None:
        args.min_max_json = os.path.join(args.db_dir, 'radiomics_features_min_max.json')

    gt_labels_path = args.gt_labels_json or os.path.join(args.db_dir, 'gt_labels.json')
    gt_labels = json.load(open(gt_labels_path)) if os.path.exists(gt_labels_path) else None

    # Tracked for the auxiliary raw-radiomics-cosine score; set whenever we run PyRadiomics.
    query_full_radiomics = None

    if args.mode == 'img':
        ids, embs = load_img_db(args.db_dir)
        print(f"[db] mode=img, {len(ids)} entries, dim={embs.shape[1]}")

        exclude_idx = None
        if args.query_id:
            if args.query_id not in ids:
                raise SystemExit(f"--query-id {args.query_id} not in DB")
            idx = ids.index(args.query_id)
            query = embs[idx]
            exclude_idx = idx
            print(f"[query] using stored embedding for {args.query_id}")
        else:
            if not (args.image and args.seg and args.ape):
                raise SystemExit("Provide --query-id OR (--image, --seg, --ape).")
            print('[query] computing image embedding...')
            query = compute_query_img_embedding(args, device)
        order, scores = cosine_topk(query, embs, args.top_k, exclude_idx)

    else:
        ids, names, arr = load_radiomics_db(args.db_dir)
        cols = select_feature_columns(names, args.mode, args.feature_name)
        col_names = [names[i] for i in cols]
        print(f"[db] mode={args.mode}, {len(ids)} samples, using {len(cols)} features")
        if len(col_names) <= 8:
            print(f"[features] {col_names}")

        cache_file = cache_path(args.db_dir, args.mode, args.feature_name)
        if (not args.no_cache) and (not args.rebuild_cache) and os.path.exists(cache_file):
            embs = np.load(cache_file)
            print(f"[cache] loaded {cache_file}  shape={embs.shape}")
            model_rad = None
        else:
            print('[model] loading transtab...')
            model_rad = load_transtab(args.ckpt_dir, device)
            print(f'[forward] embedding {len(ids)} DB samples through transtab...')
            df_db = pd.DataFrame(arr[:, cols], columns=col_names)
            embs = transtab_embed(model_rad, df_db, args.batch_size)
            if not args.no_cache:
                np.save(cache_file, embs)
                print(f"[cache] saved {cache_file}  shape={embs.shape}")

        exclude_idx = None
        if args.query_id:
            if args.query_id not in ids:
                raise SystemExit(f"--query-id {args.query_id} not in DB")
            idx = ids.index(args.query_id)
            query_vec = arr[idx, cols]
            exclude_idx = idx
            print(f"[query] using stored radiomics for {args.query_id}")
        else:
            if not (args.image and args.seg):
                raise SystemExit("Provide --query-id OR (--image, --seg).")
            qfull = compute_query_radiomics_normalized(args)
            query_full_radiomics = qfull
            query_vec = qfull[cols]

        if model_rad is None:
            print('[model] loading transtab for query...')
            model_rad = load_transtab(args.ckpt_dir, device)
        df_q = pd.DataFrame([query_vec], columns=col_names)
        with torch.no_grad():
            multi, _ = model_rad.forward_withSubX([df_q], None)
            query = multi[:, 0, :].cpu().numpy().squeeze(0)
        order, scores = cosine_topk(query, embs, args.top_k, exclude_idx)

    # === Auxiliary scores: raw 72-feature radiomics cosine + label match ===
    raw_sims_full = None
    query_label_resolved = None
    if not args.no_aux:
        # Make sure we have the radiomics matrix aligned with `ids`.
        if args.mode == 'img':
            ids_rad, rad_names, arr_rad = load_radiomics_db(args.db_dir)
            rad_by_id = {x: arr_rad[i] for i, x in enumerate(ids_rad)}
            missing = [x for x in ids if x not in rad_by_id]
            if missing:
                print(f"[warn] {len(missing)} ids missing from radiomics_normalized.json; raw_sim will be NaN for them")
            arr_aligned = np.array(
                [rad_by_id.get(x, np.full(len(rad_names), np.nan)) for x in ids],
                dtype=np.float32,
            )
        else:
            arr_aligned = arr  # already aligned with `ids`

        # Resolve query's full 72-feature vector.
        if query_full_radiomics is None:
            if args.query_id:
                if args.mode == 'img':
                    query_full_radiomics = rad_by_id.get(args.query_id)
                else:
                    query_full_radiomics = arr[ids.index(args.query_id)]
            elif args.image and args.seg:
                query_full_radiomics = compute_query_radiomics_normalized(args)

        if query_full_radiomics is not None and not np.isnan(query_full_radiomics).any():
            q_n = query_full_radiomics / (np.linalg.norm(query_full_radiomics) + 1e-12)
            db_norms = np.linalg.norm(arr_aligned, axis=1, keepdims=True)
            db_n = arr_aligned / (db_norms + 1e-12)
            raw_sims_full = db_n @ q_n  # (N,) — NaN rows propagate
        else:
            print("[warn] query radiomics unavailable; raw_sim column will be omitted")

        # Resolve query label.
        if args.query_label:
            query_label_resolved = args.query_label
        elif args.query_id and gt_labels:
            query_label_resolved = gt_labels.get(args.query_id)

    # === Build results ===
    results = []
    for rank, (i, s) in enumerate(zip(order, scores), 1):
        rid = ids[i]
        e = {'rank': rank, 'id': rid, 'embedding_sim': float(s)}
        if raw_sims_full is not None:
            v = raw_sims_full[i]
            e['raw_radiomics_sim'] = None if np.isnan(v) else float(v)
        if gt_labels and rid in gt_labels:
            e['label'] = gt_labels[rid]
            if query_label_resolved is not None:
                e['label_match'] = (gt_labels[rid] == query_label_resolved)
        results.append(e)

    header = f"\n[result] mode={args.mode}"
    if args.mode == 'feature':
        header += f", feature={args.feature_name}"
    if query_label_resolved is not None:
        header += f", query_label={query_label_resolved}"
    header += f", top-{args.top_k}:"
    print(header)
    for r in results:
        line = f"  {r['rank']:2d}. {r['id']:<22s}  emb={r['embedding_sim']:+.4f}"
        if 'raw_radiomics_sim' in r and r['raw_radiomics_sim'] is not None:
            line += f"  raw={r['raw_radiomics_sim']:+.4f}"
        elif 'raw_radiomics_sim' in r:
            line += f"  raw=    n/a"
        if 'label' in r:
            line += f"  label={str(r['label']):<4s}"
            if 'label_match' in r:
                line += f" {'✓' if r['label_match'] else ' '}"
        print(line)

    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump({
                'mode': args.mode,
                'feature_name': args.feature_name,
                'query_id': args.query_id,
                'query_image': args.image,
                'query_label': query_label_resolved,
                'top_k': args.top_k,
                'results': results,
            }, f, indent=2)
        print(f"\n[saved] {args.output_json}")


if __name__ == '__main__':
    main()
