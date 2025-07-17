import os
import numpy as np
import pandas as pd
import json
import itertools
from functools import partial
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from radiomicsRetrieval_withAPE import build_RadiomicsRetireval
import radiomics_features_constants
import transtab
from monai.losses import DiceCELoss
from datetime import datetime
import h5py
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
wandb.init(entity="radiomicsRetrieval", project="NSCLC_RadiomicsRetrieval")

print("PyTorch version:", torch.__version__)
print("Is CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("Is cuDNN enabled:", torch.backends.cudnn.enabled)

start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

class MyDataset(Dataset):
    def __init__(self, jsonl_file, hdf5_file, root_dir, isTrain=False, rad_drop_rate=0.0, random_crop=False):
        super().__init__()
        self.root_dir = root_dir
        self.isTrain = isTrain
        self.rad_drop_rate = rad_drop_rate
        self.random_crop = random_crop
        self.size = 128

        self.data = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))

        self.hf = h5py.File(hdf5_file, 'r')

        radiomics_features_min_max_path = os.path.join(root_dir, 'radiomics_features_min_max.json')
        with open(radiomics_features_min_max_path, 'r') as f:
            self.radiomics_features_min_max = json.load(f)

    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        if hasattr(self, 'hf'):
            self.hf.close()

    def __getitem__(self, idx):
        item = self.data[idx]

        image_id = item['id']

        grp = self.hf[image_id]
        image_np = grp['image'][:]
        seg_np = grp['tumor'][:]
        ape_np = grp['ape'][:]

        radiomics_features_name = item['radiomics'].keys()
        radiomics_features = []
        for i, feature in enumerate(item['radiomics']):
            feature_value = item['radiomics'][feature]
            min_value, max_value = self.radiomics_features_min_max[feature]
            normalized_value = (feature_value - min_value) / (max_value - min_value)
            radiomics_features.append(normalized_value)
        label = item['label']

        if label == 'SCC':
            label = 0
        elif label == 'LCC':
            label = 1
        elif label == 'ADC':
            label = 2
        elif label == 'NOS':
            label = 3
        else:
            label = -1
        
        seg_nonzero = np.nonzero(seg_np)
        x_min_nz, x_max_nz = seg_nonzero[0].min(), seg_nonzero[0].max()
        y_min_nz, y_max_nz = seg_nonzero[1].min(), seg_nonzero[1].max()
        z_min_nz, z_max_nz = seg_nonzero[2].min(), seg_nonzero[2].max()

        shape_x, shape_y, shape_z = image_np.shape

        def get_start_range(min_nz, max_nz, shape_dim, patch_size):
            start_min = max(0, max_nz - (patch_size - 1))
            start_max = min(shape_dim - patch_size, min_nz)
            return start_min, start_max
        
        def pick_start(center_value, start_min, start_max, patch_size, random_mode):
            if start_min > start_max:
                print(f'Warning: start_min > start_max, {start_min} > {start_max}')
                return start_min
            if random_mode:
                return np.random.randint(start_min, start_max+1)
            else:
                ideal_start = int(np.floor(center_value - patch_size / 2))
                return max(start_min, min(start_max, ideal_start))
            
        x_center = 0.5 * (x_min_nz + x_max_nz)
        y_center = 0.5 * (y_min_nz + y_max_nz)
        z_center = 0.5 * (z_min_nz + z_max_nz)

        x_start_min, x_start_max = get_start_range(x_min_nz, x_max_nz, shape_x, self.size)
        y_start_min, y_start_max = get_start_range(y_min_nz, y_max_nz, shape_y, self.size)
        z_start_min, z_start_max = get_start_range(z_min_nz, z_max_nz, shape_z, self.size)

        x_start = pick_start(x_center, x_start_min, x_start_max, self.size, self.random_crop)
        y_start = pick_start(y_center, y_start_min, y_start_max, self.size, self.random_crop)
        z_start = pick_start(z_center, z_start_min, z_start_max, self.size, self.random_crop)

        image_np = image_np[x_start:x_start+self.size, y_start:y_start+self.size, z_start:z_start+self.size].astype(np.float32)
        seg_np = seg_np[x_start:x_start+self.size, y_start:y_start+self.size, z_start:z_start+self.size]
        ape_np = ape_np[:, x_start:x_start+self.size, y_start:y_start+self.size, z_start:z_start+self.size]

        seg_nonzero = np.nonzero(seg_np)
        all_nonzero_coords = np.stack(seg_nonzero, axis=1) # (N, 3)
        
        image_np = (image_np - image_np.mean()) / image_np.std()

        if image_np.shape != (self.size, self.size, self.size):
            print(f'Warning: image_np.shape={image_np.shape}, expected shape={self.size}')
        if seg_np.shape != (self.size, self.size, self.size):
            print(f'Warning: seg_np.shape={seg_np.shape}, expected shape={self.size}')
        if ape_np.shape != (3, self.size, self.size, self.size):
            print(f'Warning: ape_tensor.shape={ape_np.shape}, expected shape=(3, {self.size}, {self.size}, {self.size})')
        if len(radiomics_features) != 72:
            print(f'Warning: len(radiomics_features)={len(radiomics_features)}, expected length=72')
        return {
            'idx': idx,
            'id': image_id,
            'image': torch.from_numpy(image_np).float().unsqueeze(0),
            'seg': torch.from_numpy(seg_np).long().unsqueeze(0),
            'ape': torch.from_numpy(ape_np).float(),
            'all_nonzero_coords': all_nonzero_coords,
            'radiomics_features': list(radiomics_features),
            'radiomics_features_name': list(radiomics_features_name),
            'label': label
        }

def my_collate_fn(batch, max_num_clicks=1):
    idxes = [item['idx'] for item in batch]
    ids = [item['id'] for item in batch]
    images = torch.stack([item['image'] for item in batch], dim=0)
    segs = torch.stack([item['seg'] for item in batch], dim=0)
    apes = torch.stack([item['ape'] for item in batch], dim=0)
    radiomics_features = [item['radiomics_features'] for item in batch]
    radiomics_features = pd.DataFrame(radiomics_features, columns=batch[0]['radiomics_features_name'])
    labels = [item['label'] for item in batch]

    batch_size = len(batch)
    num_clicks = np.random.randint(1, max_num_clicks+1)

    point_coords = torch.zeros(batch_size, num_clicks, 3, dtype=torch.long)
    point_labels = torch.ones(batch_size, num_clicks, dtype=torch.long)
    ape_values = np.zeros((batch_size, num_clicks, 3)).astype(np.float32)

    for b_idx, item in enumerate(batch):
        all_coords = item['all_nonzero_coords'] # (N, 3)
        N = all_coords.shape[0]

        if num_clicks == 1:
            # pick the center point
            point_x, point_y, point_z = np.mean(all_coords, axis=0).astype(int)
            if item['seg'][0,point_x, point_y, point_z] == 0:
                nonzero_idx = np.random.randint(N)
                point_x, point_y, point_z = all_coords[nonzero_idx]
            chosen_coords = np.array([[point_x, point_y, point_z]]) # (1, 3)
        else:
            chosen_idx = np.random.choice(N, size=num_clicks, replace=(N < num_clicks))
            chosen_coords = all_coords[chosen_idx] # (num_clicks, 3)

        point_coords[b_idx] = torch.from_numpy(chosen_coords).long()
        for c_idx, (x, y, z) in enumerate(chosen_coords):
            ape_values[b_idx, c_idx] = apes[b_idx, :, x, y, z]
    ape_col_names = []
    for i in range(1, num_clicks+1):
        for axis in ['x', 'y', 'z']:
            ape_col_names.append(f'Anatomical_Positional_Embedding_{i}_{axis}')

    ape_values = ape_values.reshape(batch_size, num_clicks*3)
    ape_df = pd.DataFrame(ape_values, columns=ape_col_names)

    return {
        'idxes': idxes,
        'ids': ids,
        'images': images,
        'segs': segs,
        'apes': apes,
        'point_coords': point_coords,
        'point_labels': point_labels,
        'radiomics_features': radiomics_features,
        'radiomics_features_name': batch[0]['radiomics_features_name'],
        'ape_df': ape_df,
        'labels': labels
    }


class PairAugmentBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True, shuffle=True, seed=0):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.group_size = batch_size // 2

    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            indices = torch.randperm(self.num_samples, generator=self.generator).tolist()

        for start_idx in range(0, self.num_samples, self.group_size):
            chunk = indices[start_idx:start_idx+self.group_size]
            if len(chunk) < self.group_size and self.drop_last:
                continue
            
            batch_indices = []
            for idx in chunk:
                batch_indices.append(idx)
                batch_indices.append(idx)

            yield batch_indices

    def __len__(self):
        n_full_chunks = self.num_samples // self.group_size
        if self.drop_last:
            return n_full_chunks
        else:
            remainder = self.num_samples % self.group_size
            return n_full_chunks + (1 if remainder > 0 else 0)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_jsonl_file",
        type=str,
        default=None,
        required=True,
        help="The jsonl file for training data.",
    )
    parser.add_argument(
        "--val_jsonl_file",
        type=str,
        default=None,
        required=True,
        help="The jsonl file for validation data.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        required=True,
        help="The root directory of the data.",
    )
    parser.add_argument(
        "--hdf5_file",
        type=str,
        default=None,
        required=True,
        help="The hdf5 file containing the images and segmentations.",
    )
    parser.add_argument(
        "--transtab_pretrain_dir",
        type=str,
        default=None,
        help="The directory of the pretrained Transtab model.",
    )
    parser.add_argument(
        "--transtab_pretrain_folder",
        type=str,
        default=None,
        help="The folder of the pretrained Transtab model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory of the model.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="The checkpoint to resume training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="Device.",
    )
    parser.add_argument(
        "--contrastive_temperature",
        type=float,
        default=0.07,
        help="Contrastive temperature.",
    )
    parser.add_argument(
        "--contrastive_loss_weight",
        type=float,
        default=1.0,
        help="Contrastive loss weight.",
    )
    parser.add_argument(
        "--cls_loss_weight",
        type=float,
        default=1.0,
        help="Classification loss weight.",
    )
    parser.add_argument(
        "--max_num_clicks",
        type=int,
        default=10,
        help="Maximum number of random clicks.",
    )
    parser.add_argument(
        "--ape_drop_rate",
        type=float,
        default=0.1,
        help="Anatomical Positional Embedding drop rate.",
    )
    parser.add_argument("--accumulate_steps", type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # create model
    ckpt_path = "sam_med3d_turbo.pth"
    if args.resume_checkpoint is not None:
        ckpt_path = os.path.join(args.output_dir, args.resume_checkpoint, 'image_model', 'model.pth')
    print(f'Loading image model from {ckpt_path}')
    model = build_RadiomicsRetireval(pretrained=True, checkpoint_path=ckpt_path, num_multimask_outputs=1, num_class=4)
    model.to(device)
    model.train()
    model.requires_grad_(False)
    model.prompt_encoder.ape_downsampler.requires_grad_(True)
    model.mask_decoder.requires_grad_(True)
    model.projection_head.requires_grad_(True)
    model.classifier.requires_grad_(True)

    if (args.transtab_pretrain_dir is not None) and (args.transtab_pretrain_folder is not None):
        ckpt_path = os.path.join(args.transtab_pretrain_dir, args.transtab_pretrain_folder)
    else:
        ckpt_path = None
        
    if args.resume_checkpoint is not None:
        ckpt_path = os.path.join(args.output_dir, args.resume_checkpoint, 'transtab')
    print(f'Loading radiomics model from {ckpt_path}')
    
    model_radiomics = transtab.build_radiomics_learner(
        checkpoint=ckpt_path,
        numerical_columns=radiomics_features_constants.RADIOMICS_FEATURES_NAMES,
        num_class=4,
        hidden_dim=128,
        num_layer=2,
        hidden_dropout_prob=0.1,
        projection_dim=384,
        activation='leakyrelu',
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        ape_drop_rate=args.ape_drop_rate,
        device=device
    )
    ape_cols = []
    for i in range(1, 11):
        for axis in ['x', 'y', 'z']:
            ape_cols.append(f'Anatomical_Positional_Embedding_{i}_{axis}')
    model_radiomics.input_encoder.feature_extractor.update(num=ape_cols)
    model_radiomics.to(device)
    model_radiomics.train()

    params_to_opt = itertools.chain(
        model.prompt_encoder.ape_downsampler.parameters(),
        model.mask_decoder.parameters(),
        model.projection_head.parameters(),
        model.classifier.parameters(),
        model_radiomics.parameters()
    )
    optimizer = torch.optim.Adam(params_to_opt, lr=args.lr, weight_decay=args.weight_decay)

    train_dataset = MyDataset(args.train_jsonl_file, args.hdf5_file, args.root_dir, isTrain=True, random_crop=True)
    val_dataset = MyDataset(args.val_jsonl_file, args.hdf5_file, args.root_dir, isTrain=False, random_crop=False)

    train_collate_fn = partial(my_collate_fn, max_num_clicks=args.max_num_clicks)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=PairAugmentBatchSampler(
            dataset=train_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            seed=args.seed,
        ),
        collate_fn=train_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_collate_fn = partial(my_collate_fn, max_num_clicks=1)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_sampler=PairAugmentBatchSampler(
            dataset=val_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            seed=args.seed,
        ),
        collate_fn=val_collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    train_losses = dict()
    val_losses = dict()
    val_losses['best'] = 1e9

    train_seg_losses = dict()
    val_seg_losses = dict()

    train_contrastive_losses = dict()
    val_contrastive_losses = dict()
    val_contrastive_losses['best'] = 1e9

    train_dices = dict()
    val_dices = dict()
    val_dices['best'] = 0

    train_class_losses = dict()
    val_class_losses = dict()

    train_img_class_losses = dict()
    val_img_class_losses = dict()

    train_downstream_losses = dict()
    val_downstream_losses = dict()
    val_downstream_losses['best'] = 1e9

    classification_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    dice_ce_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    def get_dice_score(pred_masks, true_masks, threshold=0.5):
        """
        pred_masks: (B, 1, D, H, W) - Apply sigmoid to get values in [0, 1]
        true_masks: (B, 1, D, H, W) - 0/1
        """
        pred_bin = (pred_masks > threshold).float()
        
        # Assumes true_masks are already binary (0 or 1)
        intersection = (pred_bin * true_masks).sum(dim=[1,2,3,4])
        union = pred_bin.sum(dim=[1,2,3,4]) + true_masks.sum(dim=[1,2,3,4])
        
        dice = 2.0 * intersection / (union + 1e-6)  # Avoid division by zero with epsilon
        return dice.mean()

    def simclr_nt_xent_loss_multi_pos(
        embeddings: torch.Tensor,  # [B, d]
        idxes,
        temperature: float = 0.07
    ) -> torch.Tensor:
        device = embeddings.device
        if not isinstance(idxes, torch.Tensor):
            idxes = torch.tensor(idxes, device=device, dtype=torch.long)

        B, d = embeddings.shape
        z = F.normalize(embeddings, dim=1)                  # 1) L2-normalize
        sim_matrix = torch.mm(z, z.t()) / temperature       # 2) (B,B) similarity matrix

        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e4)  # 3) Mask diagonal with large negative value

        pos_mask = (idxes.unsqueeze(1) == idxes.unsqueeze(0)) & (~diag_mask)
        logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        log_prob = sim_matrix - logsumexp

        pos_log_prob_sum = (pos_mask * log_prob).sum(dim=1)
        num_pos = pos_mask.sum(dim=1).clamp_min(1)
        pos_log_prob_mean = pos_log_prob_sum / num_pos

        loss = -pos_log_prob_mean.mean()
        return loss

    def compute_multimodal_contrastive_loss_singleSimCLR(
        image_token_embedding: torch.Tensor,       # [B, 384]
        radiomics_token_embedding: torch.Tensor,   # [B, n_rad, 384]
        idxes: torch.Tensor,                       # [B]
        temperature: float = 0.07
    ):
        """
        Combine image and radiomics embeddings into a single (n_rad+1, d) tensor,
        then compute SimCLR (NT-Xent) loss with positive pairs defined by identical indices.


        Args:
            image_token_embedding: [B, d] — image embeddings
            radiomics_token_embedding: [B, n_rad, d] — radiomics tokens per tumor
            idxes: list or tensor of sample IDs (length B)
            temperature: temperature parameter for NT-Xent loss

        Returns:
            Scalar NT-Xent contrastive loss
        """
        device = image_token_embedding.device
        B, n_rad, d = radiomics_token_embedding.shape

        # 1) Flatten radiomics tokens: [B, n_rad, d] → [B*n_rad, d]
        rad_all = radiomics_token_embedding.view(B*n_rad, d)

        # 2) Concatenate with image tokens → [n_rad+1, d]
        combined = torch.cat([image_token_embedding, rad_all], dim=0)  # (B*(n_rad+1), d)

        # 3) Expand sample indices to match tokens → [B*(n_rad+1)]
        if isinstance(idxes, torch.Tensor):
            idxes_rad = idxes.repeat_interleave(n_rad)  # (B*n_rad,)
            combined_idxes = torch.cat([idxes, idxes_rad], dim=0)  # (B*(n_rad+1),)
        else:
            idxes_rad = []
            for x in idxes:
                idxes_rad.extend([x]*n_rad)
            combined_idxes = list(idxes) + idxes_rad

        # 4) Compute SimCLR loss on all tokens at once
        loss = simclr_nt_xent_loss_multi_pos(combined, combined_idxes, temperature=temperature)
        return loss
    
    max_train_steps = len(train_loader) * args.epochs
    start_epoch = 0
    if args.resume_checkpoint is not None:
        # Extract epoch number from checkpoint name, e.g., "...epoch121_..." → 121
        start_epoch = int(args.resume_checkpoint.split('epoch')[-1].split('_')[0])
    start_step = start_epoch * len(train_loader)
    progress_bar = tqdm(
        range(max_train_steps),
        initial=start_step,
        desc='Steps',
    )
    accumulate_steps = args.accumulate_steps

    scaler = torch.amp.GradScaler()

    def train_epoch(epoch, train_loader, progress_bar):
        model.train()
        model_radiomics.train()

        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_dice = 0.0
        epoch_class_loss = 0.0
        epoch_img_class_loss = 0.0

        for step, batch in enumerate(train_loader):

            optimizer.zero_grad()
            
            images = batch['images'].to(device).float()
            segs = batch['segs'].to(device).float()
            apes = batch['apes'].to(device).float()
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)
            radiomics_features = batch['radiomics_features']
            ape_df = batch['ape_df']
            labels = batch['labels']

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):

                radiomics_embeddings, logits = model_radiomics(radiomics_features, ape_df)
            
                with torch.no_grad():
                    image_embeddings = model.image_encoder(images)

                sparse_embeddings, dense_embeddings, ape_down = model.prompt_encoder(
                    points=[point_coords, point_labels],
                    ape_map=apes,
                    masks=None,
                )
                image_pe = model.prompt_encoder.get_dense_pe().expand_as(image_embeddings)
                if ape_down is not None:
                    pos_src = image_pe + ape_down
                else:
                    pos_src = image_pe

                low_res_masks, iou, mask_tokens_out, iou_token_out, radiomics_token_out, cls_token_out = model.mask_decoder(
                    image_embeddings = image_embeddings,    # (B, 384, 8, 8, 8)
                    image_pe=pos_src,                       # (B, 384, 8, 8, 8)
                    sparse_prompt_embeddings=sparse_embeddings,  # (B, N_total, 384)
                    dense_prompt_embeddings=dense_embeddings,  # (B, 384, 8, 8, 8)
                    multimask_output=False,
                )
                image_patch_embeddings = model.projection_head(radiomics_token_out)
        
                prev_masks = F.interpolate(low_res_masks, size=images.shape[-3:], mode='trilinear', align_corners=False)
                seg_loss = dice_ce_loss(prev_masks, segs)
                dice_score = get_dice_score(torch.sigmoid(prev_masks), segs)

                # radiomics_token_out: (B, 384)
                # contrastive_loss = compute_contrastive_loss(radiomics_token_out, batch['idxes'], margin=args.contrastive_margin)
                contrastive_loss = compute_multimodal_contrastive_loss_singleSimCLR(
                    image_token_embedding=image_patch_embeddings,
                    radiomics_token_embedding=radiomics_embeddings,
                    idxes=batch['idxes'],
                    temperature=args.contrastive_temperature,
                )
                img_logits = model.classifier(cls_token_out)
                img_class_loss = classification_loss(img_logits, torch.tensor(labels).to(device).long())

                class_loss = classification_loss(logits, torch.tensor(labels).to(device).long())

                step_loss = seg_loss + args.contrastive_loss_weight * contrastive_loss + args.cls_loss_weight * class_loss + args.cls_loss_weight *  img_class_loss
            
            scaler.scale(step_loss).backward()
            if ((step + 1) % accumulate_steps == 0) or ((step + 1) == len(train_loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += step_loss.item()
            epoch_seg_loss += seg_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()
            epoch_dice += dice_score.item()
            epoch_class_loss += class_loss.item()
            epoch_img_class_loss += img_class_loss.item()

            progress_bar.set_postfix(
                {
                    'step': step,
                    'seg_loss': seg_loss.item(),
                    'cont_loss': contrastive_loss.item(),
                    'dice': dice_score.item(),
                    'class_loss': class_loss.item(),
                    'img_class_loss': img_class_loss.item(),
                }
            )
            progress_bar.update()

        n_steps = len(train_loader)
        epoch_loss /= n_steps
        epoch_seg_loss /= n_steps
        epoch_contrastive_loss /= n_steps
        epoch_dice /= n_steps
        epoch_class_loss /= n_steps
        epoch_img_class_loss /= n_steps

        return epoch_loss, epoch_seg_loss, epoch_contrastive_loss, epoch_dice, epoch_class_loss, epoch_img_class_loss
    
    def val_epoch(epoch, val_loader, desc='Val'):
        model.eval()
        model_radiomics.eval()

        epoch_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_dice = 0.0
        epoch_class_loss = 0.0
        epoch_img_class_loss = 0.0

        for step, batch in enumerate(tqdm(val_loader, desc=desc)):
            images = batch['images'].to(device).float()
            segs = batch['segs'].to(device).float()
            apes = batch['apes'].to(device).float()
            point_coords = batch['point_coords'].to(device)
            point_labels = batch['point_labels'].to(device)
            radiomics_features = batch['radiomics_features']
            ape_df = batch['ape_df']
            labels = batch['labels']

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    radiomics_embeddings, logits = model_radiomics(radiomics_features, ape_df)

                    image_embeddings = model.image_encoder(images)

                    sparse_embeddings, dense_embeddings, ape_down = model.prompt_encoder(
                        points=[point_coords, point_labels],
                        ape_map=apes,
                        masks=None,
                    )
                    image_pe = model.prompt_encoder.get_dense_pe().expand_as(image_embeddings)
                    if ape_down is not None:
                        pos_src = image_pe + ape_down
                    else:
                        pos_src = image_pe

                    low_res_masks, iou, mask_tokens_out, iou_token_out, radiomics_token_out, cls_token_out = model.mask_decoder(
                        image_embeddings = image_embeddings, # (1, 384, 8, 8, 8)
                        image_pe=pos_src, # (1, 384, 8, 8, 8)
                        sparse_prompt_embeddings=sparse_embeddings,  # (1, 2, 384)
                        dense_prompt_embeddings=dense_embeddings,  # (1, 384, 8, 8, 8)
                        multimask_output=False,
                    )
                    image_patch_embeddings = model.projection_head(radiomics_token_out)
                
                    prev_masks = F.interpolate(low_res_masks, size=images.shape[-3:], mode='trilinear', align_corners=False)
                    seg_loss = dice_ce_loss(prev_masks, segs)
                    dice_score = get_dice_score(torch.sigmoid(prev_masks), segs)

                    # radiomics_token_out: (B, 384)
                    contrastive_loss = compute_multimodal_contrastive_loss_singleSimCLR(
                        image_token_embedding=image_patch_embeddings,
                        radiomics_token_embedding=radiomics_embeddings,
                        idxes=batch['idxes'],
                        temperature=args.contrastive_temperature,
                    )
                    img_logits = model.classifier(cls_token_out)
                    img_class_loss = classification_loss(img_logits, torch.tensor(labels).to(device).long())

                    class_loss = classification_loss(logits, torch.tensor(labels).to(device).long())

                    step_loss = seg_loss + args.contrastive_loss_weight * contrastive_loss + args.cls_loss_weight * class_loss + args.cls_loss_weight * img_class_loss

                    epoch_loss += step_loss.item()
                    epoch_seg_loss += seg_loss.item()
                    epoch_contrastive_loss += contrastive_loss.item()
                    epoch_dice += dice_score.item()
                    epoch_class_loss += class_loss.item()
                    epoch_img_class_loss += img_class_loss.item()

        n_steps = len(val_loader)
        epoch_loss /= n_steps
        epoch_seg_loss /= n_steps
        epoch_contrastive_loss /= n_steps
        epoch_dice /= n_steps
        epoch_class_loss /= n_steps
        epoch_img_class_loss /= n_steps

        return epoch_loss, epoch_seg_loss, epoch_contrastive_loss, epoch_dice, epoch_class_loss, epoch_img_class_loss
    
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_seg_loss, train_contrastive_loss, train_dice, train_class_loss, train_img_class_loss = train_epoch(epoch, train_loader, progress_bar)
        val_loss, val_seg_loss, val_contrastive_loss, val_dice, val_class_loss, val_img_class_loss = val_epoch(epoch, val_loader, desc='Val')

        train_losses[epoch] = train_loss
        train_seg_losses[epoch] = train_seg_loss
        train_contrastive_losses[epoch] = train_contrastive_loss
        train_dices[epoch] = train_dice
        train_class_losses[epoch] = train_class_loss
        train_img_class_losses[epoch] = train_img_class_loss

        val_losses[epoch] = val_loss
        val_seg_losses[epoch] = val_seg_loss
        val_contrastive_losses[epoch] = val_contrastive_loss
        val_dices[epoch] = val_dice
        val_class_losses[epoch] = val_class_loss
        val_img_class_losses[epoch] = val_img_class_loss

        train_downstream_losses[epoch] = train_seg_loss + train_contrastive_loss
        val_downstream_losses[epoch] = val_seg_loss + val_contrastive_loss
        
        print()
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss}, Val Loss: {val_loss}')
        print(f'Train Seg Loss: {train_seg_loss}, Val Seg Loss: {val_seg_loss}')
        print(f'Train Cont Loss: {train_contrastive_loss}, Val Cont Loss: {val_contrastive_loss}')
        print(f'Train Dice: {train_dice}, Val Dice: {val_dice}')
        print(f'Train Class Loss: {train_class_loss}, Val Class Loss: {val_class_loss}')
        print(f'Train Img Class Loss: {train_img_class_loss}, Val Img Class Loss: {val_img_class_loss}')

        if val_seg_loss+val_contrastive_loss < val_downstream_losses['best']:
            print(f'## Best validation downstream performance model saved at epoch {epoch} ##')
            save_folder = os.path.join(args.output_dir, f'best_val_downstream_model_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(save_folder, exist_ok=True)
            save_model_radiomics_folder = os.path.join(save_folder, 'transtab')
            os.makedirs(save_model_radiomics_folder, exist_ok=True)
            model_radiomics.save(ckpt_dir=save_model_radiomics_folder)

            save_model_folder = os.path.join(save_folder, 'image_model')
            os.makedirs(save_model_folder, exist_ok=True)
            model.save(ckpt_dir=save_model_folder)
            train_args = {}
            for k,v in vars(args).items():
                if k != 'device':
                    train_args[k] = v
            with open(os.path.join(save_folder, 'train_args.json'), 'w') as f:
                json.dump(train_args, f, indent=4)
            val_downstream_losses['best'] = val_seg_loss + val_contrastive_loss
            print(f'Saved model to {save_folder}')

        if val_loss < val_losses['best']:
            print(f'** Best validation loss model saved at epoch {epoch} **')
            save_folder = os.path.join(args.output_dir, f'best_val_loss_model_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(save_folder, exist_ok=True)
            save_model_radiomics_folder = os.path.join(save_folder, 'transtab')
            os.makedirs(save_model_radiomics_folder, exist_ok=True)
            model_radiomics.save(ckpt_dir=save_model_radiomics_folder)

            save_model_folder = os.path.join(save_folder, 'image_model')
            os.makedirs(save_model_folder, exist_ok=True)
            model.save(ckpt_dir=save_model_folder)
            train_args = {}
            for k,v in vars(args).items():
                if k != 'device':
                    train_args[k] = v
            with open(os.path.join(save_folder, 'train_args.json'), 'w') as f:
                json.dump(train_args, f, indent=4)
            val_losses['best'] = val_loss
            print(f'Saved model to {save_folder}')

        if val_contrastive_loss < val_contrastive_losses['best']:
            print(f'** Best validation contrastive loss model saved at epoch {epoch} **')
            save_folder = os.path.join(args.output_dir, f'best_val_contrastive_loss_model_epoch{epoch}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            os.makedirs(save_folder, exist_ok=True)
            save_model_radiomics_folder = os.path.join(save_folder, 'transtab')
            os.makedirs(save_model_radiomics_folder, exist_ok=True)
            model_radiomics.save(ckpt_dir=save_model_radiomics_folder)

            save_model_folder = os.path.join(save_folder, 'image_model')
            os.makedirs(save_model_folder, exist_ok=True)
            model.save(ckpt_dir=save_model_folder)
            train_args = {}
            for k,v in vars(args).items():
                if k != 'device':
                    train_args[k] = v
            with open(os.path.join(save_folder, 'train_args.json'), 'w') as f:
                json.dump(train_args, f, indent=4)
            val_contrastive_losses['best'] = val_contrastive_loss
            print(f'Saved model to {save_folder}')

        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_seg_loss': train_seg_loss,
            'val_seg_loss': val_seg_loss,
            'train_contrastive_loss': train_contrastive_loss,
            'val_contrastive_loss': val_contrastive_loss,
            'train_dice': train_dice,
            'val_dice': val_dice,
            'train_class_loss': train_class_loss,
            'val_class_loss': val_class_loss,
            'train_img_class_loss': train_img_class_loss,
            'val_img_class_loss': val_img_class_loss,
            'train_downstream_losses': train_seg_loss + train_contrastive_loss,
            'val_downstream_losses': val_seg_loss + val_contrastive_loss,
        })
            
        metrics_path = os.path.join(args.output_dir, f'metrics_{start_time}.json')
        with open(metrics_path, 'w') as f:
            json.dump(
                {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_seg_losses': train_seg_losses,
                    'val_seg_losses': val_seg_losses,
                    'train_contrastive_losses': train_contrastive_losses,
                    'val_contrastive_losses': val_contrastive_losses,
                    'train_dices': train_dices,
                    'val_dices': val_dices,
                    'train_class_losses': train_class_losses,
                    'val_class_losses': val_class_losses,
                    'train_img_class_losses': train_img_class_losses,
                    'val_img_class_losses': val_img_class_losses,
                    'train_downstream_losses': train_downstream_losses,
                    'val_downstream_losses': val_downstream_losses,
                },
                f,
                indent=4
            )

if __name__ == "__main__":
    main()