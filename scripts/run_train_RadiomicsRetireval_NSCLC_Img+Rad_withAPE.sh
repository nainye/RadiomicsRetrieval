export OUTPUT_DIR='/workspace/RadiomicsRetrieval/results/NSCLC_Img+Rad_withAPE_ct0.07_cw1.0'
export DATA_DIR='/workspace/RadiomicsRetrieval/data/NSCLC'
export TRAIN_JSONL_FILE='/workspace/RadiomicsRetrieval/data/NSCLC/train.jsonl'
export VAL_JSONL_FILE='/workspace/RadiomicsRetrieval/data/NSCLC/val.jsonl'
export DATA_HDF5_FILE='/workspace/RadiomicsRetrieval/data/NSCLC/NSCLC_data.hdf5'

cd /workspace/RadiomicsRetrieval/source

python train_RadiomicsRetrieval_NSCLC_Img+Rad_withAPE.py \
    --train_jsonl_file=$TRAIN_JSONL_FILE \
    --val_jsonl_file=$VAL_JSONL_FILE \
    --root_dir=$DATA_DIR \
    --hdf5_file=$DATA_HDF5_FILE \
    --output_dir=$OUTPUT_DIR \
    --seed=42 \
    --batch_size=80 \
    --num_workers=6 \
    --epochs=3000 \
    --lr=1e-4 \
    --weight_decay=1e-4 \
    --device=cuda:0 \
    --contrastive_temperature=0.07 \
    --contrastive_loss_weight=1.0 \
    --cls_loss_weight=0.1 \
    --accumulate_steps=1 \
    --max_num_clicks=10 \
    --ape_drop_rate=0.3 \