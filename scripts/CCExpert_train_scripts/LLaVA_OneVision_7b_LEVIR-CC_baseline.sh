# >>>>>>>>>>>>>>>>>>>>>>>>>> Multi GPU hyper <<<<<<<<<<<<<<<<<<<<<<<<<<<
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

NUM_GPUS=8  # !!!! You must fill your gpu num !!!!
NNODES=1
RANK=0
ADDR="localhost"
PORT=13333

PROJECT_NAME="CCExpert_Github_by_Meize0729"
WORK_DIR="/mnt/bn/chenhaobo-va-data/wangmingze/work_dir/RS3"
REPORT_TO="none"  # If you have wandb, you can use "wandb"

# >>>>>>>>>>>>>>>>>>>>>>>>>> SFT Run <<<<<<<<<<<<<<<<<<<<<<<<<<<
SFT_RUN_NAME="LLaVA_OneVision_7b_LEVIR-CC_baseline"
# If you cannot connect to Huggingface on your machine, you can move to the following link to download.
# LLaVA-OneVision-7b: https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov Download this and fill this part by local absolute path
# Siglip-so400m-patch14-384: https://huggingface.co/google/siglip-so400m-patch14-384 Download this and fill this part by local absolute path
BASE_CKPT_PATH_OR_NAME="lmms-lab/llava-onevision-qwen2-7b-ov"
VISION_TOWER_CKPT_PATH_OR_NAME="google/siglip-so400m-patch14-384"

# Total Batchsize = per_device_train_batch_size x GPU_num x gradient_accumulation_steps = 2 x 8 x 16 = 256.
# Test relevant metrics once per epoch and save the ckpt with the best 'Sm' performance.
# Base LR: 1e-5, ViT LR: 2e-6.
# We use BF16. If you do not have NVIDIA A/H-100/800, you can choose FP16.
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_eval.py \
    --deepspeed "./scripts/zero2.json" \
    --model_name_or_path ${BASE_CKPT_PATH_OR_NAME} \
    --version "qwen_1_5" \
    --data_path "./scripts/CCExpert_data_scripts/benchmark_LEVIR-CC_train_absolute_path.yaml" \
    --eval_data_path "./scripts/CCExpert_data_scripts/benchmark_LEVIR-CC_test_absolute_path.yaml" \
    --loop_format "random_choice" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model,mm_vision_tower" \
    --vision_tower "$VISION_TOWER_CKPT_PATH_OR_NAME" \
    --mm_vision_tower_lr=2e-6 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length False \
    --image_aspect_ratio "square" \
    --mm_patch_merge_type "flat" \
    --bf16 True \
    --project_name "$PROJECT_NAME" \
    --run_name $SFT_RUN_NAME \
    --output_dir "$WORK_DIR/$PROJECT_NAME/$SFT_RUN_NAME" \
    --num_train_epochs 50 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "epoch" \
    --eval_steps 1 \
    --metric_for_best_model "Sm" \
    --greater_is_better True \
    --save_strategy "epoch" \
    --save_steps 1 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to "$REPORT_TO" \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last False \
    --frames_upbound 32 \
    --seed 42