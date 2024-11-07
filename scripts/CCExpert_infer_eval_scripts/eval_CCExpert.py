from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import json
import os
import torch
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp
from time import time
import torch.distributed as dist
from transformers import set_seed

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import sys
import warnings

from llava import conversation as conversation_lib
from llava.dataset.cc_dataset import LazySupervisedDataset_CC, DataCollatorForSupervisedDataset
from llava.train.train_eval import DataArguments
from llava.train.llava_trainer_eval import get_eval_score

os.environ["NCCL_DEBUG"] = ""
precision_dict = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def setup(args, rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.DDP_port
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
    )

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def inference(rank, world_size, args):
    set_seed(42)

    this_rank_gpu_index = rank

    if args.DDP:
        torch.cuda.set_device(this_rank_gpu_index)
        setup(args, rank, world_size)

    disable_torch_init()

    device = torch.device("cuda:" + str(this_rank_gpu_index) if torch.cuda.is_available() else "cpu")
    
    overwrite_config = {}

    llava_model_args = {
        "multimodal": True,
        "overwrite_config": {
            "vocab_size": 152064 if "7b" in args.model_path.lower() else 151936
        }
    }
    
    tokenizer, model, image_processor, max_length = load_pretrained_model(args.model_path, None, args.model_name, device_map=None, attn_implementation=None, **llava_model_args)
    model.eval()
    model = model.to(device).to(precision_dict[args.precision])

    data_args = DataArguments(
        data_path = args.data_path,
        image_aspect_ratio = args.image_aspect_ratio,
        is_multimodal = True,
    )
    data_args.image_processor = image_processor
    data_args.mm_patch_merge_type = model.config.mm_patch_merge_type
    data_args.mm_use_im_patch_token = model.config.mm_use_im_patch_token
    data_args.mm_use_im_start_end = model.config.mm_use_im_start_end

    conversation_lib.default_conversation = conversation_lib.conv_templates[getattr(args, "prompt_version", "qwen_1_5")]
    dataset = LazySupervisedDataset_CC(args.data_path, tokenizer=tokenizer, data_args=data_args, is_eval=True)
    
    if args.DDP:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            pin_memory=True,
            sampler=sampler,
        )
        rf = open(args.out_path + "/" + "result.worker_" + str(rank), "w")
    else:
        dataloader = DataLoader(
            dataset,
            num_workers=1,
            batch_size=1,
            collate_fn=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
            pin_memory=True,
        )
        rf = open(args.out_path + "/" + "result", "w")

    prog_bar = tqdm(dataloader, total=len(dataloader), desc="worker_" + str(rank)) if rank == 0 else dataloader
    """
    There are actually some problems here. I didn't deliberately modify the dataset. 
    It is still consistent with training. Then I take out a series of tokens in front of "assistant" for inference. It will be modified in the future.
    """
    for test_batch in prog_bar:
        data_id = test_batch.pop("uid") if "uid" in test_batch.keys() else None
        input_ids = test_batch["input_ids"].to(device)
        attention_mask = test_batch["attention_mask"].to(device)
        extra_info = test_batch.pop("extra_info") if "extra_info" in test_batch.keys() else ""
        
        images = [img.to(precision_dict[args.precision]).to(device) for img in test_batch["images"]]
        assistant_index = (input_ids[0] == 77091).nonzero(as_tuple=True)[0][-1].item()
        input_ids = input_ids[:, :assistant_index+2]
        with torch.inference_mode():
            output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=test_batch['image_sizes'],
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )
        text_outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        results = dict(
            uid = data_id[0],
            pred = text_outputs,
            change_caption = extra_info[0]["change_caption"],
        )
        rf.write(json.dumps(results))
        rf.write("\n")
    
    rf.close()

def gather_result(args, world_size):
    num_worker = world_size
    with open(args.out_path + "/result", "w") as f:
        for i in range(num_worker):
            with open(args.out_path + "/" + "result.worker_" + str(i), "r") as tf:
                tmp_result = tf.readlines()
            f.writelines(tmp_result)
            os.remove(args.out_path + "/" + "result.worker_" + str(i))

def eval_metrics(args):
    with open(args.out_path + "/result", "r") as f:
        results = f.readlines()
    results = [json.loads(result) for result in results]
    all_gts, all_preds = [], []
    for result in results:
        all_gts.append(result["change_caption"])
        all_preds.append([result["pred"]])

    def process_text_list(text_list):
        processed_list = []
        for sublist in text_list:
            processed_sublist = []
            for sentence in sublist:
                # 将字符串转换为小写并去除句号
                sentence = sentence.lower().replace('.', '')
                processed_sublist.append(sentence)
            processed_list.append(processed_sublist)
        return processed_list

    score_metrics = get_eval_score(process_text_list(all_gts), process_text_list(all_preds))
    # 存到路径下txt，指标
    with open(args.out_path + "/score_metrics.txt", "w") as f:
        f.write(json.dumps(score_metrics))
    print(score_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llava_qwen_cc",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./work_dir/v1.0_7b_2expertlayer_add_sft_InitFromCptStage2Terminal",
    )
    parser.add_argument("--data_path", type=str, required=False, default="./scripts/CCExpert_data_scripts/benchmark_LEVIR-CC_test_absolute_path.yaml")
    parser.add_argument(
        "--out_path",
        type=str,
        required=False,
        default="./work_dir/v1.0_7b_2expertlayer_add_sft_InitFromCptStage2Terminal",
    )
    parser.add_argument("--max_new_tokens", type=int, required=False, default=512)
    parser.add_argument("--prompt_version", type=str, default="qwen_1_5")
    parser.add_argument("--image_aspect_ratio", type=str, default="square")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--DDP_port", default="19999")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--precision", type=str, default="fp32")
    args = parser.parse_args()
    print(args)

    start = time()
    if args.DDP:
        mp.spawn(inference, args=(args.world_size, args), nprocs=args.world_size)
        gather_result(args, args.world_size)
        eval_metrics(args)
    else:
        inference(0, args.world_size, args)
        eval_metrics(args)
    end = time()
    print(f"Inference took {end - start:8.4f} seconds.")

'''
python3 ./scripts/CCExpert_infer_eval_scripts/eval_CCExpert.py \
        --model_name "llava_qwen_cc" \
        --model_path "./work_dir/v1.0_7b_2expertlayer_add_sft_InitFromCptStage2Terminal" \
        --out_path "./work_dir/v1.0_7b_2expertlayer_add_sft_InitFromCptStage2Terminal" \
        --DDP \
        --world_size 8
'''

# python3 valley/inference/inference_valley.py --world_size=8 --DDP
