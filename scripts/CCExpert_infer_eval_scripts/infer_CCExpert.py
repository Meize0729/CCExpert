from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

import sys
import warnings

warnings.filterwarnings("ignore")
# Load model
pretrained_ckpt = "./work_dir/v1.0_7b_2expertlayer_add_sft_InitFromCptStage2Terminal"
model_name = "llava_qwen_cc"
device = "cuda"
device_map = "auto"
llava_model_args = {
    "multimodal": True,
    "overwrite_config": {
        "vocab_size": 152064 if "7b" in pretrained_ckpt.lower() else 151936
    }
}
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained_ckpt, None, model_name, device_map=device_map, attn_implementation=None, **llava_model_args)

model.eval()
model = model.to(torch.bfloat16)

# Load two images
url1 = "./levir-cc/images/test/A/test_000004.png"
url2 = "./levir-cc/images/test/B/test_000004.png"

image1 = Image.open(url1)
image2 = Image.open(url2)

images = [image1, image2]
image_tensors = process_images(images, image_processor, model.config)
image_tensors = [_image.to(dtype=torch.bfloat16, device=device) for _image in image_tensors]

# Prepare interleaved text-image input
conv_template = "qwen_1_5"
question = f"This is the Image1 <image>. This is the second Image2 <image>.\nWhat difference happened from Image1 to Image2?"

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size for image in images]

# tokenizer.batch_decode(torch.clamp(input_ids, min=0))

# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])

import pdb; pdb.set_trace()