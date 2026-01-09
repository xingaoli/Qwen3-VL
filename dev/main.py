from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
import torch
import time

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "../ckpt/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     # attn_implementation="flash_attention_2",
#     device_map="auto",
# )

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "../ckpt/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 2100*28*28
# max_pixels = 1340*28*28
processor = AutoProcessor.from_pretrained("../ckpt/Qwen2-VL-2B-Instruct", min_pixels=min_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg",
            },
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg",
            },
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg",
            },
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg",
                },
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg",
            },
            {
                "type": "image",
                "image": "/home/xingao/data/nuscenes_mini/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg",
            },
            {
                "type": "text",
                "text": "Suppose you are driving, generate a description of the driving scene which includes the key factors"
"for driving planning, including the traffic conditions, weather, time of day and road conditions, indicating"
"smooth surfaces or the presence of obstacles; The description should be concise, and accurate to facilitate"
"informed decision-making."
            },
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

start_time = time.perf_counter()
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=512)



generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

end_time = time.perf_counter()
print(len(generated_ids_trimmed[0]))
print(f"速度: {len(generated_ids_trimmed[0])/(end_time - start_time):.2f} token/s")

output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# Qwen2.5-VL
# 3b: 40ms/token
# 7b: 325ms/token

# Qwen2-VL
# 2b: 29ms/token
# 7b: 322ms/token
