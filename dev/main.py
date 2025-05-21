from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

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

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../ckpt/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 1340*28*28
# max_pixels = 1340*28*28
processor = AutoProcessor.from_pretrained("../ckpt/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/home/xingao/桌面/VLM能力测试/实车路测/03典型场景添加感知结果/0408/003point/图片/0001_红棉立交东1.jpg",
            },
            {
                "type": "text",
                "text": "这张图片的尺寸为1920*1080，以左上角为像素坐标系的原点。\n"
                "请输出坐标点[1616, 722]附近有什么特征。"
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
generated_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
end_time = time.perf_counter()
print(output_text)
print(f"耗时: {(end_time - start_time) * 1000:.2f} ms")
# Qwen2.5-VL
# 3b: 40ms/token
# 7b: 325ms/token

# Qwen2-VL
# 2b: 29ms/token
# 7b: 322ms/token
