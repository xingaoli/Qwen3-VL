from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from qwen_vl_utils import process_vision_info
import torch
import time
import glob

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "../ckpt/Qwen2-VL-2B-Instruct",
#     torch_dtype=torch.bfloat16,
#     # attn_implementation="flash_attention_2",
#     device_map="auto",
# )

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "../ckpt/Qwen2.5-VL-3B-Instruct",
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 1340*28*28
max_pixels = 1340*28*28
processor = AutoProcessor.from_pretrained("../ckpt/Qwen2.5-VL-3B-Instruct",
                                          min_pixels=min_pixels, max_pixels=max_pixels)

image_paths = glob.glob("/home/xingao/桌面/VLM能力测试/实车路测/典型场景添加感知结果/0401_select/其它/*.jpg")
image_paths.sort()

# 定义批处理大小
batch_size = 4  # 根据显存调整


def process_batch(image_paths_batch):
    # 构建批量messages
    batch_messages = []
    for img_path in image_paths_batch:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {
                        "type": "text",
                        "text":
                            "问题：用一句话描述图片上红色边界框区域里是什么。\n"
                            "要求：回复格式是“数量词[可选]+形容词[可选] +名词”，不要输出其他的信息。\n"
                            "例如：两辆绿色的轿车。"
                    }
                ]
            }
        ]
        batch_messages.append(messages)

    # 预处理文本和图像
    texts = []
    all_images = []
    all_videos = []

    for messages in batch_messages:
        # 生成文本模板
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)
        all_images.extend(image_inputs)  # 假设每个样本返回单张图像
        # all_videos.extend(video_inputs)

    # 使用processor批量处理输入
    inputs = processor(
        text=texts,
        images=all_images if all_images else None,
        videos=all_videos if all_videos else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    return inputs


# 分批处理所有图片
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]

    # 处理当前批次
    inputs = process_batch(batch_paths)

    # 推理
    start_time = time.perf_counter()
    generated_ids = model.generate(**inputs, max_new_tokens=32)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    end_time = time.perf_counter()
    print(f"本批次耗时: {(end_time - start_time) * 1000:.2f} ms\n")

    # 打印结果
    print(f"Batch {i // batch_size + 1} 结果:")
    for j, text in enumerate(output_texts):
        print(f"图片路径: {batch_paths[j]}")
        print(f"样本 {i + j}: {text}")
        print("-" * 50)

# Qwen2.5-VL
# 3b: 40ms/token
# 7b: 325ms/token

# Qwen2-VL
# 2b: 29ms/token
# 7b: 322ms/token
