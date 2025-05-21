import os


def add_index_to_filenames(folder_path):
    # 定义支持的图片扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')

    # 获取文件夹下所有图片文件
    all_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith(image_extensions)
                 and os.path.isfile(os.path.join(folder_path, f))]

    # 按文件名排序（区分大小写）
    sorted_files = sorted(all_files)

    # 遍历并重命名文件
    for index, filename in enumerate(sorted_files):
        # 生成四位数字前缀
        new_name = f"{index:04d}_{filename}"

        # 构造完整文件路径
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)

        # 执行重命名操作
        os.rename(src, dst)
        print(f"已重命名：{filename} -> {new_name}")


if __name__ == "__main__":
    target_folder = "/home/xingao/桌面/VLM能力测试/实车路测/典型场景添加感知结果/0401_select/其它"  # 替换为你的实际路径
    add_index_to_filenames(target_folder)
    print("\n所有文件重命名完成！")