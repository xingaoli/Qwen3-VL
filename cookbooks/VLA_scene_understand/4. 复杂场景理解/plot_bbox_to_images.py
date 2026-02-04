import cv2
import numpy as np
from typing import List, Tuple
import argparse

def normalize_to_pixel_coords(
    normalized_bboxes: List[Tuple[float, float, float, float]],
    img_width: int,
    img_height: int
) -> List[Tuple[int, int, int, int]]:
    """
    将归一化坐标转换为像素坐标
    
    Args:
        normalized_bboxes: 归一化边界框列表 [(xmin, ymin, xmax, ymax), ...]
        img_width: 图片实际宽度（像素）
        img_height: 图片实际高度（像素）
    
    Returns:
        像素坐标边界框列表
    """
    pixel_bboxes = []
    for bbox in normalized_bboxes:
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = bbox
        
        # 将归一化坐标转换为像素坐标
        xmin_pixel = int(xmin_norm/1000 * img_width)
        ymin_pixel = int(ymin_norm/1000 * img_height)
        xmax_pixel = int(xmax_norm/1000 * img_width)
        ymax_pixel = int(ymax_norm/1000 * img_height)
        
        pixel_bboxes.append((xmin_pixel, ymin_pixel, xmax_pixel, ymax_pixel))
    
    return pixel_bboxes

def draw_bounding_boxes(
    image_path: str,
    normalized_bboxes: List[Tuple[float, float, float, float]],
    colors: List[Tuple[int, int, int]] = None,
    thickness: int = 2,
    show_text: bool = True
) -> np.ndarray:
    """
    在图片上绘制边界框
    
    Args:
        image_path: 图片路径
        normalized_bboxes: 归一化边界框列表
        colors: 每个边界框的颜色列表 (B, G, R)，如果为None则自动生成
        thickness: 边界框线宽
        show_text: 是否显示框的编号
    
    Returns:
        绘制了边界框的图片
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = image.shape[:2]
    print(f"图片尺寸: {width} x {height}")
    
    # 转换坐标
    pixel_bboxes = normalize_to_pixel_coords(normalized_bboxes, width, height)
    
    # 如果没有指定颜色，自动生成不同的颜色
    if colors is None:
        colors = []
        for i in range(len(pixel_bboxes)):
            # 生成不同的颜色（HSV空间均匀分布）
            hue = (i * 180 // len(pixel_bboxes))  # 0-180之间（OpenCV的HSV范围）
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
    
    # 绘制每个边界框
    for idx, (xmin, ymin, xmax, ymax) in enumerate(pixel_bboxes):
        color = colors[idx % len(colors)]
        
        # 绘制矩形框
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
        
        # 如果需要，在框的左上角显示编号
        if show_text:
            text = f"Box {idx+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            
            # 计算文字大小
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, text_thickness
            )
            
            # 绘制文字背景（小矩形）
            cv2.rectangle(
                image,
                (xmin, ymin - text_height - 5),
                (xmin + text_width, ymin),
                color,
                -1  # 填充
            )
            
            # 绘制文字
            cv2.putText(
                image,
                text,
                (xmin, ymin - 5),
                font,
                font_scale,
                (0, 0, 0),  # 白色文字
                text_thickness
            )
        
        # 打印坐标信息
        print(f"框 {idx+1}: 归一化 [{normalized_bboxes[idx]}] -> "
              f"像素 [{xmin}, {ymin}, {xmax}, {ymax}]")
    
    return image

def display_image(image: np.ndarray, window_name: str = "Bounding Boxes"):
    """
    显示图片
    
    Args:
        image: 要显示的图片
        window_name: 窗口名称
    """
    # 如果图片太大，按比例缩小显示
    screen_height = 800
    screen_width = 1200
    
    h, w = image.shape[:2]
    scale = min(screen_width / w, screen_height / h)
    
    if scale < 1:
        display_w = int(w * scale)
        display_h = int(h * scale)
        display_image = cv2.resize(image, (display_w, display_h))
    else:
        display_image = image
    
    cv2.imshow(window_name, display_image)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_image(image: np.ndarray, output_path: str = "output_with_boxes.jpg"):
    """
    保存图片
    
    Args:
        image: 要保存的图片
        output_path: 保存路径
    """
    cv2.imwrite(output_path, image)
    print(f"图片已保存到: {output_path}")

# 主函数
def main():
    parser = argparse.ArgumentParser(description='在图片上绘制边界框')
    parser.add_argument('image_path', type=str, help='图片路径')
    parser.add_argument('--bboxes', type=str, 
                       default='0.1,0.1,0.3,0.3;0.4,0.4,0.6,0.6;0.7,0.2,0.9,0.5',
                       help='边界框坐标，格式: xmin,ymin,xmax,ymax;xmin2,ymin2,...')
    parser.add_argument('--output', type=str, default='output.jpg',
                       help='输出图片路径')
    parser.add_argument('--no-show', action='store_true',
                       help='不显示图片，只保存')
    
    args = parser.parse_args()
    
    # 解析边界框字符串
    bbox_strs = args.bboxes.split(';')
    normalized_bboxes = []
    
    for bbox_str in bbox_strs:
        coords = list(map(float, bbox_str.split(',')))
        if len(coords) != 4:
            raise ValueError(f"边界框格式错误: {bbox_str}，应为 xmin,ymin,xmax,ymax")
        normalized_bboxes.append(tuple(coords))
    
    print(f"读取到 {len(normalized_bboxes)} 个边界框")
    
    try:
        # 绘制边界框
        result_image = draw_bounding_boxes(
            args.image_path,
            normalized_bboxes,
            show_text=True
        )
        
        # 保存图片
        # save_image(result_image, args.output)
        
        # 显示图片（除非指定不显示）
        if not args.no_show:
            display_image(result_image)
            
    except Exception as e:
        print(f"错误: {e}")

# 直接使用示例
if __name__ == "__main__":
    main()
