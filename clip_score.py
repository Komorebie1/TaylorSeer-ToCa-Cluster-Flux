import os
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torchmetrics.multimodal.clip_score import CLIPScore

def load_images_and_prompts(image_folder):
    """
    加载图像，并从其 Exif 元数据中提取提示（ImageDescription 字段）。
    
    Args:
        image_folder (str): 图像所在文件夹路径。
    
    Returns:
        images (list): 图像文件路径列表。
        prompts (list): 与图像对应的提示文本列表。
    """
    images = []
    prompts = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, filename)
            try:
                img = Image.open(img_path)
                exif_data = img._getexif()
                if exif_data is not None:
                    exif = {}
                    for tag_id, value in exif_data.items():
                        tag = ExifTags.TAGS.get(tag_id, tag_id)
                        exif[tag] = value
                    prompt = exif.get('ImageDescription', None)
                    if prompt:
                        images.append(img_path)
                        prompts.append(prompt)
                    else:
                        print(f"图像 {filename} 中未找到提示信息。")
                else:
                    print(f"图像 {filename} 中无 Exif 元数据。")
            except Exception as e:
                print(f"处理图像 {filename} 时出错：{e}")
    return images, prompts

def process_batch(images_batch, prompts_batch, model_path, device):
    """
    对一个批次的图像和提示计算 CLIP Score。
    
    Args:
        images_batch (list): 当前批次图像文件路径列表。
        prompts_batch (list): 与图像对应的提示列表。
        model_path (str): CLIP 模型路径或名称。
        device: PyTorch 设备（GPU）。
    
    Returns:
        (batch_avg, count): 当前批次的平均得分以及批次中成功处理的图像数量。
    """
    # 初始化 CLIPScore 度量模型
    clip_score_metric = CLIPScore(model_name_or_path=model_path).to(device)
    transform = ToTensor()
    image_tensors = []
    for img_path in images_batch:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img)
            image_tensors.append(tensor)
        except Exception as e:
            print(f"加载图像 {img_path} 时出错：{e}")
    if len(image_tensors) == 0:
        return None, 0
    images_tensor = torch.stack(image_tensors, dim=0).to(device)
    
    with torch.no_grad():
        # 将整个批次的图像和对应的文本传入模型，计算批次平均得分
        clip_score_metric.update(images_tensor.float(), prompts_batch)
        batch_avg = clip_score_metric.compute().item()
    return batch_avg, len(image_tensors)

def main(image_folder, batch_size, model_path):
    """
    主函数：加载图像和提示，分批计算 CLIP Score 并输出全局平均得分。
    
    Args:
        image_folder (str): 图像所在文件夹路径。
        batch_size (int): 批处理大小。
        model_path (str): CLIP 模型路径或名称。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU！")
    print(f"当前使用设备：{device}")
    
    # 加载图像及对应提示
    images, prompts = load_images_and_prompts(image_folder)
    if not images:
        print("在指定文件夹中未找到包含提示的图像。")
        return
    
    total_batches = (len(images) + batch_size - 1) // batch_size
    total_score = 0.0
    total_count = 0
    progress_bar = tqdm(total=total_batches, desc="Processing batches")
    
    # 分批处理
    for i in range(0, len(images), batch_size):
        images_batch = images[i:i+batch_size]
        prompts_batch = prompts[i:i+batch_size]
        batch_avg, count = process_batch(images_batch, prompts_batch, model_path, device)
        if count > 0:
            # 累加本批次总得分（平均得分 * 图像数量）
            total_score += batch_avg * count
            total_count += count
        progress_bar.update(1)
    progress_bar.close()
    
    if total_count > 0:
        overall_avg = total_score / total_count
        print(f"文件夹中所有图像的最终平均 CLIP Score 为：{overall_avg}")
        f = open("clip_score.txt", "a")
        f.write(f"{image_folder}: {overall_avg}\n")
        f.close()
    else:
        print("没有有效的图像被处理。")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="基于图像元数据中提示的 CLIP Score 测评脚本。")
    parser.add_argument("--image_folder", type=str, required=True, help="图像文件夹的路径。")
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小。")
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/pretrained_models/laion/CLIP-ViT-g-14-laion2B-s12B-b42K",
                        help="CLIP 模型的路径或名称。")
    
    args = parser.parse_args()
    main(image_folder=args.image_folder, batch_size=args.batch_size, model_path=args.model_path)
