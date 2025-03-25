import os
import torch
import ImageReward as RM
from tqdm import tqdm
from PIL import Image, ExifTags

def load_images_and_prompts(image_folder):
    """Load images and extract prompts from their metadata.

    Args:
        image_folder (str): Path to the folder containing images.

    Returns:
        images (list): List of image file paths.
        prompts (list): Corresponding list of prompts extracted from images.
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
                        print(f"No prompt found in image metadata: {filename}")
                else:
                    print(f"No Exif metadata found in image: {filename}")
            except Exception as e:
                print(f"Error processing image {filename}: {e}")
    return images, prompts

def process_batch(images_batch, prompts_batch, model, device):
    """Process a batch of images and prompts to compute scores.

    Args:
        images_batch (list): List of image file paths in the batch.
        prompts_batch (list): Corresponding list of prompts.
        model: ImageReward model instance.
        device: PyTorch device.

    Returns:
        scores (list): List of computed scores for the batch.
    """
    scores = []
    for prompt, img_path in zip(prompts_batch, images_batch):
        with torch.no_grad():
            score = model.score(prompt, img_path)
            scores.append(score)
    return scores

def main(image_folder, batch_size, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")

    # Load model
    model = RM.load(name=model_name,download_root="/root/autodl-tmp/pretrained_models/THUDM/ImageReward").to(device)
    print(f"Loaded model '{model_name}' on {device}")
    print(f"image_folder: {image_folder}")

    # Load images and prompts from metadata
    images, prompts = load_images_and_prompts(image_folder)
    # print(prompts)

    # Check if we have images to process
    if not images:
        print("No images with prompts found in the specified folder.")
        return
    assert len(images) == 200
    # Process batches
    total_batches = (len(images) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Processing batches")

    all_scores = []
    for i in range(0, len(images), batch_size):
        images_batch = images[i:i + batch_size]
        prompts_batch = prompts[i:i + batch_size]
        scores = process_batch(images_batch, prompts_batch, model, device)
        all_scores.extend(scores)
        progress_bar.update(1)
    progress_bar.close()

    # Compute average score
    average_score = sum(all_scores) / len(all_scores)
    print(f"Final average score: {average_score}")
    with open("image_reward_scores.txt", "a") as f:
        f.write(f"{image_folder.split('/')[-1]}: {average_score}\n")


def multi_folder(root_folder, batch_size, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available!")
    image_folders = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    # Load model
    model = RM.load(name=model_name,download_root="/root/autodl-tmp/pretrained_models/THUDM/ImageReward").to(device)
    print(f"Loaded model '{model_name}' on {device}")
    for image_folder in image_folders:
        print(f"image_folder: {image_folder}")

        # Load images and prompts from metadata
        images, prompts = load_images_and_prompts(image_folder)
        # print(prompts)

        # Check if we have images to process
        if not images:
            print("No images with prompts found in the specified folder.")
            return
        if len(images) != 200:
            print(f"image_folder: {image_folder} has {len(images)} images")
            continue
        
        # Process batches
        total_batches = (len(images) + batch_size - 1) // batch_size
        progress_bar = tqdm(total=total_batches, desc="Processing batches")

        all_scores = []
        for i in range(0, len(images), batch_size):
            images_batch = images[i:i + batch_size]
            prompts_batch = prompts[i:i + batch_size]
            scores = process_batch(images_batch, prompts_batch, model, device)
            all_scores.extend(scores)
            progress_bar.update(1)
        progress_bar.close()

        # Compute average score
        average_score = sum(all_scores) / len(all_scores)
        print(f"Final average score: {average_score}")
        with open("image_reward_scores.txt", "a") as f:
            f.write(f"{image_folder.split('/')[-1]}: {average_score}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Single-GPU ImageReward scoring script.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing.")
    parser.add_argument("--model_name", type=str, default="ImageReward-v1.0", help="ImageReward model name.")
    parser.add_argument("--root_folder", type=str, default="", help="Root folder containing image folders.")

    args = parser.parse_args()
    if args.root_folder:
        multi_folder(root_folder=args.root_folder, batch_size=args.batch_size, model_name=args.model_name)
    else:
        main(image_folder=args.image_folder, batch_size=args.batch_size, model_name=args.model_name)
