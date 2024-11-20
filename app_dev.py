import argparse
import os
from datetime import datetime

import gradio as gr
import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from model.cloth_masker import AutoMasker, vis_mask
from model.pipeline import CatVTONPipeline
from utils import init_weight_dtype, resize_and_crop, resize_and_padding

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="booksforcharlie/stable-diffusion-inpainting",  # Change to a copy repo as runawayml delete original repo
        help=(
            "The path to the base model to use for evaluation. This can be a local path or a model identifier from the Model Hub."
        ),
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default="zhengchong/CatVTON",
        help=(
            "The Path to the checkpoint of trained tryon model."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="resource/demo/output",
        help="The output directory where the model predictions will be written.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--repaint", 
        action="store_true", 
        help="Whether to repaint the result image with the original background."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        default=True,
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


args = parse_args()
repo_path = snapshot_download(repo_id=args.resume_path)
# Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=args.base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(args.mixed_precision),
    use_tf32=args.allow_tf32,
    device='cuda'
)
# AutoMasker
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

def preprocess_person_images(image_paths, target_size):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = resize_and_crop(image, target_size)
        images.append(image)
    return images

def preprocess_cloth_images(image_paths, target_size):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = resize_and_padding(image, target_size)
        images.append(image)
    return images

def generate_masks(person_images, cloth_type):
    masks = []
    for person_image in person_images:
        mask = automasker(
            person_image,
            cloth_type
        )['mask']
        mask = mask_processor.blur(mask, blur_factor=9)
        masks.append(mask)
    return masks

def images_to_tensor(images):
    tensors = []
    for image in images:
        # arr = np.array(image).astype(np.float32) / 255.0  # H x W x C or H x W
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0  # H x W x C or H x W
        if arr.ndim == 2:
            # Grayscale image, add channel dimension
            arr = arr[:, :, np.newaxis]  # Now arr.shape is (H, W, 1)
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # Now tensor has shape C x H x W
        tensors.append(tensor)
    return torch.stack(tensors)

def submit_function(
    person_files,
    cloth_files,
    person_examples_selected,
    condition_examples_selected,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type
):

    root_path = "resource/demo/example"
    # Combine uploaded and selected person images
    person_files = person_files or []
    if person_examples_selected:
        selected_person_paths = [os.path.join(root_path, "person", "men", label)
                                 if os.path.exists(os.path.join(root_path, "person", "men", label))
                                 else os.path.join(root_path, "person", "women", label)
                                 for label in person_examples_selected]
        person_files += selected_person_paths

    # Combine uploaded and selected cloth images
    cloth_files = cloth_files or []
    if condition_examples_selected:
        selected_condition_paths = [os.path.join(root_path, "condition", "upper", label)
                                    if os.path.exists(os.path.join(root_path, "condition", "upper", label))
                                    else os.path.join(root_path, "condition", "overall", label)
                                    if os.path.exists(os.path.join(root_path, "condition", "overall", label))
                                    else os.path.join(root_path, "condition", "person", label)
                                    for label in condition_examples_selected]
        cloth_files += selected_condition_paths

    MAX_IMAGES = 10
    person_images_paths = person_files[:MAX_IMAGES]
    cloth_images_paths = cloth_files[:MAX_IMAGES]

    # Preprocess person images
    person_images = preprocess_person_images(person_images_paths, (args.width, args.height))

    # Preprocess cloth images
    cloth_images = preprocess_cloth_images(cloth_images_paths, (args.width, args.height))

    # Generate masks for person images
    masks = generate_masks(person_images, cloth_type)

    # Generate combinations of person images and cloth images
    combinations = []
    for person_image, mask in zip(person_images, masks):
        for cloth_image in cloth_images:
            combinations.append((person_image, cloth_image, mask))

    batch_size = args.batch_size if hasattr(args, 'batch_size') else 4
    results = []
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        person_batch = [item[0] for item in batch]
        cloth_batch = [item[1] for item in batch]
        mask_batch = [item[2] for item in batch]

        # Convert images and masks to tensors
        person_tensors = images_to_tensor(person_batch).to(pipeline.device)
        cloth_tensors = images_to_tensor(cloth_batch).to(pipeline.device)
        mask_tensors = images_to_tensor(mask_batch).to(pipeline.device)

        print(f'person shape: {person_tensors.shape}')
        print(f'cloth shape: {cloth_tensors.shape}')
        print(f'mask shape: {mask_tensors.shape}')

        # Set random seed
        generator = None
        if seed != -1:
            generator = torch.Generator(device='cuda').manual_seed(seed)

        # Inference
        result_images = pipeline(
            person_tensors,
            cloth_tensors,
            mask_tensors,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )

        # Post-process results
        for idx, result_image in enumerate(result_images):
            person_image = person_batch[idx]
            cloth_image = cloth_batch[idx]
            mask = mask_batch[idx]

            if show_type == "result only":
                display_image = result_image
            else:
                width, height = person_image.size
                if show_type == "input & result":
                    condition_width = width // 2
                    conditions = image_grid([person_image, cloth_image], 2, 1)
                else:
                    masked_person = vis_mask(person_image, mask)
                    condition_width = width // 3
                    conditions = image_grid([person_image, masked_person , cloth_image], 3, 1)
                conditions = conditions.resize((condition_width, height), Image.NEAREST)
                new_result_image = Image.new("RGB", (width + condition_width + 5, height))
                new_result_image.paste(conditions, (0, 0))
                new_result_image.paste(result_image, (condition_width + 5, 0))
                display_image = new_result_image

            # Append result
            results.append(display_image)

    return results



def person_example_fn(image_path):
    return image_path

HEADER = """
<h1 style="text-align: center;"> 🐈 CatVTON: Concatenation Is All You Need for Virtual Try-On with Diffusion Models </h1>
<div style="display: flex; justify-content: center; align-items: center;">
  <a href="http://arxiv.org/abs/2407.15886" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/arXiv-2407.15886-red?style=flat&logo=arXiv&logoColor=red' alt='arxiv'>
  </a>
  <a href='https://huggingface.co/zhengchong/CatVTON' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Hugging Face-ckpts-orange?style=flat&logo=HuggingFace&logoColor=orange' alt='huggingface'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/GitHub-Repo-blue?style=flat&logo=GitHub' alt='GitHub'>
  </a>
  <a href="http://120.76.142.206:8888" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Demo-Gradio-gold?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href="https://huggingface.co/spaces/zhengchong/CatVTON" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Space-ZeroGPU-orange?style=flat&logo=Gradio&logoColor=red' alt='Demo'>
  </a>
  <a href='https://zheng-chong.github.io/CatVTON/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://github.com/Zheng-Chong/CatVTON/LICENCE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC--SA--4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>
<br>
· This demo and our weights are only for <span>Non-commercial Use</span>. <br>
· You can try CatVTON in our <a href="https://huggingface.co/spaces/zhengchong/CatVTON">HuggingFace Space</a> or our <a href="http://120.76.142.206:8888">online demo</a> (run on 3090). <br>
· Thanks to <a href="https://huggingface.co/zero-gpu-explorers">ZeroGPU</a> for providing A100 for our <a href="https://huggingface.co/spaces/zhengchong/CatVTON">HuggingFace Space</a>. <br>
· SafetyChecker is set to filter NSFW content, but it may block normal results too. Please adjust the <span>`seed`</span> for normal outcomes.<br> 
"""

def app_gradio():
    with gr.Blocks(title="CatVTON") as demo:
        gr.Markdown(HEADER)
        
        with gr.Row():
            person_images = gr.File(
                label="Person Images",
                type="filepath",
                file_types=['image'],
                interactive=True,
                file_count="multiple",
            )
        with gr.Row():
            with gr.Column(scale=1, min_width=230):
                cloth_images = gr.File(
                    label="Condition Images",
                    type="filepath",
                    file_types=['image'],
                    interactive=True,
                    file_count="multiple",
                )
            with gr.Column(scale=1, min_width=120):
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">Masks are generated automatically based on the selected `Try-On Cloth Type`.</span>'
                )
                cloth_type_batch = gr.Radio(
                    label="Try-On Cloth Type",
                    choices=["upper", "lower", "overall"],
                    value="upper",
                )
        submit = gr.Button("Submit")
                
        # Shared Advanced Options
        gr.Markdown(
            '<center><span style="color: #FF0000">!!! Click only Once, Wait for Delay !!!</span></center>'
        )
        
        gr.Markdown(
            '<span style="color: #808080; font-size: small;">Advanced options can adjust details:<br>1. `Inference Step` may enhance details;<br>2. `CFG` is highly correlated with saturation;<br>3. `Random seed` may improve pseudo-shadow.</span>'
        )
        with gr.Accordion("Advanced Options", open=False):
            num_inference_steps = gr.Slider(
                label="Inference Step", minimum=10, maximum=100, step=5, value=50
            )
            guidance_scale = gr.Slider(
                label="CFG Strength", minimum=0.0, maximum=7.5, step=0.5, value=2.5
            )
            seed = gr.Slider(
                label="Seed", minimum=-1, maximum=10000, step=1, value=42
            )
            show_type = gr.Radio(
                label="Show Type",
                choices=["result only", "input & result", "input & mask & result"],
                value="input & mask & result",
            )

        # Use Gallery to display multiple results
        result_gallery = gr.Gallery(label="Results", columns=3, height='auto')
        with gr.Row():
            # Photo Examples
            root_path = "resource/demo/example"
            with gr.Column():
                # Person Examples Gallery
                person_example_paths = [
                    os.path.join(root_path, "person", "men", filename)
                    for filename in os.listdir(os.path.join(root_path, "person", "men"))
                ] + [
                    os.path.join(root_path, "person", "women", filename)
                    for filename in os.listdir(os.path.join(root_path, "person", "women"))
                ]
                person_example_labels = [os.path.basename(path) for path in person_example_paths]

                # Display images in a gallery (non-interactive)
                gr.Markdown("#### Person Examples:")
                gr.Gallery(value=person_example_paths, columns=4, height='auto')

                # CheckboxGroup for selection
                person_examples = gr.CheckboxGroup(
                    label="Select Person Examples",
                    choices=person_example_labels,
                )
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">*Person examples come from the demos of ...</span>'
                )
            with gr.Column():
                # Condition Examples Gallery
                condition_example_paths = [
                    os.path.join(root_path, "condition", "upper", filename)
                    for filename in os.listdir(os.path.join(root_path, "condition", "upper"))
                ] + [
                    os.path.join(root_path, "condition", "overall", filename)
                    for filename in os.listdir(os.path.join(root_path, "condition", "overall"))
                ] + [
                    os.path.join(root_path, "condition", "person", filename)
                    for filename in os.listdir(os.path.join(root_path, "condition", "person"))
                ]
                condition_example_labels = [os.path.basename(path) for path in condition_example_paths]

                # Display images in a gallery (non-interactive)
                gr.Markdown("#### Condition Examples:")
                gr.Gallery(value=condition_example_paths, columns=4, height='auto')

                # CheckboxGroup for selection
                condition_examples = gr.CheckboxGroup(
                    label="Select Condition Examples",
                    choices=condition_example_labels,
                )
                gr.Markdown(
                    '<span style="color: #808080; font-size: small;">*Condition examples come from the Internet. </span>'
                )
        
        # Connect submit button to function
        
        submit.click(
            submit_function,
            inputs=[
                person_images,
                cloth_images,
                person_examples,
                condition_examples,
                cloth_type_batch,
                num_inference_steps,
                guidance_scale,
                seed,
                show_type,
            ],
            outputs=result_gallery,
        )
    
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
