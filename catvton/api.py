import os

import torch
from diffusers.image_processor import VaeImageProcessor
from huggingface_hub import snapshot_download
from PIL import Image

from catvton.model.cloth_masker import AutoMasker, vis_mask
from catvton.model.pipeline import CatVTONPipeline
from catvton.utils import init_weight_dtype
from catvton.preprocessor.image_processor import ImageProcessor
from typing import List, Literal
from pydantic import BaseModel
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse


# FIXME
base_model_path="booksforcharlie/stable-diffusion-inpainting"
output_dir="resource/demo/output"
allow_tf32=True
mixed_precision="bf16" #["no", "fp16", "bf16"]
resume_path="zhengchong/CatVTON"
repo_path = snapshot_download(repo_id=resume_path)


# Inference Pipeline
pipeline = CatVTONPipeline(
    base_ckpt=base_model_path,
    attn_ckpt=repo_path,
    attn_ckpt_version="mix",
    weight_dtype=init_weight_dtype(mixed_precision),
    use_tf32=allow_tf32,
    device='cuda'
)
mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)
automasker = AutoMasker(
    densepose_ckpt=os.path.join(repo_path, "DensePose"),
    schp_ckpt=os.path.join(repo_path, "SCHP"),
    device='cuda', 
)

def image_to_base64(img: Image.Image) -> str:
    buffer = BytesIO()
    img.save(buffer, format="PNG")  # Convert to PNG format
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


async def inference(
    person_files: List[str],
    cloth_files: List[str],
    cloth_type: Literal["upper", "lower", "overall"]="upper",
    num_inference_steps: int=10,
    guidance_scale: float=2.5,
    seed: int =42,
    show_type: Literal["result only", "input & result", "input & mask & result"]="input & mask & result",
    batch_size: int=4,
    target_width: int=768,
    target_height: int=1024
):
    import torch
    
    MAX_IMAGES = 10
    person_images_paths = person_files[:MAX_IMAGES]
    cloth_images_paths = cloth_files[:MAX_IMAGES]

    # Preprocess person images
    person_images = ImageProcessor.preprocess_person_images(person_images_paths, (target_width, target_height))

    # Preprocess cloth images
    cloth_images = ImageProcessor.preprocess_cloth_images(cloth_images_paths, (target_width, target_height))

    # Generate masks for person images
    masks = ImageProcessor.generate_masks(person_images, cloth_type, automasker, mask_processor)

    # Generate combinations of person images and cloth images
    combinations = []
    for person_image, mask in zip(person_images, masks):
        for cloth_image in cloth_images:
            combinations.append((person_image, cloth_image, mask))

    results = []
    for i in range(0, len(combinations), batch_size):
        batch = combinations[i:i+batch_size]
        person_batch = [item[0] for item in batch]
        cloth_batch = [item[1] for item in batch]
        mask_batch = [item[2] for item in batch]

        try:
            # Convert images and masks to tensors
            person_tensors = ImageProcessor.images_to_tensor(person_batch).to(pipeline.device)
            cloth_tensors = ImageProcessor.images_to_tensor(cloth_batch).to(pipeline.device)
            mask_tensors = ImageProcessor.images_to_tensor(mask_batch).to(pipeline.device)

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

            # Move result images to CPU before post-processing
            result_images = [img.cpu() if torch.is_tensor(img) else img for img in result_images]

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
                        conditions = ImageProcessor.image_grid([person_image, cloth_image], 2, 1)
                    else:
                        masked_person = vis_mask(person_image, mask)
                        condition_width = width // 3
                        conditions = ImageProcessor.image_grid([person_image, masked_person, cloth_image], 3, 1)
                    conditions = conditions.resize((condition_width, height), Image.NEAREST)
                    new_result_image = Image.new("RGB", (width + condition_width + 5, height))
                    new_result_image.paste(conditions, (0, 0))
                    new_result_image.paste(result_image, (condition_width + 5, 0))
                    display_image = new_result_image

                # Append result
                results.append(display_image)

        finally:
            # Explicitly clear tensors from GPU memory
            del person_tensors
            del cloth_tensors
            del mask_tensors
            if 'result_images' in locals():
                del result_images
            torch.cuda.empty_cache()  # Clear CUDA cache

    return results


class InputRequest(BaseModel):
    person_files: List[str]
    cloth_files: List[str]
    cloth_type: Literal["upper", "lower", "overall"] = "upper"
    num_inference_steps: int = 10
    guidance_scale: float = 2.5
    seed: int = 42
    show_type: Literal["result only", "input & result", "input & mask & result"] = "input & mask & result"
    batch_size: int = 4
    target_width: int = 768
    target_height: int = 1024
    

app = FastAPI()

@app.post("/infer")
async def submit(request: InputRequest):
    try:
        result_images = await inference(
            person_files=request.person_files,
            cloth_files=request.cloth_files,
            cloth_type=request.cloth_type,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
            show_type=request.show_type,
            batch_size=request.batch_size,
            target_width=request.target_width,
            target_height=request.target_height,
        )

        # Convert images to base64 for JSON response
        encoded_images = [image_to_base64(img) for img in result_images]

        return JSONResponse(
            content={
                "status": "success",
                "images": encoded_images,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)

