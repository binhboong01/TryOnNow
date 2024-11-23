import gradio as gr
import os
import requests
import io
import base64
from PIL import Image
import logging
from requests.exceptions import HTTPError, ConnectionError, Timeout, RequestException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000/infer"

def submit_function(
    person_files,
    cloth_files,
    person_examples_selected,
    condition_examples_selected,
    cloth_type,
    num_inference_steps,
    guidance_scale,
    seed,
    show_type,
    target_width,
    target_height
):
    
    root_path = os.path.join(os.getcwd(), "resource/demo/example")
    # Combine uploaded and selected person images
    person_files = person_files or []
    print(person_files)
    if person_examples_selected:
        selected_person_paths = [os.path.join(root_path, "person", "men", label)
                                 if os.path.exists(os.path.join(root_path, "person", "men", label))
                                 else os.path.join(root_path, "person", "women", label)
                                 for label in person_examples_selected]
        person_files += selected_person_paths
    print(person_files)
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

    payload = {
        "person_files": person_images_paths,
        "cloth_files": cloth_images_paths,
        "cloth_type": cloth_type,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "show_type": show_type,
        "target_width": target_width,
        "target_height": target_height,
        "batch_size": 4,
    }
    print(payload)
    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        output = []
        if "images" in data:
            images = data["images"]
            for idx, img_base64 in enumerate(images):
                # Decode the base64 string back to an image
                img_data = base64.b64decode(img_base64)
                # print(img_data)
                img = Image.open(io.BytesIO(img_data))
                output.append(img)
        return output
        
    except HTTPError as http_err:
        logger.warning(f"HTTP error occurred: {http_err} (Status Code: {response.status_code})")
    except ConnectionError as conn_err:
        logger.warning(f"Connection error occurred: {conn_err}")
    except Timeout as timeout_err:
        logger.warning(f"Timeout error occurred: {timeout_err}")
    except RequestException as req_err:
        logger.warning(f"Request failed: {req_err}")
    except Exception as general_err:
        logger.warning(f"An unexpected error occurred: {general_err}")
        
    logger.warning(f"Failed to add to generate Images")
    return []       
        

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
                label="Inference Step", minimum=10, maximum=100, step=5, value=10
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
            target_width = gr.Radio(
                label="Target Width (616 for efficiency, 768 for accuracy)", 
                choices =[616, 768],
                value=616
            )
            target_height = gr.Radio(
                label="Target Height (820 for efficiency, 1024 for accuracy)", 
                choices = [820, 1024],
                value=820
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
                target_width,
                target_height
            ],
            outputs=result_gallery,
        )
    
    demo.queue().launch(share=True, show_error=True)


if __name__ == "__main__":
    app_gradio()
