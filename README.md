# Catvton Setup and Usage Guide

This README provides simple steps to set up and run the **Catvton** application, which includes both a backend API and a frontend UI.

## Features
- Accepts images of persons and clothing.
- Supports different clothing types (upper, lower, overall).
- Allows configuration of inference steps, guidance scale, and image dimensions.

## Setup Instructions

1. Create and activate a Python environment:
   ```bash
   conda create -n catvton python=3.9
   conda activate catvton
   ```

2. Install the package in editable mode:
   ```bash
   git clone https://github.com/binhboong01/Clothes-Virtual-Try-On.git
   cd Clothes-Virtual-Try-On.git
   pip install -e .
   ```

## Running the Application

1. Open **two terminals**:
   
   - **Terminal 1**: Start the backend API:
     ```bash
     python catvton/api.py
     ```

   - **Terminal 2**: Launch the frontend UI:
     ```bash
     python catvton/demo_ui.py
     ```


## API Documentation
Once the server starts, it will be accessible at `http://127.0.0.1:8000`.

FastAPI automatically generates interactive API documentation:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

## API Usage

### Endpoint
- **URL**: `POST http://127.0.0.1:8000/infer`
- **Request Body**:
  ```json
  {
      "person_files": ["/path/to/person1.jpg", "/path/to/person2.jpg"],
      "cloth_files": ["/path/to/cloth1.jpg", "/path/to/cloth2.jpg"],
      "cloth_type": "upper",
      "num_inference_steps": 10,
      "guidance_scale": 2.5,
      "seed": 42,
      "show_type": "input & mask & result",
      "batch_size": 4,
      "target_width": 768,
      "target_height": 1024
  }
  ```

### Example Request with Python

Experiment with [`catvton/notebooks/call_api.ipynb`](catvton/notebooks/call_api.ipynb)
Run the client script:
```bash
python catvton/inference.py
```

---
---
---
---
---
---
# Clothes-Virtual-Try-On

## Resources:

### Virtual try-on
1. Example guide: https://www.youtube.com/watch?v=C94pTaKoLbU

2. Possible model for virtual try-on: https://github.com/levihsu/OOTDiffusion
* Overall, the performance is good on upper, lower and full body with clothes and dress. However, there are some limitations:
"First, since our models are trained on paired human and garment
images, it may fail to get perfect results for cross-category virtual try-on, e.g., to
put a T-shirt on a woman in a long dress, or to let a man in pants wear a skirt.
This issue can be partially solved in the future by collecting datasets of each
person wearing different clothes in the same pose. Another limitation is that
some details in the original human image might be altered after virtual try-on,
such as muscles, watches or tattoos, etc. The reason is that the relevant body
area is masked and repainted by the diffusion model. Thus more practical pre-
and post-processing methods are required for addressing such problems."

3. Lightweight version possible model to deploy ( <= 8GB VRAM required): https://github.com/Zheng-Chong/CatVTON
* Test on RTX 3060 Mobile 6GB VRAM:
  - GPU Memory Usage:
    + Load model only: 3320MiB
    + During inference: 5020MiB
  - Processing Time: approximately 1 minute
* Test on RTX 3090 24GB VRAM:
  - GPU Memory Usage:
    + Load model only: 3466MiB
    + During inference: 8366MiB
  - Processing Time: 30s

**Important Note:** 
* Processing time can be changed by changing inference step. Those numbers above are for the default setting (50 inference step). For example, if reduce the inference step to 20, the processing time on RTX 3090 is only 10s
* Use 8GB VRAM will make outputs better than use 6GB VRAM

5. Simple guide to build a Full-stack Generative AI App: https://xiaolishen.medium.com/a-fullstack-text-to-image-generative-ai-app-you-can-build-in-an-afternoon-31990657344b

### Recommendation system:
1. Basic guide: https://www.geeksforgeeks.org/what-are-recommender-systems/
2. Recommendation system design:
- https://www.youtube.com/watch?v=lh9CNRDqKBk&t=1s
- https://www.youtube.com/watch?v=FoSCaue3lcg
- (Short) Architectural Patterns: https://www.youtube.com/shorts/nZuWCo52wTg
  
