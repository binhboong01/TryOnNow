from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from catvton.utils import resize_and_crop, resize_and_padding

class ImageProcessor(BaseModel):

    @staticmethod
    def image_grid(imgs, rows, cols):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    @staticmethod
    def preprocess_person_images(image_paths, target_size):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = resize_and_crop(image, target_size)
            images.append(image)
        return images

    @staticmethod
    def preprocess_cloth_images(image_paths, target_size):
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = resize_and_padding(image, target_size)
            images.append(image)
        return images

    @staticmethod
    def generate_masks(person_images, cloth_type, automasker, mask_processor):
        masks = []
        for person_image in person_images:
            mask = automasker(
                person_image,
                cloth_type
            )['mask']
            mask = mask_processor.blur(mask, blur_factor=9)
            masks.append(mask)
        return masks

    @staticmethod
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
