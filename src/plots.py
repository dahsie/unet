import matplotlib.pyplot as plt
import torch
import cv2
import os
import numpy as np

def prepare_plot(original_image, original_mask, pred_mask):
    """
    Prepares a matplotlib figure displaying the original image, the original mask, and the predicted mask.

    Parameters:
    -----------
    original_image : np.ndarray
        The original image used for prediction.
    original_mask : np.ndarray
        The ground truth segmentation mask.
    pred_mask : np.ndarray
        The segmentation mask predicted by the model.
    """
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
    ax[0].imshow(original_image)
    ax[1].imshow(original_mask)
    ax[2].imshow(pred_mask)
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    figure.tight_layout()

def make_predictions(model, image_path, mask_path, input_image_height=224, input_image_width=224, device="cuda", threshold=0.5):
    """
    Generates segmentation mask predictions from a given image using a UNet model.

    Parameters:
    -----------
    model : torch.nn.Module
        The UNet model used for prediction.
    image_path : str
        Path to the input image.
    mask_path : str
        Path to the ground truth segmentation mask.
    input_image_height : int, optional
        The height of the input image after resizing. Default is 224.
    input_image_width : int, optional
        The width of the input image after resizing. Default is 224.
    device : str, optional
        The device used for inference (e.g., "cuda" or "cpu"). Default is "cuda".
    threshold : float, optional
        The threshold for converting the predicted mask to binary. Default is 0.5.
    """
    model.eval()

    with torch.no_grad():
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = cv2.resize(image, (input_image_height, input_image_width))
        original_image = image.copy()
        
        groundTruthPath = os.path.join(mask_path)
        gtMask = cv2.imread(groundTruthPath, 0)
        gtMask = cv2.resize(gtMask, (input_image_height, input_image_width))
        
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)

        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()

        predMask = (predMask > threshold) * 255
        predMask = predMask.astype(np.uint8)
        
        prepare_plot(original_image, gtMask, predMask)
