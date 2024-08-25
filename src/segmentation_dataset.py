
from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):
    """
    Custom PyTorch Dataset class for loading image and mask pairs for image segmentation tasks.

    This dataset class is designed to load images and their corresponding segmentation masks from 
    specified file paths. The images are converted from BGR to RGB format, and the masks are loaded 
    in grayscale. Optionally, transformations can be applied to both images and masks.

    Parameters:
    -----------
    imagePaths : list[str]
        A list of file paths to the images.
    maskPaths : list[str]
        A list of file paths to the corresponding segmentation masks. The order of the masks should 
        correspond to the order of the images in `imagePaths`.
    transforms : callable, optional
        A function/transform that takes in an image and a mask and returns transformed versions of 
        both. This can be used for data augmentation.

    Methods:
    --------
    __len__() -> int
        Returns the total number of samples in the dataset, which is equal to the number of images.
    
    __getitem__(idx: int) -> tuple
        Retrieves the image and corresponding mask at the specified index `idx`. The image is loaded 
        and converted from BGR to RGB format, and the mask is loaded in grayscale mode. If 
        transformations are specified, they are applied to both the image and the mask before 
        returning them.

    Example:
    --------
    >>> from torchvision import transforms
    >>> dataset = SegmentationDataset(imagePaths=["path/to/image1.png", "path/to/image2.png"],
                                      maskPaths=["path/to/mask1.png", "path/to/mask2.png"],
                                      transforms=transforms.ToTensor())
    >>> image, mask = dataset[0]
    >>> print(image.shape, mask.shape)
    torch.Size([3, H, W]) torch.Size([1, H, W])
    """
    
    def __init__(self, image_paths, mask_paths, image_transforms=None, mask_transforms=None):
        # Store the image and mask file paths, and the optional transformations
        self.imagePaths = image_paths
        self.maskPaths = mask_paths
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
    
    def __len__(self):
        # Return the number of total samples in the dataset
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        # Grab the image path from the current index
        imagePath = self.imagePaths[idx]
        
        # Load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)
        
        # Check to see if we are applying any transformations
        if self.image_transforms is not None:
            # Apply the transformations to both the image and its mask
            image = self.image_transforms(image)

        if self.mask_transforms is not None:  
            mask = self.mask_transforms(mask)
        
        # Return a tuple of the image and its mask
        return image, mask
