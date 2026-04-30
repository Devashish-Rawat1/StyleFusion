from torch.utils.data import Dataset
import os
from PIL import Image #PIL (Python Imaging Library) is a library in Python that provides tools for opening, manipulating, and saving many different image file formats. It is commonly used for image processing tasks such as resizing, cropping, and applying filters to images.
from torchvision import transforms

# Custom dataset class for loading images from a directory. It inherits from torch.utils.data.Dataset, which is a PyTorch class that provides an interface for accessing and manipulating datasets. The ImageFolderDataset class is designed to load images from a specified directory, apply transformations to them, and return the processed images when accessed.
class ImageFolderDataset(Dataset):
    # The __init__ method initializes the dataset by taking the root directory and an optional transform as input. It lists all the files in the root directory and filters out only those that are images (with extensions .jpg, .png, .jpeg). The transform is stored for later use when processing the images.
    def __init__(self, root, transform = None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]
    
    # The __len__ method returns the total number of images in the dataset, which is determined by the length of the self.files list.
    def __len__(self):
        return len(self.files)
    
    # The __getitem__ method takes an index (idx) as input and retrieves the corresponding image from the dataset. It constructs the full path to the image, opens it using PIL, and converts it to RGB format. If a transform is provided, it applies the transform to the image before returning it. This method allows us to access individual images from the dataset in a way that is compatible with PyTorch's DataLoader.
    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


# The get_transform function is a utility function that creates a transformation pipeline for preprocessing images. It takes three arguments: size, crop, and final_size. The function constructs a list of transformations based on the input parameters. If size is greater than 0, it adds a resizing transformation to the list. If crop is True, it adds a random cropping transformation; otherwise, it adds another resizing transformation to ensure the final output size is consistent. Finally, it adds a transformation to convert the image to a tensor and composes all the transformations into a single pipeline using transforms.Compose. 
def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
        

# The adaptive_instance_normalization function is a key component of the style transfer process. It takes two feature maps as input: content_feat and style_feat. The function first calculates the mean and standard deviation of both the content and style feature maps using the calc_mean_std function. Then, it normalizes the content feature map by subtracting its mean and dividing by its standard deviation. Finally, it scales and shifts the normalized content feature map using the mean and standard deviation of the style feature map to produce the output that combines the content and style information.
def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels, h, w]
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)


# The calc_mean_std function calculates the mean and standard deviation of a given feature map (feat). It first checks that the input feature map has four dimensions (batch size, channels, height, width). Then, it computes the mean and variance across the spatial dimensions (height and width) for each channel and batch. The mean is calculated by reshaping the feature map and taking the mean along the appropriate dimension, while the variance is calculated similarly but with an additional step to ensure numerical stability by adding a small epsilon value. Finally, the standard deviation is obtained by taking the square root of the variance.
def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels, h, w]
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std