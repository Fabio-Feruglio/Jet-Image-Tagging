
import torch


### Data Augmentation Classes (attempt)

class RandomDeadPixel(object):
    """
    Simulates dead pixels in a 2D image tensor by randomly setting a certain number of pixels to zero.
    """
    def __init__(self, p = 0.5, max_dead_pixels = 10):
        self.prob = p
        self.max_dead_pixels = max_dead_pixels

    def __call__(self, tensor):

        # Randomply decide whether to apply the transofrm
        if torch.rand(1).item() > self.prob:
            return tensor

        # Clone the input tensor
        img = tensor.clone()
        channels, height, width = img.shape
        
        # Choose a random number of dead pixels
        num_dead = torch.randint(1, self.max_dead_pixels + 1, (1,)).item()

        # Choose which pixels to set to zero
        for _ in range(num_dead):
            y = torch.randint(0, height, (1,)).item()
            x = torch.randint(0, width, (1,)).item()
            img[:, y, x] = 0.0 

        return img