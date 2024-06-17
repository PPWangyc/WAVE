import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchmetrics.functional import mean_squared_error

class fid_wrapper:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fid = FrechetInceptionDistance(feature=64)
        self.fid.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
        ])

    @torch.no_grad()
    def __call__(self, pred_imgs, gt_imgs):
        self.fid.reset()
        pred_imgs, gt_imgs = self.transform(pred_imgs), self.transform(gt_imgs)
        pred_imgs = transforms.functional.convert_image_dtype(pred_imgs, dtype=torch.uint8)
        gt_imgs = transforms.functional.convert_image_dtype(gt_imgs, dtype=torch.uint8)
        self.fid.update(gt_imgs, real=True)
        self.fid.update(pred_imgs, real=False)
        return self.fid.compute().item()
    
def calculate_snr(recon_images, origin_images):
    recon_images, origin_images = transforms.Resize((256, 256))(recon_images), transforms.Resize((256, 256))(origin_images)
    recon_images = transforms.functional.convert_image_dtype(recon_images, dtype=torch.float32)
    origin_images = transforms.functional.convert_image_dtype(origin_images, dtype=torch.float32)
    # Calculate Mean Squared Error between reconstructed and original images
    mse = mean_squared_error(recon_images, origin_images)
    
    # Calculate power of the original signal
    signal_power = torch.mean(origin_images ** 2)
    
    # Calculate SNR in decibels
    snr = 10 * torch.log10(signal_power / mse)
    
    return snr.item()