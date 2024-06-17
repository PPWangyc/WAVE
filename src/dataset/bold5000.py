import torch
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import convert_image_dtype

class BOLD5000_Test(torch.utils.data.Dataset):
    def __init__(self, root_dir, model, subj, seed, transform=None, mind_vis_path=None, mind_vis_group=0):
        self.root_dir = root_dir
        self.subj = subj
        self.seed = seed
        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        
        img_path = os.path.join(root_dir, "model_{}".format(model),"subj_{}".format(3), "seed_{}".format(seed), "origin_img")
        self.img_name = [os.path.join(img_path, img_name) for img_name in os.listdir(img_path)]
        self.avg_origin_img_path = os.path.join(root_dir, "avg_test_origin")
        self.mind_vis_path = mind_vis_path
        self.mind_vis_group = mind_vis_group
        if mind_vis_path:
            # list all the folders
            acc_paths = os.listdir(os.path.join(mind_vis_path, "val"))
            # sort the folders according to acc: round1_0.15
            self.acc_paths = sorted(acc_paths, key=lambda x: float(x.split("_")[-1]))
            self.img_len = len(os.listdir(os.path.join(self.mind_vis_path, "val", self.acc_paths[-1])))//5
            if self.img_len == 106:
                print("csi4")
                self.avg_origin_img_path = os.path.join(root_dir, "csi4")
            elif self.img_len == 445:
                print("All Subjects")
                self.avg_origin_img_path = os.path.join(root_dir, "all")
            self.img_name = [os.path.join(self.avg_origin_img_path, img_name) for img_name in os.listdir(self.avg_origin_img_path)]
            self.img_name = sorted(self.img_name, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            

    def __len__(self):
        if self.mind_vis_path:
            return self.img_len
        return len(self.img_name)
    
    def __getitem__(self, idx):
        img_name = self.img_name[idx]
        if self.mind_vis_path:
            recon_img_name = os.path.join(self.mind_vis_path, "val", self.acc_paths[-1], "test{}-{}.png".format(idx, self.mind_vis_group))
        else:
            recon_img_name = img_name.replace("origin_img", "recon_img")
        image = Image.open(img_name).convert('RGB')
        recon_image = Image.open(recon_img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
            recon_image = self.transform(recon_image)
        image = convert_image_dtype(image, torch.uint8)
        recon_image = convert_image_dtype(recon_image, torch.uint8)
        return image, recon_image, idx, img_name