# inference_pipeline.py

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import time
import csv
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# ✅ Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ✅ Dataset Class
class HelenDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, image_size=(256, 256)):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        self.filenames = sorted([
            f for f in os.listdir(blur_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.exists(os.path.join(sharp_dir, f))
        ])

    def __len__(self): return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        blur = Image.open(os.path.join(self.blur_dir, fname)).convert('RGB')
        sharp = Image.open(os.path.join(self.sharp_dir, fname)).convert('RGB')
        return self.transform(blur), self.transform(sharp), fname

# ✅ Load Paths
blur_path = r"E:\DBlur\Helen\test\blur"
sharp_path = r"E:\DBlur\Helen\test\sharp"

test_dataset = HelenDataset(blur_path, sharp_path, image_size=(256, 256))
test_loader  = DataLoader(test_dataset, batch_size=4)

print(f"✅ Loaded: Test={len(test_dataset)}")

# ✅ Model Definitions
class DnCNN(nn.Module):
    def __init__(self, channels=3, depth=17, filters=64):
        super().__init__()
        layers = [nn.Conv2d(channels, filters, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(filters, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(filters, channels, 3, padding=1))
        self.model = nn.Sequential(*layers)  

    def forward(self, x):
        return x - self.model(x)


class BetterStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    def forward(self, x): return self.model(x)

# ✅ Load pretrained weights
teacher = DnCNN().to(device)
student = BetterStudent().to(device)

teacher.load_state_dict(torch.load("teacher_model.pth", map_location=device))
student.load_state_dict(torch.load("student_model.pth", map_location=device))

teacher.eval()
student.eval()

# ✅ Evaluation Metrics
def calculate_ssim(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    return compare_ssim(img1, img2, data_range=1.0, channel_axis=-1)

def calculate_psnr(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(1,2,0)
    img2 = img2.detach().cpu().numpy().transpose(1,2,0)
    return compare_psnr(img1, img2, data_range=1.0)

def evaluate(model, dataloader, name="Model"):
    model.eval()
    total_ssim, total_psnr, count = 0, 0, 0
    with torch.no_grad():
        for blurred, sharp, _ in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            for i in range(output.size(0)):
                total_ssim += calculate_ssim(output[i], sharp[i])
                total_psnr += calculate_psnr(output[i], sharp[i])
                count += 1
    print(f"✅ {name} SSIM: {total_ssim / count:.4f}, PSNR: {total_psnr / count:.2f} dB")

# ✅ SSIM Comparison Report
def compare_blur_to_outputs(student, teacher, dataloader, device, save_csv=False):
    total_bt = total_bs = total_bg = total_sg = 0
    count = 0
    rows = [["Image#", "Blur-Teacher SSIM", "Blur-Student SSIM", "Blur-GT SSIM", "Student-GT SSIM"]]

    with torch.no_grad():
        for batch_idx, (blurred, sharp, fnames) in enumerate(dataloader):
            blurred, sharp = blurred.to(device), sharp.to(device)
            t_out = teacher(blurred).clamp(0, 1)
            s_out = student(blurred).clamp(0, 1)

            for i in range(blurred.size(0)):
                b = blurred[i].cpu().numpy().transpose(1, 2, 0)
                t = t_out[i].cpu().numpy().transpose(1, 2, 0)
                s = s_out[i].cpu().numpy().transpose(1, 2, 0)
                g = sharp[i].cpu().numpy().transpose(1, 2, 0)

                ssim_bt = compare_ssim(b, t, data_range=1.0, channel_axis=-1)
                ssim_bs = compare_ssim(b, s, data_range=1.0, channel_axis=-1)
                ssim_bg = compare_ssim(b, g, data_range=1.0, channel_axis=-1)
                ssim_sg = compare_ssim(s, g, data_range=1.0, channel_axis=-1)

                total_bt += ssim_bt
                total_bs += ssim_bs
                total_bg += ssim_bg
                total_sg += ssim_sg
                count += 1

                rows.append([fnames[i], f"{ssim_bt:.4f}", f"{ssim_bs:.4f}", f"{ssim_bg:.4f}", f"{ssim_sg:.4f}"])

    print("\n📊 Average SSIM Comparison:")
    print(f"🔹 Blur vs Teacher:       {total_bt / count:.4f}")
    print(f"🔸 Blur vs Student:       {total_bs / count:.4f}")
    print(f"🎯 Blur vs Ground Truth:  {total_bg / count:.4f}")
    print(f"🎓 Student vs Ground Truth: {total_sg / count:.4f}")

    if save_csv:
        with open("ssim_comparison_report.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print("✅ CSV saved: ssim_comparison_report.csv")

# ✅ Visualize Selected Files
def visualize_images(model_teacher, model_student, dataloader, image_names):
    model_teacher.eval()
    model_student.eval()

    with torch.no_grad():
        for blurred, sharp, fnames in dataloader:
            for i, fname in enumerate(fnames):
                if fname in image_names:
                    b = blurred[i:i+1].to(device)
                    s = sharp[i:i+1].to(device)
                    t_out = model_teacher(b).clamp(0, 1)
                    s_out = model_student(b).clamp(0, 1)

                    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
                    axs[0].imshow(TF.to_pil_image(b[0].cpu()))
                    axs[0].set_title("🔸 Blurred Input")
                    axs[1].imshow(TF.to_pil_image(t_out[0].cpu()))
                    axs[1].set_title("🧠 Teacher Output")
                    axs[2].imshow(TF.to_pil_image(s_out[0].cpu()))
                    axs[2].set_title("🎓 Student Output")
                    axs[3].imshow(TF.to_pil_image(s[0].cpu()))
                    axs[3].set_title("🎯 Ground Truth")
                    for ax in axs: ax.axis("off")
                    plt.tight_layout()
                    plt.savefig(f"visualization_{fname}", bbox_inches='tight')
                    
                    plt.show(block=True)

                    

# ✅ Measure FPS
def measure_fps(model, dataloader, device, warmup=5):
    model.eval()
    total_time, total_images = 0, 0
    with torch.no_grad():
        for i, (blurred, _, _) in enumerate(dataloader):
            blurred = blurred.to(device)
            if i < warmup:
                _ = model(blurred)
                continue
            start = time.time()
            _ = model(blurred)
            total_time += time.time() - start
            total_images += blurred.size(0)
    print(f"⚡ Inference FPS: {total_images / total_time:.2f} images/sec")

# ✅ Run Everything
evaluate(teacher, test_loader, name="Teacher TEST")
evaluate(student, test_loader, name="Student TEST")

compare_blur_to_outputs(student, teacher, test_loader, device, save_csv=True)
plt.show(block=True)


# Show a few example results 
visualize_images(teacher, student, test_loader, image_names=[
    "6.jpg", "8.jpg", "59.jpg"
])


measure_fps(student, test_loader, device)
