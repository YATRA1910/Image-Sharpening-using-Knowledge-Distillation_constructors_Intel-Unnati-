
#  Setup
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import pytorch_msssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#  Dataset Class
class HelenDataset(Dataset):
    def __init__(self, blur_dir, sharp_dir, image_size=(256, 256)):
        self.blur_dir = blur_dir
        self.sharp_dir = sharp_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])
        blur_files = sorted([
            f for f in os.listdir(blur_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        sharp_files = sorted([
            f for f in os.listdir(sharp_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.filenames = [f for f in blur_files if f in sharp_files]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        blur = Image.open(os.path.join(self.blur_dir, fname)).convert('RGB')
        sharp = Image.open(os.path.join(self.sharp_dir, fname)).convert('RGB')
        return self.transform(blur), self.transform(sharp)

# Load Dummy Datasets (all point to same 10 dummy images)
train_dataset = HelenDataset(
    blur_dir='Helen/train/blur',
    sharp_dir='Helen/train/sharp',
    image_size=(256, 256)
)
valid_dataset = train_dataset
test_dataset = train_dataset

print(f"Loaded: Train={len(train_dataset)}, Valid={len(valid_dataset)}, Test={len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=2)
test_loader  = DataLoader(test_dataset, batch_size=2)

#  Model Definitions
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

    def forward(self, x):
        return self.model(x)

#  Metrics
def calculate_ssim(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return compare_ssim(img1, img2, data_range=1.0, channel_axis=-1)

def calculate_psnr(img1, img2):
    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0)
    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(img1, img2, data_range=1.0)

def evaluate(model, dataloader, name="Model"):
    model.eval()
    total_ssim, total_psnr, count = 0, 0, 0
    with torch.no_grad():
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            for i in range(output.size(0)):
                total_ssim += calculate_ssim(output[i], sharp[i])
                total_psnr += calculate_psnr(output[i], sharp[i])
                count += 1
    print(f" {name} SSIM: {total_ssim / count:.4f}, PSNR: {total_psnr / count:.2f} dB")

#  Training Loops
def train_teacher(model, dataloader, device, epochs=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    print("Training started...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (blurred, sharp) in enumerate(dataloader):
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            loss = F.mse_loss(output, sharp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"📘 Teacher Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

def distill_student(student, teacher, dataloader, device, epochs=5):
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    student.train()
    teacher.eval()
    for epoch in range(epochs):
        total_loss = 0
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            with torch.no_grad():
                teacher_out = teacher(blurred)
            student_out = student(blurred)
            loss = F.mse_loss(student_out, sharp) + 0.5 * F.mse_loss(student_out, teacher_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"🎓 Student Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

def distill_loss(student_out, teacher_out, target):
    mse = F.mse_loss(student_out, target)
    distill = F.mse_loss(student_out, teacher_out)
    ssim_loss = 1 - pytorch_msssim.ssim(student_out, target, data_range=1.0)
    return mse + 0.5 * distill + 0.3 * ssim_loss

# Main
if __name__ == "__main__":
    teacher = DnCNN().to(device)
    print("Training Teacher...")
    train_teacher(teacher, train_loader, device, epochs=3)
    evaluate(teacher, valid_loader, name="Teacher VALID")

    student = BetterStudent().to(device)
    print("Training Student with Distillation...")
    distill_student(student, teacher, train_loader, device, epochs=5)
    evaluate(student, valid_loader, name="Student VALID")
    print("Final Evaluation on TEST set:")
    evaluate(student, test_loader, name="Student TEST")

    torch.save(teacher.state_dict(), "teacher_model.pth")
    torch.save(student.state_dict(), "student_model.pth")
    print("Models saved.")
