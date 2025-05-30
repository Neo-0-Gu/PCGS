import os
import torch
import torchvision.transforms as transforms
from torchvision.models.inception import inception_v3
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from scipy import linalg
from tqdm import tqdm

class ImageFolder(Dataset):
    def __init__(self, root, transform):
        self.paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(image)

def get_activations(dataloader, model, device):
    model.eval()
    activations = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            batch = batch.to(device)
            pred = model(batch)
            activations.append(pred.cpu().numpy())

    return np.concatenate(activations, axis=0)

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("Warning: fid calculation produced singular product; adding small epsilon.")
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

def calculate_fid(source_dir, stego_dir, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load datasets
    source_set = ImageFolder(source_dir, transform)
    stego_set = ImageFolder(stego_dir, transform)
    source_loader = DataLoader(source_set, batch_size=batch_size, shuffle=False, num_workers=4)
    stego_loader = DataLoader(stego_set, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load inception model
    inception = inception_v3(pretrained=True, transform_input=False).to(device)
    inception.fc = torch.nn.Identity()  # remove classification head
    inception.eval()

    # Extract features
    act1 = get_activations(source_loader, inception, device)
    act2 = get_activations(stego_loader, inception, device)

    # Compute statistics
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)

    # Calculate FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

# Example usage:
if __name__ == "__main__":
    source_dir = "../data/source"
    stego_dir = "../data/stego"
    fid_score = calculate_fid(source_dir, stego_dir)
    print(f"FID score between source and stego: {fid_score:.4f}")
