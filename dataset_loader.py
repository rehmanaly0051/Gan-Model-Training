from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset verification
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root='./house_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

for i, (images, labels) in enumerate(dataloader):
    print(f"Batch {i+1}: {images.shape}, {labels.shape}")
    break
