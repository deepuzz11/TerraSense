import os
from torch.utils.data import Dataset
from PIL import Image

class TerrainDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.root_dir = os.path.join(root_dir, mode)
        self.transform = transform
        self.terrain_classes = {'Deserts': 0, 'Mountains': 1, 'Forest Cover': 2}
        self.samples = []
        for terrain, label in self.terrain_classes.items():
            folder_path = os.path.join(self.root_dir, terrain)
            for filename in os.listdir(folder_path):
                if filename.endswith(('.jpg', '.png')):
                    self.samples.append((os.path.join(folder_path, filename), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
