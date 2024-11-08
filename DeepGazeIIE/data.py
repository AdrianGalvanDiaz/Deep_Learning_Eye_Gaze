import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from boltons.cacheutils import cached, LRU
from .utils.data_utils import load_fixation_data, create_fixation_mask

class SaliencyDataset(Dataset):
    def __init__(self, data_list, centerbias_model=None, transform=None):
        self.data_list = data_list
        self.centerbias_model = centerbias_model
        self.transform = transform
        self._shapes = None
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data_item = self.data_list[idx]
        
        # Cargar imagen
        image = Image.open(data_item['image_path']).convert('RGB')
        
        # Obtener centro bias si existe el modelo
        if self.centerbias_model is not None:
            centerbias = self.centerbias_model.log_density(np.array(image))
        else:
            centerbias = np.zeros(image.size[::-1], dtype=np.float32)
        
        # Cargar datos de fijación
        fixation_data = load_fixation_data(
            data_item['fixloc_path'],
            'CAT2000' if 'CAT2000' in data_item['image_path'] else 'MIT1003'
        )
        
        # Crear máscara de fijación
        fixation_mask = create_fixation_mask(
            fixation_data['positions'],
            fixation_data['resolution']
        )
        
        # Aplicar transformaciones si existen
        if self.transform:
            image = self.transform(image)
            centerbias = self.transform(torch.from_numpy(centerbias).unsqueeze(0)).squeeze(0)
            fixation_mask = self.transform(fixation_mask.unsqueeze(0)).squeeze(0)
        
        return {
            'image': image,
            'centerbias': centerbias,
            'fixation_mask': fixation_mask,
            'category': data_item.get('category', 'unknown')
        }

    def get_shapes(self):
        """Devuelve las formas originales de las imágenes."""
        if self._shapes is None:
            self._shapes = []
            for item in self.data_list:
                with Image.open(item['image_path']) as img:
                    self._shapes.append(img.size[::-1])  # (H, W)
        return self._shapes