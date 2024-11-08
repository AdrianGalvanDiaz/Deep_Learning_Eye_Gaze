import os
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from boltons.iterutils import chunked

def load_cat2000_data(stimuli_path, fixmap_path, fixloc_path):
    """
    Carga los datos de CAT2000 manteniendo la estructura original.
    """
    data = []
    categories = [d for d in os.listdir(stimuli_path) 
                 if os.path.isdir(os.path.join(stimuli_path, d))]
    
    for category in categories:
        stim_cat_path = os.path.join(stimuli_path, category)
        fix_cat_path = os.path.join(fixmap_path, category)
        fixloc_cat_path = os.path.join(fixloc_path, category)
        
        # Solo procesar archivos de imagen
        images = [f for f in os.listdir(stim_cat_path) 
                 if f.endswith('.jpg') and not f.endswith('SaliencyMap.jpg')]
        
        for img_name in images:
            base_name = img_name.split('.')[0]
            fixloc_file = os.path.join(fixloc_cat_path, f"{base_name}.mat")
            
            # Verificar que existe el archivo .mat
            if not os.path.exists(fixloc_file):
                continue
                
            data.append({
                'image_path': os.path.join(stim_cat_path, img_name),
                'fixmap_path': os.path.join(fix_cat_path, f"{base_name}.jpg"),
                'fixloc_path': fixloc_file,
                'category': category
            })
    
    return data

def load_mit1003_data(stimuli_path, fixmap_path, data_path):
    """
    Carga los datos de MIT1003 para testing.
    """
    data = []
    images = [f for f in os.listdir(stimuli_path) 
             if f.endswith('.jpg') and not f.startswith('.')]
    
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        data.append({
            'image_path': os.path.join(stimuli_path, img_name),
            'fixmap_path': os.path.join(fixmap_path, f"{base_name}_fixMap.jpg"),
            'fixpts_path': os.path.join(fixmap_path, f"{base_name}_fixPts.jpg"),
            'data_path': os.path.join(data_path, f"{base_name}.mat")
        })
    
    return data

def load_fixation_data(mat_path, dataset_type='CAT2000'):
    """
    Carga datos de fijación exactamente como en el paper original.
    """
    mat_data = sio.loadmat(mat_path)
    
    if dataset_type == 'CAT2000':
        # Formato CAT2000: [x_coords, y_coords, durations]
        gaze_data = mat_data['gaze']
        fixation_data = {
            'positions': {
                'x': gaze_data[0].astype(np.float32),
                'y': gaze_data[1].astype(np.float32)
            },
            'duration': gaze_data[2].astype(np.float32) if gaze_data.shape[0] > 2 else None,
            'resolution': mat_data['resolution'].flatten().astype(np.int32)
        }
    
    elif dataset_type == 'MIT1003':
        # Formato MIT1003
        fixation_data = {
            'positions': {
                'x': mat_data['gaze']['x'][0][0].flatten().astype(np.float32),
                'y': mat_data['gaze']['y'][0][0].flatten().astype(np.float32)
            },
            'duration': mat_data['gaze']['time'][0][0].flatten().astype(np.float32),
            'resolution': mat_data['resolution'].flatten().astype(np.int32)
        }
    
    return fixation_data

def create_fixation_mask(positions, resolution):
    """
    Crea una máscara de fijación siguiendo el formato original.
    """
    mask = torch.zeros(resolution, dtype=torch.float32)
    x_coords = positions['x']
    y_coords = positions['y']
    
    for x, y in zip(x_coords, y_coords):
        if 0 <= x < resolution[0] and 0 <= y < resolution[1]:
            mask[int(y), int(x)] = 1
    
    return mask