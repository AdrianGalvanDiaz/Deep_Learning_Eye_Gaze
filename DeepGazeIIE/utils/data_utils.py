import os
from pathlib import Path
import numpy as np
import scipy.io as sio
from PIL import Image
import torch

def load_cat2000_data(stimuli_path, fixmap_path, fixloc_path):
    """Carga los datos de CAT2000"""
    data = []
    categories = [d for d in os.listdir(stimuli_path) if os.path.isdir(os.path.join(stimuli_path, d))]
    
    for category in categories:
        stim_cat_path = os.path.join(stimuli_path, category)
        fix_cat_path = os.path.join(fixmap_path, category)
        fixloc_cat_path = os.path.join(fixloc_path, category)
        
        # Solo procesar archivos de imagen
        images = [f for f in os.listdir(stim_cat_path) if f.endswith('.jpg')]
        
        for img_name in images:
            base_name = img_name.split('.')[0]
            
            data.append({
                'image_path': os.path.join(stim_cat_path, img_name),
                'fixmap_path': os.path.join(fix_cat_path, f"{base_name}.jpg"),
                'fixloc_path': os.path.join(fixloc_cat_path, f"{base_name}.mat"),
                'category': category
            })
    
    return data

def load_mit1003_data(stimuli_path, fixmap_path, data_path):
    """Carga los datos de MIT1003"""
    data = []
    images = [f for f in os.listdir(stimuli_path) if f.endswith('.jpg')]
    
    for img_name in images:
        base_name = img_name.split('.')[0]
        
        data.append({
            'image_path': os.path.join(stimuli_path, img_name),
            'fixmap_path': os.path.join(fixmap_path, f"{base_name}_fixMap.jpg"),
            'fixpts_path': os.path.join(fixmap_path, f"{base_name}_fixPts.jpg"),
            'data_path': os.path.join(data_path, f"{base_name}.mat")
        })
    
    return data

def load_fixation_data(mat_path):
    """Carga datos de fijación desde archivo .mat"""
    mat_data = sio.loadmat(mat_path)
    return {
        'positions': mat_data.get('gaze', []),
        'duration': mat_data.get('duration', []),
        'resolution': mat_data.get('resolution', [])
    }