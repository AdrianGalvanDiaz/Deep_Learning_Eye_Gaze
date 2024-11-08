from pathlib import Path

# Rutas base
BASE_PATH = Path(r"C:\Users\Adrian\Downloads\Data DeepGaze")

# Rutas CAT2000
CAT2000_PATH = BASE_PATH / "CAT2000"
CAT2000_TRAIN_STIMULI = CAT2000_PATH / "trainSet" / "Stimuli"
CAT2000_TRAIN_FIXMAPS = CAT2000_PATH / "trainSet" / "FIXATIONMAPS"
CAT2000_TRAIN_FIXLOCS = CAT2000_PATH / "trainSet" / "FIXATIONLOCS"

# Rutas MIT1003 (solo para testing)
MIT1003_PATH = BASE_PATH / "MIT1003"
MIT1003_STIMULI = MIT1003_PATH / "ALLSTIMULI"
MIT1003_FIXMAPS = MIT1003_PATH / "ALLFIXATIONMAPS"
MIT1003_DATA = MIT1003_PATH / "DATA"

