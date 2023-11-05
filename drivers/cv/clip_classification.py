# %%
import os 
import sys 
from glob import glob
from random import shuffle

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))

# %%
from IPython.display import Image as IPythonImage, display

# %%
from src.utils.logger import setup_logger

# %%
log = setup_logger("Image Classification with CLIP")
log.setLevel('INFO')

# %%
import torch
from PIL import Image
# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")

# %%
import pandas as pd 

# %%
DATASET_PATH = "/share/ju/urbanECG/training_datasets/flooding_1000/"


# %%
sep29_whole = '/share/ju/nexar_data/2023/2023-09-29'
aug18_whole = '/share/ju/nexar_data/2023/2023-08-18'

# %%
#dataset = glob(os.path.join(DATASET_PATH, "*.jpg"))
#log.info(f"Found {len(dataset)} images in dataset")

# %%
dataset = glob(os.path.join(aug18_whole, "*", "frames","*.jpg"))
log.info(f"Found {len(dataset)} images in dataset")

# %%
from lavis.models import load_model_and_preprocess
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

# %%
def caption(image_path, model, vis_processors, device): 
    try:
        raw_image = Image.open(image_path).convert("RGB")

        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        caption = model.generate({"image": image})

        
        log.success(f"caption: {caption}")
        display(IPythonImage(filename=image_path))

        return caption
    except Exception as e:
        log.error(e)
        return None


# %%
zeroshot_model, zeroshot_vis_processors, zeroshot_txt_processors = load_model_and_preprocess("clip_feature_extractor", model_type="ViT-L-14", is_eval=True, device=device)
log.info(f"Loaded CLIP model: {zeroshot_model}")
# %%
def zeroshot(image_path, choices, model, vis_processors, txt_processors, device):
    raw_image = Image.open(image_path).convert("RGB")
    

    choices = [txt_processors["eval"](choice) for choice in choices]
    log.debug(f"choices: {choices}")

    sample = {"image": vis_processors['eval'](raw_image).unsqueeze(0).to(device), "text_input": choices}

    clip_features = model.extract_features(sample)

    image_features = clip_features.image_embeds_proj
    text_features = clip_features.text_embeds_proj

    sims = (image_features @ text_features.t())[0] / 0.01
    probs = torch.nn.Softmax(dim=0)(sims).tolist()

    for cls_nm, prob in zip(choices, probs):
        log.debug(f"{cls_nm}: {prob}")

    most_likely = max(zip(choices, probs), key=lambda x: x[1])
    if most_likely[1] < 0.5:
        log.debug(f"None of the choices are likely, most likely: {most_likely}")
        return most_likely

    if most_likely[0] == 'USA Flag' or most_likely[0] == 'LGBTQ Pride Rainbow Flag':
        #display(raw_image.resize((1280//2, 720//2)))
        #save symlink of image 
        os.symlink(image_path, f"/share/ju/urbanECG/training_datasets/flags/{os.path.splitext(os.path.basename(image_path))[0]}_{most_likely[0]}_{most_likely[1]}.jpg")
        log.success(f"{most_likely[0]} detected with probability {most_likely[1]}")
    
    return most_likely


# %%
choices = {"No Flag Present", "Street Sign", "Interstate Sign", "Highway Sign", "USA Flag", "LGBTQ Pride Flag"}

#prefix = 'a photo showing a'

#choices = [f"{prefix} {choice}" for choice in choices]


# %%
shuffle(dataset)
log.info(f"Shuffled dataset")
for img in dataset:
    zeroshot(img, choices, zeroshot_model, zeroshot_vis_processors, zeroshot_txt_processors, device)


