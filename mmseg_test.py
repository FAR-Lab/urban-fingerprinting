from glob import glob 
import random 


import matplotlib.pyplot as plt
import mmcv

img_path = random.choice(glob("/share/ju/nexar_data/2023/2023-11-08/*/frames/*.jpg"))
print(img_path)

config_path = "/share/ju/urbanECG/static/libraries/mmsegmentation/output_swin/upernet_swin_bdd100k-720x1280.py"
checkpoint_path = "/share/ju/urbanECG/static/libraries/mmsegmentation/output_swin/iter_80000.pth"

from mmseg.apis import init_model, inference_model, show_result_pyplot


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)

# read image with cv2 
img = mmcv.imread(img_path)

# visualize results and save, don't show
tensor = result.pred_sem_seg.data.cpu().numpy()[0]

fig, ax = plt.subplots(figsize=(12.8, 7.2))
ax.imshow(img)
ax.imshow(tensor, alpha=0.5)
plt.axis('off')
plt.tight_layout()
plt.savefig("result.jpg", dpi=300)
