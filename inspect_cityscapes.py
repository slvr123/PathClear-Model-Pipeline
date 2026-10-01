from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

print("Script started...")

image_path = r"C:\datasets\cityscapes\leftImg8bit\train\aachen\aachen_000000_000019_leftImg8bit.png"
mask_path = r"C:\datasets\cityscapes\gtFine\train\aachen\aachen_000000_000019_gtFine_labelTrainIds.png"

print("Loading image...")
image = Image.open(image_path)

print("Loading mask...")
mask = np.array(Image.open(mask_path))

print("Image size:", image.size)
print("Mask shape:", mask.shape)
print("Unique class IDs found:", np.unique(mask))

palette = np.array([
    [128,  64, 128],   # road
    [244,  35, 232],   # sidewalk
    [ 70,  70,  70],   # building
    [102, 102, 156],   # wall
    [190, 153, 153],   # fence
    [153, 153, 153],   # pole
    [250, 170,  30],   # traffic light
    [220, 220,   0],   # traffic sign
    [107, 142,  35],   # vegetation
    [152, 251, 152],   # terrain
    [ 70, 130, 180],   # sky
    [220,  20,  60],   # person
    [255,   0,   0],   # rider
    [  0,   0, 142],   # car
    [  0,   0,  70],   # truck
    [  0,  60, 100],   # bus
    [  0,  80, 100],   # train
    [  0,   0, 230],   # motorcycle
    [119,  11,  32],   # bicycle
], dtype=np.uint8)

colored_mask = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)

for class_id in range(19):
    colored_mask[mask == class_id] = palette[class_id]

print("Creating visualization...")

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Cityscapes Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(colored_mask)
plt.title("19-Class Training Mask")
plt.axis("off")

plt.tight_layout()

# Save a copy even if the popup does not work
plt.savefig("cityscapes_check.png", dpi=150)

print("Saved cityscapes_check.png")
print("Opening visualization...")

plt.show()

print("Script finished.")