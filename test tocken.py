from PIL import Image
from torchvision import transforms
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
vectors = []

image_folder = "images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

for i in image_paths:
    img = Image.open(i)
    token = transform(img)
    image_vector = token.view(-1).numpy().reshape(1, -1)
    vectors.append(image_vector)

# Appliquer PCA
pca = PCA(n_components=2)
points_2d = pca.fit_transform(np.vstack(vectors))

# Affichage
import matplotlib.pyplot as plt
for i, point in enumerate(points_2d):
    plt.scatter(point[0], point[1], label=f'Image {i + 1}')
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection 2D des deux images")
plt.legend()
plt.grid(True)
plt.show()




