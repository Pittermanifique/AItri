import json

from PIL import Image
from scipy.spatial import ConvexHull
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import os


def tracer_contour(points, color):
    hull = ConvexHull(points)
    plt.scatter(points[:, 0], points[:, 1], color=color)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=color)

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
pca = PCA(n_components=2)
ref_vectors = {}


directoryimages = ["images/chiens","images/poissons"]
colors = ["red","blue"]
all_vectors = []
group_vectors = []

for image in directoryimages:
    image_paths = [os.path.join(image, f) for f in os.listdir(image) if f.endswith(".jpg")]
    temp_vectors = []
    for i in image_paths:
        img = Image.open(i)
        token = transform(img)
        image_vector = token.view(-1).numpy().reshape(1, -1)
        temp_vectors.append(image_vector)
        all_vectors.append(image_vector)
    group_vectors.append(temp_vectors)

# Entraîne le PCA sur tous les vecteurs réunis
pca.fit(np.vstack(all_vectors))
# Ensuite, tu projettes chaque groupe
compteur = 0
for i, group in enumerate(group_vectors):
    group_array = np.vstack(group)
    points_2d = pca.transform(group_array)
    ref_vector = np.mean(group_array, axis=0)
    ref_proj = pca.transform(ref_vector.reshape(1, -1))[0]
    label = os.path.basename(directoryimages[compteur])
    ref_vectors[label] = ref_vector.tolist()
    for pt in points_2d:
        plt.scatter(pt[0], pt[1], color=colors[i])
    plt.scatter(ref_proj[0], ref_proj[1], color="purple", s=100)
    tracer_contour(points_2d, colors[i])
    compteur += 1


with open("vecteurs_reference.json", "w") as f:
    json.dump(ref_vectors, f, indent=4)
print("Fichier JSON créé avec succès ✅")


plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection 2D des vecteurs d'images")
plt.grid(True)
plt.show()