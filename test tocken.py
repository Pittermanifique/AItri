from PIL import Image
from scipy.spatial import ConvexHull
from torchvision import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import os



def tracer_cercle(points, color):
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1))
    cercle = plt.Circle(center, radius, color=color, fill=False, linestyle='--')
    plt.gca().add_patch(cercle)

def tracer_contour(points, color):
    hull = ConvexHull(points)
    plt.scatter(points[:, 0], points[:, 1], color=color)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], color=color)

def point_dans_enveloppe(point, cloud_points):
    hull = ConvexHull(cloud_points)
    contour = cloud_points[hull.vertices]
    polygon = Path(contour)
    return polygon.contains_point(point)

def similarite_cos(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    return np.dot(v1, v2)



transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])

color1 = "red"
color2 = "blue"

vectors1 = []
vectors2 = []

pca = PCA(n_components=2)

image_folder = "images"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

for i in image_paths:
    print(i)
    img = Image.open(i)
    token = transform(img)
    image_vector = token.view(-1).numpy().reshape(1, -1)
    if i[7] == "f":
        vectors1.append(image_vector)
    elif i[7] == "d":
        vectors2.append(image_vector)

# Appliquer PC
all_vectors = np.vstack(vectors1 + vectors2)
pca.fit(all_vectors)

# Transformer chaque groupe individuellement
points_2d1 = pca.transform(np.vstack(vectors1))
points_2d2 = pca.transform(np.vstack(vectors2))

# Test pour savoir si un point appartien a une envelope convex
point_test = points_2d1[0]
print(point_dans_enveloppe(point_test,points_2d1))


ref_vector1 = np.mean(np.vstack(vectors1), axis=0)
ref_vector2 = np.mean(np.vstack(vectors2), axis=0)
ref_vector1_proj = pca.transform(ref_vector1.reshape(1, -1))[0]
ref_vector2_proj = pca.transform(ref_vector2.reshape(1, -1))[0]


# Test pour determine de quel verteur de rèfe elle reseble le plus

similarity1 = similarite_cos(points_2d1[0], ref_vector1_proj)
similarity2 = similarite_cos(points_2d1[0], ref_vector2_proj)
print(similarity1, similarity2)



# Affichage
plt.xlim(-200, 200)
plt.ylim(-110, 100)

for i, point in enumerate(points_2d1):
    if np.allclose(point, point_test):
        plt.scatter(point[0], point[1], color="green",s=100)
    else:
        plt.scatter(point[0], point[1], color=color1)
for i, point in enumerate(points_2d2):
    plt.scatter(point[0], point[1],color=color2)
plt.scatter(ref_vector1_proj[0], ref_vector1_proj[1], color="purple", s=100)
plt.scatter(ref_vector2_proj[0], ref_vector2_proj[1], color="yellow", s=100)
tracer_contour(points_2d1, color1)
tracer_contour(points_2d2, color2)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection 2D des deux images")
plt.grid(True)
plt.show()




