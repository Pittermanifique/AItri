import requests
import json
import os


# 🔑 Ta clé API Pexels
PEXELS_API_KEY = "L5VjkIXreZ584oPhSrwGYtDfDgKJNTg60FrRYcb9cq8cR3nK85SZ9VKE"  # remplace par ta vraie clé

# 🔍 Paramètres de recherche
headers = {"Authorization": PEXELS_API_KEY}
query = "chats"
per_page = 15  # max 80
pages = 3  # nombre de pages à récupérer

image_count = 0

for page in range(1, pages + 1):
    url = f"https://api.pexels.com/v1/search?query={query}&per_page={per_page}&page={page}"
    response = requests.get(url, headers=headers)
    data = response.json()

    for photo in data.get("photos", []):
        image_url = photo["src"]["original"]
        image_data = requests.get(image_url).content
        save_folder = "images/chats"
        os.makedirs(save_folder, exist_ok=True)

        filename = f"{query}_{image_count}.jpg"
        save_path = os.path.join(save_folder, filename)

        with open(save_path, "wb") as f:
            f.write(image_data)

        image_count += 1

print(f"{image_count} images téléchargées avec succès !")
