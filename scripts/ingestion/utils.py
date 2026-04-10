import os


def load_text_files(folder_path: str):
    data = []

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                data.append({
                    "text": f.read(),
                    "source": file
                })

    return data


def list_images(folder_path: str):
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]