import requests
import json

API_URL = "http://127.0.0.1:8000/parse"

IMAGE_PATH = r"C:\Users\Admin\Mihir_personal\Omni_Parser\OmniParser\test_images\test1.png"


def test_omni_parser():

    with open(IMAGE_PATH, "rb") as f:
        files = {"file": f}

        response = requests.post(API_URL, files=files)

    if response.status_code != 200:
        print("❌ Error:", response.status_code)
        print(response.text)
        return

    data = response.json()

    print("✅ Number of components:", len(data["components"]))

    # print first few components
    for comp in data["components"][:5]:
        print(json.dumps(comp, indent=2))


if __name__ == "__main__":
    test_omni_parser()