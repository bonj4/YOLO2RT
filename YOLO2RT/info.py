import requests

def get_latest_version():
    url = "https://raw.githubusercontent.com/bonj4/YOLO2RT/refs/heads/main/YOLO2RT/version.py"
    response = requests.get(url)
    if response.status_code == 200:
        content = response.text
        version_line = [line for line in content.split('\n') if 'VERSION' in line][0]
        latest_version = version_line.split('"')[1]
        return latest_version
    return None


