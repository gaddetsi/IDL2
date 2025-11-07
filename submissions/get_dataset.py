from pathlib import Path
from zipfile import ZipFile

import requests


def download_data(url: str):
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download file: {response.status_code}")
    total_size = int(response.headers.get("content-length", 0))

    if total_size == 0:
        print("file does not exist")
        return

    if "content-disposition" in response.headers:
        filename = (
            response.headers["content-disposition"].split("filename=")[-1].strip('"')
        )
    else:
        filename = url.split("/")[-1]

    out_file = Path(filename).absolute()
    extract_dir = Path("data") / out_file.stem
    if extract_dir.exists():
        print(f"Data already exists at {extract_dir}. Skipping download.")
        return extract_dir

    with open(filename, "wb") as f:
        print(f"Downloading {filename}...")
        curr_size = 0
        for chunk in response.iter_content(chunk_size=8192):
            curr_size += len(chunk)
            done = int(50 * curr_size / total_size)
            if chunk:
                f.write(chunk)
            print(
                f"\r[{'=' * done}{' ' * (50 - done)}] {curr_size / 1024 / 1024:.2f}/{total_size / 1024 / 1024:.2f}MB {curr_size / total_size:.2%}",
                end="",
            )
    print(f"\nDownloaded file: {filename}")

    out_file = Path(filename).absolute()
    extract_dir = Path("data") / out_file.stem

    with ZipFile(out_file, "r") as zip_ref:
        print(f"Extracting {filename} to {extract_dir}...")
        zip_ref.extractall(extract_dir)

    out_file.unlink()  # Delete the zip file after extraction


if __name__ == "__main__":
    url = (
        r"https://surfdrive.surf.nl/public.php/dav/files/Nznt5c48Mzlb2HY/A1_data_75.zip"
    )
    download_data(url)

    url = r"https://surfdrive.surf.nl/public.php/dav/files/Nznt5c48Mzlb2HY/A1_data_150.zip"
    download_data(url)
