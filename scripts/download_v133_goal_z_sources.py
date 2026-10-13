"""Download the two official Goal Z source archives and record provenance.

The files are downloaded once to a workspace sibling directory. Existing files
are never overwritten. No archive is extracted by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2] / "goal-z-sources"
SOURCES = {
    "musique": {
        "url": "https://drive.google.com/uc?export=download&id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
        "filename": "musique_v1.0.zip",
        "official_page": "https://github.com/StonyBrookNLP/musique",
    },
    "2wiki": {
        "url": "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1",
        "filename": "2wiki_data_ids_april7.zip",
        "official_page": "https://github.com/Alab-NII/2wikimultihop",
    },
}


def download(name: str) -> None:
    source = SOURCES[name]
    ROOT.mkdir(exist_ok=True)
    destination = ROOT / source["filename"]
    if destination.exists():
        raise FileExistsError(destination)
    session = requests.Session()
    response = session.get(source["url"], timeout=(30, 120), stream=True)
    response.raise_for_status()
    if name == "musique" and "text/html" in response.headers.get("content-type", ""):
        html = response.text
        form = BeautifulSoup(html, "html.parser").find("form")
        if form is None or form.get("action") != "https://drive.usercontent.google.com/download":
            raise RuntimeError("Unexpected Google Drive confirmation page")
        params = {item.get("name"): item.get("value") for item in form.find_all("input") if item.get("name")}
        response = session.get(form["action"], params=params, timeout=(30, 120), stream=True)
        response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        raise RuntimeError("Source returned HTML instead of archive")
    partial = destination.with_suffix(".partial")
    if partial.exists():
        raise FileExistsError(partial)
    digest = hashlib.sha256()
    size = 0
    try:
        with partial.open("wb") as stream:
            for chunk in response.iter_content(1 << 20):
                if chunk:
                    stream.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
        with partial.open("rb") as stream:
            if stream.read(4) != b"PK\x03\x04":
                raise RuntimeError("Downloaded content is not ZIP")
        partial.rename(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    record = {"source": name, "official_page": source["official_page"],
              "source_url": source["url"], "bytes": size, "sha256": digest.hexdigest(),
              "filename": destination.name}
    (ROOT / f"{name}-provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=SOURCES)
    download(parser.parse_args().source)
