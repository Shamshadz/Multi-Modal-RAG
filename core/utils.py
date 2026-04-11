"""
utils.py
========
Shared helper functions for file I/O.

Functions
---------
load_text_files
    Read all ``.txt`` files from a folder into a list of dicts.
list_images
    Enumerate image file paths under a folder.
"""

from __future__ import annotations

import os
from typing import Dict, List


def load_text_files(folder_path: str) -> List[Dict[str, str]]:
    """
    Load every ``.txt`` file in *folder_path*.

    Parameters
    ----------
    folder_path : str
        Directory to scan.

    Returns
    -------
    list[dict]
        Each element: ``{"text": <file content>, "source": <filename>}``.
    """
    data = []
    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as fh:
            data.append({"text": fh.read(), "source": file})
    return data


def list_images(folder_path: str) -> List[str]:
    """
    Return absolute paths of all images in *folder_path*.

    Recognised extensions: ``.png``, ``.jpg``, ``.jpeg``, ``.webp``.

    Parameters
    ----------
    folder_path : str
        Directory to scan.

    Returns
    -------
    list[str]
        Sorted list of absolute image file paths.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in exts
    )