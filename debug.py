"""
debug.py
========
Quick diagnostic script – prints summary statistics for the saved stores.

Usage
-----
.. code-block:: bash

    python debug.py
"""

from core.vector_store import VectorStore


def inspect(store_path: str, label: str) -> None:
    print(f"\n{'='*50}")
    print(f"  {label}  ({store_path})")
    print(f"{'='*50}")

    store = VectorStore.load(store_path)
    print(f"  Total vectors : {len(store)}")
    print(f"  Dimension     : {store.dim}")
    print(f"  Index type    : {store.index_type}")

    print("\n  Sample IDs:")
    for sid in store.id_map[:5]:
        print(f"    {sid}")

    print("\n  Sample metadata:")
    for meta in store.metadata[:3]:
        preview = {k: str(v)[:80] for k, v in meta.items()}
        print(f"    {preview}")


if __name__ == "__main__":
    inspect("stores/text_store", "Text Store")
    inspect("stores/image_store", "Image Store")