from core.vector_store import VectorStore

store = VectorStore.load("stores/text_store")

print("Total vectors:", len(store.id_map))
print("Sample metadata:", store.metadata[:3])