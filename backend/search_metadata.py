import os
import faiss
import pickle
import numpy as np

#Search similaity with metadata
def search_similar_metadata(user_id, q_emb, query_keywords, top_k=5,
                            threshold=0.5, fallback_threshold=0.7):
    base = f"user_data/{user_id}"
    idx_path = f"{base}/metadata.index"
    map_path = f"{base}/metadata_mapping.pkl"
    inv_path = f"{base}/inverted_index.pkl"
    emb_path = f"{base}/embeddings.npy"

    if not all(os.path.exists(p) for p in (idx_path, map_path, inv_path, emb_path)):
        print(" One or more required index files are missing.")
        return []

    # Load artefacts
    idx      = faiss.read_index(idx_path)
    mapping  = pickle.load(open(map_path, 'rb'))
    inverted = pickle.load(open(inv_path, 'rb'))
    all_embs = np.load(emb_path)

    # Keyword filtering
    cand_idxs = set()
    for kw in query_keywords:
        cand_idxs |= set(inverted.get(kw, []))
    use_full = not cand_idxs     #if true, we search whole index

    # Vector search
    qv = q_emb.reshape(1, -1)
    if use_full:
        D, I = idx.search(qv, top_k)
        hits = list(zip(I[0], D[0]))
    else:
        sub_idxs = list(cand_idxs)
        sub_embs = all_embs[sub_idxs]
        sub_idx  = faiss.IndexFlatL2(sub_embs.shape[1])
        sub_idx.add(sub_embs)
        D, I = sub_idx.search(qv, min(len(sub_idxs), top_k))
        hits = [(sub_idxs[i], d) for i, d in zip(I[0], D[0])]

    # Treshold application, 0.5 used
    hits.sort(key=lambda x: x[1])
    results = []
    for i, dist in hits:
        if dist <= threshold:
            rec = mapping[i].copy()
            rec["_distance"] = float(dist)
            results.append(rec)

    # Fallback logic, if threshold fails, return atleast top option under 0.7
    if not results and hits:
        i_best, dist_best = hits[0]
        if dist_best <= fallback_threshold:
            rec = mapping[i_best].copy()
            rec["_distance"] = float(dist_best)
            results.append(rec)     # else: leave results empty

    return results


#--------------------------------------------TESTING-----------------------------------------
# if __name__ == "__main__":
#     from query_handler import embed_query_sentence, build_query_sentence

#     # Use a valid user_id with indexed metadata
#     user_id = "110913943088152059091"  # Replace with your real user_id

#     # TEST CASES
#     tests = [
#         {
#             "desc": "Exact match file: invoice.pdf modified March 2024",
#             "sentence": build_query_sentence("Ophthmate", "folder", None, ["Ophthmate"]),
#         },
#         {
#             "desc": "Query with just keywords",
#             "sentence": build_query_sentence(None, None, None, ["graduation", "party", "image"]),
#         },
#         {
#             "desc": "Query with only type and date",
#             "sentence": build_query_sentence(None, "image", "2023-05-21", []),
#         },
#         {
#             "desc": "Random keywords with no match",
#             "sentence": build_query_sentence(None, None, None, ["xylophone", "penguin"]),
#         }
#     ]

#     for test in tests:
#         print(f"\n🧪 {test['desc']}")
#         emb = embed_query_sentence(test["sentence"])
#         results = search_similar_metadata(user_id, emb, test["sentence"].split(), top_k=5, threshold=0.5)
#         print("🔍 Top Results:\n", results if results else "No results found.")
