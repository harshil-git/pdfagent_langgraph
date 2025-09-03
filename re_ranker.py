from sentence_transformers import CrossEncoder



model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")



def ranker(query,retrieved_docs):
    input_pairs = [(query, doc) for doc in retrieved_docs]
    scores = model.predict(input_pairs)
    ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    top_paragraphs = [i[0] for i in ranked[:3]]
    return top_paragraphs
