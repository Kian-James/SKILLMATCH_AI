from sklearn.metrics.pairwise import cosine_similarity

def cosine(a, b):
    return cosine_similarity([a], [b])[0][0]
