from sklearn.metrics import f1_score

def metrics(y, score):
    return {'macro_f1_at_0.5': float(f1_score(y, score >= 0.5, average='macro', labels=[0, 1], zero_division=0))}
