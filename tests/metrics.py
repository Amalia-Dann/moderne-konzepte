
def recall_at_k(res, gold):
    return int(gold in res)

def mrr(res, gold):
    if gold in res:
        return 1/(res.index(gold)+1)
    return 0
