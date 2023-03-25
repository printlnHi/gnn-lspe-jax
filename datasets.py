import pickle
from utils import pad_all
def mutag():
    with open('data/mutag.pickle', 'rb') as f:
        ds = pickle.load(f)
    # TODO: Does it make sense to pad and then turn into tuples?
    ds = pad_all(ds)
    ds = [(d["input_graph"], d["target"]) for d in ds]
    return (ds[:150], ds[150:-10], ds[-10:])

def zinc():
    with open('data/ZINC.pkl', 'rb') as f:
        ds = pickle.load(f)
    return ds