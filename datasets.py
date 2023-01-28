import pickle
def mutag():
    with open('data/mutag.pickle', 'rb') as f:
        ds = pickle.load(f)
    return (ds[:150], ds[150:])

def zinc():
    with open('data/ZINC.pkl', 'rb') as f:
        ds = pickle.load(f)
    return ds