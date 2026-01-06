import os
import pickle
import numpy as np

from joblib import load as joblib_load
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch

DATA_DIR = "./data"
TRAIN_PKL = os.path.join(DATA_DIR, "train_dataset.pkl")
TEST_PKL  = os.path.join(DATA_DIR, "test_dataset.pkl")

ML_MODEL_PKL = "./model/ml_model_xgb.pkl"
AD_GATE_PKL  = "./model/ad_gate.pkl"

DL_MODULE_PY = "./dl_model.py"
DL_MODEL_PKL = "./model/net_0.pkl"

BATCH_SIZE = 64

def _to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()

def load_datalist(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def datalist_to_fp_y(datalist):
    X = np.vstack([_to_numpy(d.fp) for d in datalist]).astype(np.float32, copy=False)
    y = np.asarray([float(d.y.item()) for d in datalist], dtype=np.float32)
    return X, y

def metrics(y, yhat):
    return {
        "r2": float(r2_score(y, yhat)),
        "rmse": float(mean_squared_error(y, yhat, squared=False)),
        "mae": float(mean_absolute_error(y, yhat)),
    }

def compute_nn_similarity_to_train(Xtr, Xq, n_neighbors=1):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", n_jobs=-1)
    nn.fit(Xtr)
    dists, _ = nn.kneighbors(Xq, return_distance=True)
    sims = 1.0 - dists
    return sims.max(axis=1)

def load_dl_module(module_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl_user_module", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def predict_dl_on_datalist(dl_mod, model_path, datalist, batch_size=BATCH_SIZE, device=None):

    class _TmpDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            d = self.data_list[idx]
            fp = getattr(d, "fp")
            y  = getattr(d, "y", torch.tensor([0.0]))
            return d, fp.float(), y

    def _collate(batch):
        graphs, fps, ys = zip(*batch)
        batch_graph = Batch.from_data_list(graphs)
        fps = [fp.view(-1) for fp in fps]
        fps = torch.stack(fps, dim=0).float()
        ys  = torch.stack([y.view(-1) for y in ys], dim=0)
        return batch_graph, fps, ys

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = _TmpDataset(datalist)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)

    model = dl_mod.Model(device=device).to(device).eval()
    ckpt = torch.load(model_path, map_location=device)
    
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch_graph, fps, _ in loader:
            batch_graph = batch_graph.to(device)
            fps = fps.to(device)
            yhat = model(fps.size(0), batch_graph, fps).view(-1)
            preds.append(yhat.detach().cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.float32)

def main():
    train_list = load_datalist(TRAIN_PKL)
    if os.path.exists(TEST_PKL):
        test_list = load_datalist(TEST_PKL)

    Xtr, ytr = datalist_to_fp_y(train_list)
    Xte, yte = datalist_to_fp_y(test_list)

    ml_model = joblib_load(ML_MODEL_PKL)
    ad_gate  = joblib_load(AD_GATE_PKL)

    p_ml_te = ml_model.predict(Xte).astype(np.float32)

    dl_mod = load_dl_module(DL_MODULE_PY)
    p_dl_te = predict_dl_on_datalist(dl_mod, DL_MODEL_PKL, test_list)

    s_te = compute_nn_similarity_to_train(Xtr, Xte).reshape(-1, 1)

    w_dl_te = ad_gate.predict_proba(s_te)[:, 1].astype(np.float32)

    p_ens_te = w_dl_te * p_dl_te + (1.0 - w_dl_te) * p_ml_te

    print("==> TEST")
    print(metrics(yte, p_ens_te))

if __name__ == "__main__":
    main()
