import argparse
import pickle
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from pubchemfp import GetPubChemFPs
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

def standardize(mol):

    try:
        clean_mol = rdMolStandardize.Cleanup(mol) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger() 
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
        te = rdMolStandardize.TautomerEnumerator() 
        mol_final = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        try:
            mol_final = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        except:
            mol_final = mol

    return mol_final


def build_features_from_smiles(smiles_list):
    Xdesc_list = []
    bad = 0
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(str(smi).strip())
            mol = standardize(mol)
            desc_vec = descriptors_from_mol(mol).astype(np.float32, copy=False)
            Xdesc_list.append(desc_vec)
        except Exception:
            bad += 1
            Xdesc_list.append(np.full((len(get_descriptor_names()),), np.nan, dtype=np.float32))
            continue
    X_desc = np.vstack(Xdesc_list).astype(np.float32, copy=False)
    return X_desc

def mol_to_fp(mol):
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    arr1 = np.zeros((167,), dtype=np.int8)
    ConvertToNumpyArray(maccs, arr1)

    erg = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=15, minPath=1)
    arr2 = np.array(erg, dtype=np.float32)

    pubchem = GetPubChemFPs(mol)
    arr3 = np.array(pubchem, dtype=np.int8)

    return np.concatenate([arr1, arr2, arr3])

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"r2": r2, "rmse": rmse, "mae": mae}

def _nanify(X):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    return X
nanify = FunctionTransformer(_nanify)
imputer = SimpleImputer(strategy="median")

def build_search(model_name, cv_folds=5, n_iter=40, n_jobs=-1, random_state=42):
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    if model_name == "rf":
        base = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
        pipe = Pipeline(steps=[("nanify", nanify), ("imputer", imputer), ("rf", base)])

        param_dist = {
            "rf__n_estimators": [300, 500, 800, 1000],
            "rf__max_depth": [None, 8, 12, 16, 24, 32],
            "rf__min_samples_split": [2, 5, 10],
            "rf__min_samples_leaf": [1, 2, 4],
            "rf__max_features": ["sqrt", "log2", 0.5, 0.8],
        }

    elif model_name == "svm":
        base = SVR()
        pipe = Pipeline(steps=[
            ("nanify", nanify),
            ("imputer", imputer),
            ("scaler", StandardScaler(with_mean=False)),
            ("svr", base),
        ])
        param_dist = {
            "svr__kernel": ["rbf"],
            "svr__C": np.logspace(-2, 3, 10),
            "svr__gamma": ["scale", "auto"] + list(np.logspace(-4, -1, 6)),
            "svr__epsilon": [0.01, 0.05, 0.1, 0.2, 0.3],
        }
    elif model_name == "xgb":
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=n_jobs,
            tree_method="hist",
            eval_metric="rmse",
        )
        pipe = Pipeline(steps=[("nanify", nanify), ("imputer", imputer), ("xgb", base)])
        param_dist = {
            "xgb__n_estimators": [400, 600, 800, 1000, 1200],
            "xgb__max_depth": [4, 6, 8, 10],
            "xgb__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
            "xgb__colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "xgb__reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "xgb__reg_lambda": [0.5, 1.0, 1.5, 2.0],
            "xgb__min_child_weight": [1, 3, 5, 7],
        }
    else:
        raise ValueError(f"unknown model name{model_name}")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
        refit=True,
    )
    return search

_DESC_FUNCS = Descriptors.descList 
_DESC_NAMES = [name for name, _ in _DESC_FUNCS] 

def get_descriptor_names():
    return list(_DESC_NAMES)

def descriptors_from_mol(mol, dtype=np.float32):
    d = len(_DESC_NAMES)
    row = np.full((d,), np.nan, dtype=np.float64)  
    if mol is None:
        return row.astype(dtype, copy=False)
    for j, (_, func) in enumerate(_DESC_FUNCS):
        try:
            v = func(mol)
            row[j] = float(v) 
        except Exception:
            pass
    row[~np.isfinite(row)] = np.nan
    return row.astype(dtype, copy=False)
    
def _to_numpy(x):
    import torch
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32).ravel()

def main():
    with open('./data/train_dataset.pkl', 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open('./data/test_dataset.pkl', 'rb') as f:
        test_dataset = pickle.load(f)

    Xfp_tr   = [_to_numpy(d.fp) for d in train_dataset]
    y_train  = np.asarray([float(d.y.item()) for d in train_dataset], dtype=np.float32)


    Xfp_t   = [_to_numpy(d.fp) for d in test_dataset]
    y_t    = np.asarray([float(d.y.item()) for d in test_dataset], dtype=np.float32)

    X_train = Xfp_tr
    X_t   = Xfp_t

    search = build_search(model_name='rf', cv_folds=5, n_iter=40,
                          n_jobs=-1, random_state=42)

    print(f"TRAINING...")
    search.fit(X_train, y_train)
    
    best_rmse = np.sqrt(-search.best_score_)
    
    best_r2 = cross_val_score(search.best_estimator_, X_train, y_train,
                              cv=search.cv, scoring='r2', n_jobs=-1).mean()

    print("\nBest params:", search.best_params_)
    print(f"Best CV RMSE: {best_rmse:.4f}")
    print(f"Best CV R2:   {best_r2:.4f}")

    best_model = search.best_estimator_
    yt_pred = best_model.predict(X_t)
    metrics = evaluate(y_t, yt_pred)
    print(f"VALID -> R2: {metrics['r2']:.4f} | "
          f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f}")

if __name__ == "__main__":
    main()
