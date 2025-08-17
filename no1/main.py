# main.py — 提出用：CV戦略スイッチ + 追加カテゴリ特徴
import os, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:
    StratifiedGroupKFold = None

TARGET = "default"
ID_COL = "id"

# -------------------- Utils --------------------
def write_submission(ids, pred, path="submission.csv"):
    # 30桁以内要件に合わせて小数8桁で書き出し
    df = pd.DataFrame({ID_COL: ids, TARGET: pred.astype(np.float32)})
    df.to_csv(path, index=False, float_format="%.8f")

def safe_div(a, b):
    b = np.where(np.asarray(b) == 0, 1e-9, b)
    return a / b

def bucket3_from_text(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.lower().fillna("")
    is_debt   = s.str.contains("debt",   na=False)
    is_credit = s.str.contains("credit", na=False)
    return pd.Series(np.where(is_debt, "DEBT", np.where(is_credit, "CREDIT", "OTHER")), index=s.index)

def kfold_target_encode(train_df, test_df, cols, y, n_splits=5, smooth=600, seed=42, suffix="te"):
    if isinstance(cols, (list, tuple)):
        key_tr = train_df[cols].astype(str).agg("||".join, axis=1)
        key_te = test_df[cols].astype(str).agg("||".join, axis=1)
        name = "_".join(cols) + f"_{suffix}"
    else:
        key_tr = train_df[cols].astype(str)
        key_te = test_df[cols].astype(str)
        name = f"{cols}_{suffix}"

    prior = y.mean()
    oof = pd.Series(np.nan, index=train_df.index, name=name)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(train_df, y):
        stats = y.iloc[tr_idx].groupby(key_tr.iloc[tr_idx]).agg(["count", "mean"])
        enc = (stats["mean"] * stats["count"] + prior * smooth) / (stats["count"] + smooth)
        oof.iloc[va_idx] = key_tr.iloc[va_idx].map(enc).fillna(prior).values

    stats_full = y.groupby(key_tr).agg(["count", "mean"])
    enc_full = (stats_full["mean"] * stats_full["count"] + prior * smooth) / (stats_full["count"] + smooth)
    te = key_te.map(enc_full).fillna(prior)
    return oof.astype(float), te.astype(float)

def add_text_flags(df, col, prefix):
    s = df[col].astype(str).str.lower().fillna("")
    return pd.DataFrame({
        f"{prefix}_has_debt":    s.str.contains("debt",   na=False).astype(int),
        f"{prefix}_has_credit":  s.str.contains("credit", na=False).astype(int),
        f"{prefix}_has_consol":  s.str.contains(r"consolida", na=False).astype(int),
        f"{prefix}_len":         s.str.len(),
        f"{prefix}_n_words":     s.str.split().str.len().fillna(0),
        f"{prefix}_has_digits":  s.str.contains(r"\d", regex=True, na=False).astype(int),
    }, index=df.index)

def parse_emp_len(s: pd.Series) -> pd.Series:
    s2 = s.fillna("").astype(str).str.lower()
    is_under1 = s2.str.contains("<", na=False)
    is_ge10   = s2.str.contains(r"10\+", na=False)
    num = pd.to_numeric(s2.str.extract(r"(\d+)")[0], errors="coerce")
    val = np.where(is_under1, 0.5, np.where(is_ge10, 10.0, num))
    return pd.Series(val, index=s.index, dtype=float)

def map_emp_sector(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower().fillna("")
    def pick(t):
        if any(k in t for k in ["student","retired","unemploy","homemaker"]): return "NONWORK"
        if any(k in t for k in ["self","owner","entrepreneur","freelance","contractor","consultant"]): return "SELFEMP"
        if any(k in t for k in ["nurse","doctor","md","physician","medical","health","clinic","hospital","pharma","dental","rn","lpn","therap"]): return "HEALTH"
        if any(k in t for k in ["teacher","professor","education","school","university","principal","instructor"]): return "EDU"
        if any(k in t for k in ["government","federal","state","county","city","public","police","sheriff","fire","military","navy","army","air force","marines"]): return "GOV"
        if any(k in t for k in ["bank","finance","cpa","account","loan","credit","mortgage","insurance","broker"]): return "FINANCE"
        if any(k in t for k in ["engineer","developer","software","programmer","it","tech","data","scientist"]): return "TECH"
        if any(k in t for k in ["sales","rep","associate","retail","store","cashier","customer service","csr","marketing","real estate","agent"]): return "SALES"
        if any(k in t for k in ["driver","operator","construction","mechanic","labor","warehouse","manufact","factory","maintenance","electrician","plumber","carpenter","truck"]): return "BLUE"
        if any(k in t for k in ["manager","director","supervisor","lead","vp","chief","ceo","coo","cfo","executive"]): return "MGMT"
        return "OTHER"
    return x.map(pick).astype("category")

# -------------------- Features --------------------
def make_features(train_df: pd.DataFrame, test_df: pd.DataFrame, n_folds: int, seed: int = 42):
    tr = train_df.copy()
    te = test_df.copy()
    y = tr[TARGET].astype(int)

    tr["term_num"] = pd.to_numeric(tr["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0)
    te["term_num"] = pd.to_numeric(te["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0)

    tr["emp_len_num"]     = parse_emp_len(tr["emp_length"]).fillna(0)
    te["emp_len_num"]     = parse_emp_len(te["emp_length"]).fillna(0)
    tr["emp_len_under1"]  = tr["emp_length"].astype(str).str.contains("<", na=False).astype(int)
    te["emp_len_under1"]  = te["emp_length"].astype(str).str.contains("<", na=False).astype(int)
    tr["emp_len_ge10"]    = tr["emp_length"].astype(str).str.contains("10", na=False).astype(int)
    te["emp_len_ge10"]    = te["emp_length"].astype(str).str.contains("10", na=False).astype(int)
    tr["emp_title_missing"] = (tr["emp_title"].astype(str).str.strip()=="").astype(int) | tr["emp_title"].isna().astype(int)
    te["emp_title_missing"] = (te["emp_title"].astype(str).str.strip()=="").astype(int) | te["emp_title"].isna().astype(int)

    grade_map = {g:i for i,g in enumerate(list("ABCDEFG"), start=1)}
    tr["grade_ord"] = tr["grade"].map(grade_map).fillna(0).astype(int)
    te["grade_ord"] = te["grade"].map(grade_map).fillna(0).astype(int)
    tr["grade_cat"] = tr["grade"].astype("category")
    te["grade_cat"] = te["grade"].astype("category")

    tr["purpose3"] = bucket3_from_text(tr["purpose"]).astype("category")
    te["purpose3"] = bucket3_from_text(te["purpose"]).astype("category")
    tr["title3"]   = bucket3_from_text(tr["title"]).astype("category")
    te["title3"]   = bucket3_from_text(te["title"]).astype("category")

    tr["installment_per_loan"] = safe_div(tr["installment"], tr["loan_amnt"])
    te["installment_per_loan"] = safe_div(te["installment"], te["loan_amnt"])
    tr["dti_inst"] = safe_div(tr["installment"], tr["annual_inc"]/12.0)
    te["dti_inst"] = safe_div(te["installment"], te["annual_inc"]/12.0)
    tr["dti_amt"]  = safe_div(tr["loan_amnt"], tr["annual_inc"])
    te["dti_amt"]  = safe_div(te["loan_amnt"], te["annual_inc"])
    tr["int_total"] = tr["loan_amnt"] * (tr["int_rate"]/100.0) * (tr["term_num"]/12.0)
    te["int_total"] = te["loan_amnt"] * (te["int_rate"]/100.0) * (te["term_num"]/12.0)
    tr["log_int_total"] = np.log1p(tr["int_total"])
    te["log_int_total"] = np.log1p(te["int_total"])

    st = tr.groupby("addr_state").agg(
        n=("addr_state","count"),
        inc_mean=("annual_inc","mean"),
        int_mean=("int_rate","mean"),
        loan_mean=("loan_amnt","mean"),
    )
    title_low = tr["title"].astype(str).str.lower()
    st["debt_share"]   = title_low.str.contains("debt",   na=False).groupby(tr["addr_state"]).mean()
    st["credit_share"] = title_low.str.contains("credit", na=False).groupby(tr["addr_state"]).mean()
    st = st.fillna(st.median(numeric_only=True))

    tr = tr.merge(st, left_on="addr_state", right_index=True, how="left", suffixes=("","_st"))
    te = te.merge(st, left_on="addr_state", right_index=True, how="left", suffixes=("","_st"))
    tr.rename(columns={"n":"state_log_count","inc_mean":"state_inc_mean","int_mean":"state_int_mean",
                       "loan_mean":"state_loan_mean","debt_share":"state_debt_share","credit_share":"state_credit_share"}, inplace=True)
    te.rename(columns={"n":"state_log_count","inc_mean":"state_inc_mean","int_mean":"state_int_mean",
                       "loan_mean":"state_loan_mean","debt_share":"state_debt_share","credit_share":"state_credit_share"}, inplace=True)
    tr["state_log_count"] = np.log1p(tr["state_log_count"])
    te["state_log_count"] = np.log1p(te["state_log_count"])
    tr["inc_vs_state"]    = safe_div(tr["annual_inc"], tr["state_inc_mean"])
    te["inc_vs_state"]    = safe_div(te["annual_inc"], te["state_inc_mean"])
    tr["int_diff_state"]  = tr["int_rate"] - tr["state_int_mean"]
    te["int_diff_state"]  = te["int_rate"] - te["state_int_mean"]

    vs_map = {"Not Verified":0, "Verified":1, "Source Verified":2}
    tr["verification_ord"] = tr["verification_status"].map(vs_map).fillna(1).astype(int)
    te["verification_ord"] = te["verification_status"].map(vs_map).fillna(1).astype(int)

    ho_map = {"RENT":0, "MORTGAGE":1, "OWN":2}
    tr["home_own_ord"] = tr["home_ownership"].map(ho_map).fillna(1).astype(int)
    te["home_own_ord"] = te["home_ownership"].map(ho_map).fillna(1).astype(int)

    vs_counts = pd.concat([tr["verification_status"], te["verification_status"]]).astype(str).value_counts()
    ho_counts = pd.concat([tr["home_ownership"],    te["home_ownership"]]).astype(str).value_counts()
    tr["verification_freq_log"] = np.log1p(tr["verification_status"].astype(str).map(vs_counts).fillna(0))
    te["verification_freq_log"] = np.log1p(te["verification_status"].astype(str).map(vs_counts).fillna(0))
    tr["home_own_freq_log"]    = np.log1p(tr["home_ownership"].astype(str).map(ho_counts).fillna(0))
    te["home_own_freq_log"]    = np.log1p(te["home_ownership"].astype(str).map(ho_counts).fillna(0))

    tr["emp_sector"] = map_emp_sector(tr["emp_title"])
    te["emp_sector"] = map_emp_sector(te["emp_title"])
    tr["emp_sector_te"], te["emp_sector_te"] = kfold_target_encode(tr, te, "emp_sector", y, n_splits=n_folds, smooth=1200, seed=seed, suffix="te")

    etf_tr = add_text_flags(tr, "emp_title", "emp_title")
    etf_te = add_text_flags(te, "emp_title", "emp_title")

    tr["vs_grade_te"], te["vs_grade_te"] = kfold_target_encode(tr, te, ["verification_status","grade"], y, n_splits=n_folds, smooth=800, seed=seed, suffix="te")
    tr["ho_grade_te"], te["ho_grade_te"] = kfold_target_encode(tr, te, ["home_ownership","grade"],      y, n_splits=n_folds, smooth=800, seed=seed, suffix="te")

    title_counts = pd.concat([tr["title"], te["title"]]).astype(str).str.lower().value_counts()
    emp_counts   = pd.concat([tr["emp_title"], te["emp_title"]]).astype(str).str.lower().value_counts()

    gt = tr.groupby([tr["grade"].astype(str), tr["term"].astype(str)])["int_rate"].agg(["mean","std"])
    gt["std"] = gt["std"].replace(0, 1e-6)
    key_tr = list(zip(tr["grade"].astype(str), tr["term"].astype(str)))
    key_te = list(zip(te["grade"].astype(str), te["term"].astype(str)))

    X = tr.drop(columns=[TARGET, ID_COL]).copy()
    T = te.drop(columns=[ID_COL]).copy()

    X = pd.concat([X, add_text_flags(tr, "purpose", "purpose")], axis=1)
    T = pd.concat([T, add_text_flags(te, "purpose", "purpose")], axis=1)
    X = pd.concat([X, etf_tr], axis=1)
    T = pd.concat([T, etf_te], axis=1)

    X["title_freq_log"]     = np.log1p(tr["title"].astype(str).str.lower().map(title_counts).fillna(0))
    T["title_freq_log"]     = np.log1p(te["title"].astype(str).str.lower().map(title_counts).fillna(0))
    X["emp_title_freq_log"] = np.log1p(tr["emp_title"].astype(str).str.lower().map(emp_counts).fillna(0))
    T["emp_title_freq_log"] = np.log1p(te["emp_title"].astype(str).str.lower().map(emp_counts).fillna(0))

    X["int_rate_z_gt"] = (tr["int_rate"].values - pd.Series(key_tr).map(gt["mean"]).values) / pd.Series(key_tr).map(gt["std"]).values
    T["int_rate_z_gt"] = (te["int_rate"].values - pd.Series(key_te).map(gt["mean"]).values) / pd.Series(key_te).map(gt["std"]).values

    for c in ["annual_inc", "dti_amt", "dti_inst", "int_rate"]:
        lo, hi = np.percentile(X[c].astype(float), [1, 99])
        X[c] = X[c].clip(lo, hi); T[c] = T[c].clip(lo, hi)

    drop_raw = ["term", "emp_length", "purpose", "title", "emp_title", "grade"]
    X = X.drop(columns=[c for c in drop_raw if c in X.columns], errors="ignore")
    T = T.drop(columns=[c for c in drop_raw if c in T.columns], errors="ignore")

    cat_cols = ["grade_cat", "home_ownership", "verification_status", "addr_state", "purpose3", "title3", "emp_sector"]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
            T[c] = T[c].astype("category")
            if X[c].isna().any() or T[c].isna().any():
                X[c] = X[c].cat.add_categories(["__M__"]).fillna("__M__")
                T[c] = T[c].cat.add_categories(["__M__"]).fillna("__M__")

    for df in (X, T):
        for c in df.select_dtypes(include="object").columns.tolist():
            df[c] = df[c].astype("category")

    for df in (X, T):
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols): df[num_cols] = df[num_cols].fillna(0)

    features = X.columns.tolist()
    cat_features = [c for c in cat_cols if c in X.columns]
    return X, T, y, features, cat_features, tr

# -------------------- CV helpers --------------------
def build_strata_for_multidim(train_df):
    g = train_df["grade"].astype(str).fillna("G")
    t = pd.to_numeric(train_df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").fillna(0).astype(int)
    q = pd.qcut(train_df["int_rate"], q=4, labels=False, duplicates="drop").fillna(0).astype(int)
    return (g + "_" + t.astype(str) + "_" + q.astype(str)).astype(str)

def iter_cv_splits(X, y, train_df, seed, fold_strategy, n_folds):
    if fold_strategy == "skf10":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yield from skf.split(X, y)
    elif fold_strategy == "sgkf_state":
        if StratifiedGroupKFold is None:
            raise RuntimeError("StratifiedGroupKFold is unavailable (scikit-learn>=1.1 required).")
        groups = train_df["addr_state"].astype(str)
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yield from sgkf.split(X, y, groups=groups)
    elif fold_strategy == "strata":
        strata = build_strata_for_multidim(train_df)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        yield from skf.split(X, strata)
    else:
        raise ValueError(f"Unknown FOLD_STRATEGY: {fold_strategy}")

# -------------------- IO helpers --------------------
def find_csv(basename: str, data_dir: Path) -> Path:
    candidates = [
        data_dir / f"{basename}.csv",
        data_dir / basename / f"{basename}.csv",
        Path(f"{basename}.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"{basename}.csv not found under {data_dir} or current dir.")

# -------------------- Train --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--fold-strategy", type=str, default=os.getenv("FOLD_STRATEGY","skf10"))
    parser.add_argument("--n-folds", type=int, default=int(os.getenv("N_FOLDS","5")))
    parser.add_argument("--seeds", type=str, default=os.getenv("SEEDS","42,101,202,303,404"))
    parser.add_argument("--threads", type=int, default=int(os.getenv("N_THREADS","-1")))
    parser.add_argument("--fast", type=int, default=int(os.getenv("FAST","1")))  # 1: 時間安全側
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = find_csv("train", data_dir)
    test_path  = find_csv("test",  data_dir)
    has_gt = (data_dir / "ground_truth.csv").exists() or Path("ground_truth.csv").exists()
    gt_path = data_dir / "ground_truth.csv" if (data_dir / "ground_truth.csv").exists() else Path("ground_truth.csv")

    # fastモードなら seed を1本化（fold=5のまま）。本番タイムアウト回避用。
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.fast:
        seeds = [seeds[0]]

    print(f"[CFG] data_dir={data_dir}  FOLD={args.fold_strategy}  N_FOLDS={args.n_folds}  SEEDS={seeds}  threads={args.threads}  fast={args.fast}")

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    gt = pd.read_csv(gt_path) if has_gt else None

    X, Xtest, y, feat_names, cat_feats, tr_full = make_features(train, test, n_folds=args.n_folds, seed=42)

    pred = np.zeros(len(test)); oof = np.zeros(len(train))
    imp_gain_sum  = np.zeros(len(feat_names), dtype=float)
    imp_split_sum = np.zeros(len(feat_names), dtype=float)

    for seed in seeds:
        pos_ratio = y.mean()
        neg_pos = (1 - pos_ratio) / max(pos_ratio, 1e-9)

        params = dict(
            objective="binary", metric="auc", boosting_type="gbdt",
            learning_rate=0.035, num_leaves=192, min_child_samples=80,
            max_bin=255, feature_fraction=0.75, bagging_fraction=0.75, bagging_freq=1,
            lambda_l2=60.0, is_unbalance=False, scale_pos_weight=float(neg_pos),
            min_data_per_group=100, cat_smooth=20, cat_l2=20,
            num_threads=args.threads, seed=seed, verbosity=-1,
        )

        for tr_i, va_i in iter_cv_splits(X, y, tr_full, seed, args.fold_strategy, args.n_folds):
            dtr = lgb.Dataset(X.iloc[tr_i], label=y.iloc[tr_i],
                              categorical_feature=cat_feats, free_raw_data=False)
            dva = lgb.Dataset(X.iloc[va_i], label=y.iloc[va_i],
                              categorical_feature=cat_feats, reference=dtr, free_raw_data=False)

            model = lgb.train(
                params, dtr, num_boost_round=6000,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(300), lgb.log_evaluation(0)]
            )

            oof[va_i] += model.predict(X.iloc[va_i], num_iteration=model.best_iteration) / len(seeds)
            pred      += model.predict(Xtest,       num_iteration=model.best_iteration) / (len(seeds)*args.n_folds)

            imp_gain_sum  += model.feature_importance("gain")
            imp_split_sum += model.feature_importance("split")

    print(f"[OOF]  AUC: {roc_auc_score(y, oof):.6f}")
    if has_gt and ("default" in gt.columns):
        print(f"[TEST] AUC (probabilities): {roc_auc_score(gt['default'].astype(int), pred):.6f}  n={len(test)}")

    write_submission(test[ID_COL], pred, "submission.csv")
    print("Saved: submission.csv")

    imp_df = pd.DataFrame({"feature": feat_names, "gain": imp_gain_sum, "split": imp_split_sum}).sort_values(["gain","split"], ascending=False)
    imp_df.to_csv("feature_importance.csv", index=False)
    print("Saved: feature_importance.csv")
    print("\n[Feature Importance - top 30 by gain]")
    print(imp_df.head(30).to_string(index=False))

if __name__ == "__main__":
    main()

