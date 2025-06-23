import pandas as pd, numpy as np, joblib, os, warnings
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# 1. Wczytaj i oczyść
# ------------------------------------------------------------------------------
df = pd.read_parquet("all_anomalies_combined.parquet", engine="pyarrow")
for c in ["start_time", "end_time", "time_stamp"]:
    df[c] = pd.to_datetime(df[c])

df = df.dropna(subset=["ship_type"]).reset_index(drop=True)

# bierzemy wyłącznie punkty z pewną etykietą
df = df[df["is_anomaly"].notna()]
df["y_true"] = df["is_anomaly"].astype(int)

# ------------------------------------------------------------------------------
# 2. route_id = start_port
# ------------------------------------------------------------------------------
df["route_id"] = df["start_port"]
routes = df["route_id"].unique()

# ------------------------------------------------------------------------------
# 3. Funkcje pomocnicze
# ------------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------------------
# 4. Przygotuj słownik współrzędnych portów (do stref)
# ------------------------------------------------------------------------------
port_coords = (
    df.groupby("start_port")
    .first()[["start_latitude", "start_longitude"]]
    .to_dict("index")
)
R_PORT, R_APP = 5.0, 15.0


def zone_label(r):
    dmin = min(
        haversine(r.latitude, r.longitude, p["start_latitude"], p["start_longitude"])
        for p in port_coords.values()
    )
    if dmin < R_PORT:
        return "port"
    if dmin < R_APP:
        return "approach"
    return "open_sea"


# ------------------------------------------------------------------------------
# 5. Wylicz zmiany (dv, dcourse, ddraft) i strefy
# ------------------------------------------------------------------------------
df = df.sort_values(["trip_id", "time_stamp"])
df["dv"] = df.groupby("trip_id")["speed_over_ground"].diff().abs().fillna(0)
df["dcourse"] = df.groupby("trip_id")["course_over_ground"].diff().abs().fillna(0)
df["ddraft"] = df.groupby("trip_id")["draught"].diff().abs().fillna(0)
df["zone"] = df.apply(zone_label, axis=1)
zone_dummies = pd.get_dummies(df["zone"], prefix="zone")
df = pd.concat([df, zone_dummies], axis=1)

# ------------------------------------------------------------------------------
# 6. Features wspólne dla wszystkich modeli
# ------------------------------------------------------------------------------
BASE_FEATS = [
    "speed_over_ground",
    "dv",
    "dcourse",
    "ddraft",
    "zone_port",
    "zone_approach",
    "zone_open_sea",
]

# ------------------------------------------------------------------------------
# 7. Pętla po trasach
# ------------------------------------------------------------------------------
os.makedirs("models_per_route", exist_ok=True)
dispatcher = {}  # route_id → ścieżka do pkl

for route in reversed(routes):
    print(f"\n================  {route}  ================\n")

    # policz ilości etykiet w tej trasie (wliczając None)
    mask = df['route_id'] == route
    counts = df.loc[mask, 'is_anomaly'].value_counts(dropna=False)
    # odczytaj ręcznie, żeby mieć wyświetlone None jako osobną linię
    n_true  = counts.get(True,  0)
    n_false = counts.get(False, 0)
    n_none  = counts.get(np.nan, 0)  # pandas traktuje None jako NaN
    print(f"  Liczba punktów: total={mask.sum():,}")
    print(f"    True  (anom): {n_true:,}")
    print(f"    False (norm): {n_false:,}")
    print(f"    None  (unkn): {n_none:,}\n")

    mask = df["route_id"] == route
    df_r = df.loc[mask].copy()

    # one-hot route_id (nie używamy innych tras, więc 0/1 dla bieżącego)
    df_r["route_dummy"] = 1.0
    feats = BASE_FEATS + ["route_dummy"]
    X_r = df_r[feats].fillna(0).values
    y_r = df_r["y_true"].values

    # train/test split (10 % normalnych do testu)
    X_norm = X_r[y_r == 0]
    X_ano = X_r[y_r == 1]

    norm_train, norm_test = train_test_split(
        X_norm, test_size=0.10, random_state=42
    )
    X_train = norm_train
    X_test = np.vstack([X_ano, norm_test])
    y_test = np.concatenate([np.ones(len(X_ano)), np.zeros(len(norm_test))])

    # scaling
    scaler = StandardScaler().fit(X_train)
    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    # grid-search ν
    best_auc, best_nu = -np.inf, None
    for nu in (0.1, 0.12): # for nu in (0.005, 0.01, 0.05, 0.1, 0.2):
        m = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(X_tr_s)
        s = -m.decision_function(X_te_s)
        auc = roc_auc_score(y_test, s)
        print(f"nu={nu:<5} → AUC={auc:.3f}")
        if auc > best_auc:
            best_auc, best_nu = auc, nu
    print(f"   → wybrane ν={best_nu},  AUC={best_auc:.3f}")

    # finalny model
    oc = OneClassSVM(kernel="rbf", nu=best_nu, gamma="scale").fit(X_tr_s)
    scores = -oc.decision_function(X_te_s)
    y_pred = (scores > 0).astype(int)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=3))
    print("ROC AUC:", roc_auc_score(y_test, scores), "\n")

    # zapis
    path = f"models_per_route/ocsvm_point_model_{route}.pkl"
    joblib.dump({"model": oc, "scaler": scaler, "features": feats}, path)
    dispatcher[route] = path

# ------------------------------------------------------------------------------
# 8. Zapis dispatchera (słownika route_id→plik modelu)
# ------------------------------------------------------------------------------
joblib.dump(dispatcher, "models_per_route/dispatcher.pkl")
print("\nZapisano wszystkie modele per-route oraz dispatcher.")
