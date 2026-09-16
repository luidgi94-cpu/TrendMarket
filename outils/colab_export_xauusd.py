# ============================================================
#  RECUPERATION DES DONNEES XAUUSD M5  -  a coller dans Google Colab
#  Fonctionne depuis un navigateur de telephone.
#  1. Ouvre  colab.research.google.com  -> Nouveau notebook
#  2. Colle TOUT ce bloc dans la cellule
#  3. Appuie sur le bouton lecture, attends 1 a 2 minutes
#  4. Le fichier se telecharge automatiquement
#  5. Envoie-le dans la conversation
# ============================================================

!pip install -q dukascopy-python yfinance

import pandas as pd, datetime as dt

MOIS = 12          # profondeur d'historique souhaitee
SORTIE = "XAUUSD_M5.csv"

df = None

# --- Source 1 : Dukascopy (la meilleure : plusieurs annees + volume) -----
try:
    import dukascopy_python
    from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
    fin = dt.datetime.now()
    debut = fin - dt.timedelta(days=30 * MOIS)
    df = dukascopy_python.fetch(
        INSTRUMENT_FX_METALS_XAU_USD,
        dukascopy_python.INTERVAL_MIN_5,
        dukascopy_python.OFFER_SIDE_BID,
        debut, fin,
    )
    print(f"Dukascopy OK : {len(df)} bougies")
except Exception as e:
    print("Dukascopy indisponible :", e)

# --- Source 2 : Yahoo Finance (limite a 60 jours en M5) ------------------
if df is None or len(df) < 500:
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="60d", interval="5m",
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        print(f"Yahoo OK : {len(df)} bougies (limite a 60 jours)")
    except Exception as e:
        print("Yahoo indisponible :", e)

if df is None or df.empty:
    raise SystemExit("Aucune source disponible. Previens-moi et on prend le plan B.")

# --- Normalisation : colonnes attendues par le backtest ------------------
df.columns = [str(c).lower() for c in df.columns]
ren = {"o":"open","h":"high","l":"low","c":"close","v":"volume",
       "adj close":"adj_close"}
df = df.rename(columns=ren)
cols = [c for c in ("open","high","low","close","volume") if c in df.columns]
df = df[cols].dropna()
df.index.name = "timestamp"

df.to_csv(SORTIE)
print(f"\n{len(df)} bougies  |  du {df.index[0]}  au {df.index[-1]}")
print("colonnes :", list(df.columns))
print("volume present :", "volume" in df.columns)

from google.colab import files
files.download(SORTIE)
