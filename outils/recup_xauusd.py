"""Recupere l'historique XAUUSD M1 chez Dukascopy et le met en cache.

POURQUOI CET OUTIL EXISTE
L'export MT5 du courtier a ete efface lors d'une reconstruction du
conteneur, le repertoire temporaire etant vide a cette occasion. Un fichier
de trente-six megaoctets n'a pas sa place dans le depot : c'est donc le
MOYEN de le reconstituer qui y est range, et non le fichier.

CE QUE CES DONNEES VALENT, ET CE QU'ELLES NE VALENT PAS
Dukascopy publie le cours ACHETEUR d'un courtier suisse. Il convient pour
chercher une structure de marche, qui ne depend pas du courtier. Il ne
remplace pas l'export du courtier de l'utilisateur sur deux points :

  le SPREAD n'y figure pas. Les moteurs continuent donc d'utiliser la valeur
  mesuree sur l'export MT5, mediane 0.150 USD l'once sur 588 732 minutes.

  l'HEURE SERVEUR differe. Les horodatages sont en UTC ici, alors que le
  courtier de l'utilisateur publie en UTC+3. Toute plage horaire doit donc
  etre convertie avant comparaison.

Un resultat trouve ici devra etre reverifie sur l'export du courtier avant
d'engager quoi que ce soit.

    python outils/recup_xauusd.py [debut] [fin] [sortie]
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
import time
from pathlib import Path

import pandas as pd

DEBUT = dt.datetime(2025, 1, 1)
FIN = dt.datetime(2026, 11, 24)
SORTIE = Path("/tmp/claude-0/-home-user-TrendMarket/"
              "a4352d0c-f34e-50c7-9977-cff71ea32a24/scratchpad/mt5/"
              "XAUUSD_M1_dukascopy.csv")
PAS_JOURS = 20          # taille des tranches demandees
PAUSE = 1.0             # secondes entre deux tranches, le serveur limitant


def recupere(debut: dt.datetime, fin: dt.datetime) -> pd.DataFrame:
    import dukascopy_python as d
    from dukascopy_python.instruments import I_XAU_USD
    logging.getLogger("dukascopy").setLevel(logging.WARNING)

    morceaux = []
    cur = debut
    n = 0
    while cur < fin:
        suivant = min(cur + dt.timedelta(days=PAS_JOURS), fin)
        for essai in range(4):
            try:
                c = d.fetch_candles(I_XAU_USD, 1, d.TimeUnit.MINUTE,
                                    d.OfferSide.BID, cur, suivant)
                df = d.to_dataframe(c)
                if len(df):
                    morceaux.append(df[["open", "high", "low", "close",
                                        "volume"]])
                n += len(df)
                break
            except Exception as e:                      # noqa: BLE001
                if essai == 3:
                    print(f"  {cur:%Y-%m-%d} : abandon ({type(e).__name__})",
                          flush=True)
                else:
                    time.sleep(2 ** essai)
        print(f"  {cur:%Y-%m-%d} -> {suivant:%Y-%m-%d} : {n:>8,} bougies",
              flush=True)
        cur = suivant
        time.sleep(PAUSE)

    if not morceaux:
        return pd.DataFrame()
    out = pd.concat(morceaux)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def main(argv: list[str]) -> None:
    debut = dt.datetime.fromisoformat(argv[0]) if len(argv) > 0 else DEBUT
    fin = dt.datetime.fromisoformat(argv[1]) if len(argv) > 1 else FIN
    sortie = Path(argv[2]) if len(argv) > 2 else SORTIE
    print(f"XAUUSD M1, du {debut:%d/%m/%Y} au {fin:%d/%m/%Y}", flush=True)
    t0 = time.time()
    df = recupere(debut, fin)
    if df.empty:
        print("Aucune donnee recuperee.")
        sys.exit(1)
    sortie.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(sortie)
    jours = df.index.normalize().nunique()
    print(f"\n{len(df):,} bougies M1 | {jours} jours | "
          f"du {df.index[0]:%d/%m/%Y} au {df.index[-1]:%d/%m/%Y}")
    print(f"prix {df.close.iloc[0]:.2f} -> {df.close.iloc[-1]:.2f}")
    print(f"amplitude mediane d'une bougie M1 : "
          f"{(df.high - df.low).median():.3f} USD")
    print(f"ecrit dans {sortie}  ({sortie.stat().st_size / 1e6:.1f} Mo, "
          f"{time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1:])
