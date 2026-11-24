"""VWAP de session et ses bandes d'ecart-type, moteur independant.

POURQUOI UN MOTEUR SEPARE
Le moteur d'Order Block anticipait : il calculait ses zones avec le prix de
cloture d'une bougie de detection, puis cherchait son entree a partir du
DEBUT de cette meme bougie, parce qu'un resample pandas etiquette ses
bougies par leur bord gauche. Cinquante-huit minutes d'avenir en H1. Plutot
que de reutiliser ce moteur, celui-ci est ecrit a part, sans resample et
sans unite de detection : tout se calcule bougie M1 par bougie M1, en
cumul depuis le debut de la seance. Un cumul est causal par construction,
l'erreur precedente ne peut donc pas s'y reproduire.

CE QUI EST MESURE
La VWAP de seance et ses bandes a plus ou moins k ecarts-types. Deux usages
opposes sont testes, parce qu'ils ne peuvent pas etre vrais tous les deux :

  "retour"   vendre la bande haute, acheter la bande basse, en pariant sur
             le retour vers la moyenne. C'est l'usage qu'en font les
             formateurs, et celui que montre la capture : zone de vente en
             haut, zone d'achat en bas.
  "cassure"  l'inverse exactement : acheter la bande haute, vendre la basse.

TROIS PRECAUTIONS
Les bandes sont lues avec UN RETARD D'UNE BOUGIE. Sans cela une bougie large
elargirait sa propre bande et la toucherait par construction.

Le poids est reglable. Sur l'or au comptant il n'existe pas de vrai volume :
MT5 publie un nombre de TICKS, c'est-a-dire de mises a jour de prix du
courtier, et non des contrats echanges. Une VWAP calculee la-dessus depend
donc du courtier. Le mode "plat" donne le meme poids a chaque minute et sert
de temoin : si les deux donnent le meme resultat, la ponderation ne sert a
rien et le mot VWAP est usurpe.

L'entree est un ordre LIMITE pose sur la bande : il est rempli des que le
prix l'atteint, sans condition sur la suite de la bougie. Le stop est teste
des la bougie d'entree, et avant la cible a l'interieur de chaque bougie.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VwapConfig:
    spread_usd: float = 0.15
    slippage_usd: float = 0.10
    units: float = 1.0

    sens: str = "retour"          # "retour" | "cassure"
    k: float = 2.0                # ecarts-types de la bande
    poids: str = "tickvol"        # "tickvol" | "plat"

    # Echauffement : nombre de minutes de seance avant d'autoriser un trade.
    # En debut de seance l'ecart-type est minuscule et les bandes collent au
    # prix, ce qui produirait des signaux sans contenu.
    echauffement: int = 60
    # Stop, en ecarts-types de la bande. La cible est la VWAP elle-meme, donc
    # distante de k ecarts-types : le rapport cible sur risque vaut k/stop.
    stop_sigma: float = 0.5
    cible: str = "vwap"           # "vwap" | "r"
    cible_r: float = 2.0

    max_trades_jour: int = 10
    duree_max_min: int = 480
    # Heures UTC autorisees. La plage par defaut couvre Londres et New York.
    heures: tuple[int, int] = (6, 21)


def _bandes(m1: pd.DataFrame, cfg: VwapConfig):
    """VWAP cumulee de seance et son ecart-type, decales d'une bougie."""
    tp = ((m1.high + m1.low + m1.close) / 3.0).to_numpy()
    if cfg.poids == "tickvol" and "tickvol" in m1.columns:
        v = m1["tickvol"].to_numpy().astype(float)
        v = np.where(v > 0, v, 1.0)
    else:
        v = np.ones(len(m1))

    jour = m1.index.normalize()
    # Remise a zero du cumul a chaque changement de jour.
    nouveau = np.empty(len(m1), bool)
    nouveau[0] = True
    nouveau[1:] = jour[1:] != jour[:-1]
    grp = np.cumsum(nouveau) - 1

    df = pd.DataFrame({"pv": v * tp, "pv2": v * tp * tp, "v": v, "g": grp})
    cum = df.groupby("g").cumsum()
    vwap = cum.pv.to_numpy() / cum.v.to_numpy()
    var = cum.pv2.to_numpy() / cum.v.to_numpy() - vwap * vwap
    sig = np.sqrt(np.maximum(var, 0.0))
    # Rang de la bougie dans sa seance, pour l'echauffement.
    rang = np.arange(len(m1)) - np.maximum.accumulate(
        np.where(nouveau, np.arange(len(m1)), 0))

    # UN RETARD D'UNE BOUGIE : la bande utilisee a la bougie j est celle
    # calculee a la fermeture de j-1. Decalage applique a l'interieur de
    # chaque seance, la premiere bougie du jour etant mise hors service.
    vwap_d = np.roll(vwap, 1); sig_d = np.roll(sig, 1)
    vwap_d[0] = np.nan; sig_d[0] = np.nan
    vwap_d[nouveau] = np.nan; sig_d[nouveau] = np.nan
    return vwap_d, sig_d, rang, jour


def backtest_vwap(m1: pd.DataFrame, cfg: VwapConfig):
    vwap, sig, rang, jour = _bandes(m1, cfg)
    o = m1.open.to_numpy(); h = m1.high.to_numpy()
    l = m1.low.to_numpy(); c = m1.close.to_numpy()
    heures = m1.index.hour.to_numpy()
    n = len(m1)

    ouvert = (heures >= cfg.heures[0]) & (heures < cfg.heures[1]) \
        & (rang >= cfg.echauffement) & np.isfinite(sig) & (sig > 0)

    trades = []
    par_jour = {}
    j = 1
    while j < n - 1:
        if not ouvert[j]:
            j += 1; continue
        d_jour = jour[j]
        if par_jour.get(d_jour, 0) >= cfg.max_trades_jour:
            j += 1; continue

        haute = vwap[j] + cfg.k * sig[j]
        basse = vwap[j] - cfg.k * sig[j]

        # Quelle bande a ete atteinte, et dans quel sens on la joue.
        d = 0
        niveau = np.nan
        if h[j] >= haute:
            d = -1 if cfg.sens == "retour" else 1
            niveau = haute
        elif l[j] <= basse:
            d = 1 if cfg.sens == "retour" else -1
            niveau = basse
        if d == 0:
            j += 1; continue

        # Remplissage sur la bande. Si la bougie ouvre deja au-dela, c'est
        # l'ouverture qui sert, des deux cotes : pour une vente sur la bande
        # haute cela donne un meilleur prix, pour un achat sur cette meme
        # bande un moins bon. Les deux sont corrects, et aucun des deux ne
        # suppose quoi que ce soit sur la suite de la bougie.
        entree = max(niveau, o[j]) if niveau == haute else min(niveau, o[j])

        risque = cfg.stop_sigma * sig[j]
        if risque <= 0:
            j += 1; continue
        stop = entree - d * risque
        if cfg.cible == "vwap":
            cible = vwap[j]
            # Une cible du mauvais cote n'a pas de sens : on passe.
            if (cible - entree) * d <= 0:
                j += 1; continue
        else:
            cible = entree + d * cfg.cible_r * risque
        cible_r = (cible - entree) * d / risque

        # --- gestion, bougie par bougie, stop teste en premier -----------
        # La cible n'est PAS testee sur la bougie d'entree : a l'interieur
        # d'une bougie l'ordre des evenements est inconnu, et compter un gain
        # qu'on n'a pas vu arriver serait le genre d'optimisme qui a fausse
        # le moteur precedent. Le stop, lui, est bien teste des l'entree.
        fin = j
        sortie, pourquoi = 0.0, "fin_fenetre"
        if (l[j] <= stop) if d > 0 else (h[j] >= stop):
            sortie, pourquoi = -1.0, "stop"
        else:
            touche = False
            for t in range(j + 1, min(j + cfg.duree_max_min, n)):
                fin = t
                if (l[t] <= stop) if d > 0 else (h[t] >= stop):
                    sortie, pourquoi, touche = -1.0, "stop", True
                    break
                if (h[t] >= cible) if d > 0 else (l[t] <= cible):
                    sortie, pourquoi, touche = cible_r, "cible", True
                    break
            if not touche:
                sortie = (c[fin] - entree) * d / risque

        cout = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = sortie * risque * cfg.units - cout
        par_jour[d_jour] = par_jour.get(d_jour, 0) + 1
        trades.append({"ts": m1.index[j], "date": d_jour,
                       "sens": "long" if d > 0 else "short",
                       "entree": entree, "stop_dist": risque,
                       "cible_r": cible_r, "R": pnl / (risque * cfg.units),
                       "pnl": pnl, "sortie": pourquoi, "i": j, "i_fin": fin})
        # On repart APRES la sortie : pas de positions qui se chevauchent,
        # sans quoi le nombre de trades surestimerait celui des paris
        # reellement independants.
        j = fin + 1

    df = pd.DataFrame(trades)
    nj = pd.Index(jour).nunique()
    st = {"jours": nj, "nb_trades": len(df),
          "trades_par_jour": len(df) / nj if nj else 0.0}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "stop_median": df.stop_dist.median(),
               "cible_r_median": df.cible_r.median()}
    return df, st


@dataclass
class RetestConfig(VwapConfig):
    """Cassure de bande, puis RETOUR sur la bande, pris a l'ordre limite.

    POURQUOI CETTE FORME, ET PAS UNE AUTRE
    Trois mesures la designent, et c'est la premiere fois qu'elles sont
    reunies.

      La bande de VWAP est l'evenement qui porte le plus d'information de
      tout ce projet : la jouer en retour a la moyenne perd 0.165 R de plus
      que le hasard, avec un t de -4.3, sur les deux moities et pour toutes
      les largeurs de bande essayees. Rien d'autre n'a produit un effet
      aussi gros ni aussi stable.

      Le sens qui gagne est donc l'autre : aller AVEC le mouvement donne
      +0.104 R face au temoin, positif sur les deux moities.

      Mais l'entrer a la cassure coute 0.198 R avant toute lecture du
      marche, parce qu'un ordre stop remplit APRES le depart du mouvement.
      L'ordre limite, lui, a un temoin a peu pres nul.

    D'ou : declencher sur la bande, aller dans le sens du mouvement, et
    entrer a l'ordre LIMITE sur le retour, et non a l'ordre stop sur la
    cassure.
    """
    retrace_sigma: float = 0.5    # retour attendu, en ecarts-types
    validite_min: int = 120       # duree de vie de l'ordre, en minutes


def backtest_retest(m1: pd.DataFrame, cfg: RetestConfig):
    vwap, sig, rang, jour = _bandes(m1, cfg)
    o = m1.open.to_numpy(); h = m1.high.to_numpy()
    l = m1.low.to_numpy(); c = m1.close.to_numpy()
    heures = m1.index.hour.to_numpy()
    n = len(m1)

    ouvert = (heures >= cfg.heures[0]) & (heures < cfg.heures[1]) \
        & (rang >= cfg.echauffement) & np.isfinite(sig) & (sig > 0)

    trades = []
    par_jour = {}
    j = 1
    while j < n - 1:
        if not ouvert[j]:
            j += 1; continue
        jr = jour[j]
        if par_jour.get(jr, 0) >= cfg.max_trades_jour:
            j += 1; continue

        haute = vwap[j] + cfg.k * sig[j]
        basse = vwap[j] - cfg.k * sig[j]
        # Le mouvement atteint une bande : on va DANS son sens.
        if h[j] >= haute:
            d, bande = 1, haute
        elif l[j] <= basse:
            d, bande = -1, basse
        else:
            j += 1; continue

        # Ordre limite pose en RETRAIT de la bande, du cote d'ou vient le
        # prix. On achete donc un repli apres une cassure haussiere.
        limite = bande - d * cfg.retrace_sigma * sig[j]
        fin_ordre = min(j + cfg.validite_min, n - 1)
        touche = -1
        for t in range(j + 1, fin_ordre):
            if (l[t] <= limite) if d > 0 else (h[t] >= limite):
                touche = t; break
        if touche < 0:
            # UN EVENEMENT, UN ORDRE. Reprendre a j+1 reposerait un ordre a
            # chaque bougie tant que le prix reste au-dela de la bande, et
            # le moteur garderait le premier qui se remplit : le choix de
            # l'ordre retenu dependrait alors du futur. Mesure du defaut sur
            # marche aleatoire, ou l'esperance doit etre nulle : +0.049 R
            # avec un ordre valable quinze minutes, +0.085 a trente,
            # +0.203 a cent vingt, +0.286 a quatre cent quatre-vingts. Le
            # biais grandit avec la duree de vie de l'ordre, ce qui le
            # designe sans ambiguite. On saute donc la fenetre entiere.
            j = fin_ordre; continue

        entree = min(limite, o[touche]) if d > 0 else max(limite, o[touche])
        risque = cfg.stop_sigma * sig[j] + abs(entree - limite)
        if risque <= 0:
            j = touche + 1; continue
        stop = entree - d * risque
        cible = entree + d * cfg.cible_r * risque
        cible_r = (cible - entree) * d / risque

        fin = touche
        sortie, pourquoi = 0.0, "fin_fenetre"
        if (l[touche] <= stop) if d > 0 else (h[touche] >= stop):
            sortie, pourquoi = -1.0, "stop"
        else:
            vu = False
            for t in range(touche + 1, min(touche + cfg.duree_max_min, n)):
                fin = t
                if (l[t] <= stop) if d > 0 else (h[t] >= stop):
                    sortie, pourquoi, vu = -1.0, "stop", True
                    break
                if (h[t] >= cible) if d > 0 else (l[t] <= cible):
                    sortie, pourquoi, vu = cible_r, "cible", True
                    break
            if not vu:
                sortie = (c[fin] - entree) * d / risque

        cout = (cfg.spread_usd + cfg.slippage_usd) * cfg.units
        pnl = sortie * risque * cfg.units - cout
        par_jour[jr] = par_jour.get(jr, 0) + 1
        trades.append({"ts": m1.index[touche], "date": jr,
                       "sens": "long" if d > 0 else "short",
                       "entree": entree, "stop_dist": risque,
                       "cible_r": cible_r, "R": pnl / (risque * cfg.units),
                       "pnl": pnl, "sortie": pourquoi, "i": touche,
                       "i_fin": fin})
        j = fin + 1

    df = pd.DataFrame(trades)
    nj = pd.Index(jour).nunique()
    st = {"jours": nj, "nb_trades": len(df),
          "trades_par_jour": len(df) / nj if nj else 0.0}
    if not df.empty:
        st |= {"winrate": (df.pnl > 0).mean(), "R_moyen": df.R.mean(),
               "stop_median": df.stop_dist.median(),
               "cible_r_median": df.cible_r.median()}
    return df, st
