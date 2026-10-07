# specs/06 — Router PLD/MTP hybride

HEAD base 69d2f60. Contraintes respectées: séquentiel, memory-guard (<20 Go libre
= abort; plancher observé ~66 Go), max-tokens 128, ignore_eos, greedy.

## 0. Corrections de la premise de départ (mesuré, pas inventé)

1. **Il n'existe aucun "gate ngram-score 0.010" dans le repo** — PLD est un flag
 opt-in (`--pld`, greedy, `pld_enable`), jamais gaté automatiquement.
2. **Les deux moteurs SONT déjà combinés**: `generate_mtp_depth`
 (session.rs ~601-666) mélange copy-draft (PLD) et chaîne MTP PAR ROUND
 ("copy hits lead the proposal and the MTP chain fills the remainder",
 lossless, full-copy round = chaîne sautée mais restart maintenu §9.3).
3. Le PLD pur (`pld_decode`, generate.rs) est le chemin sans drafter.

## 1. Coexistence

- Mémoire/process: oui — un seul modèle chargé; PLD = zéro poids additionnel,
 MTP = sidecar drafter. Les deux chemins vivent dans le même binaire et la
 même `Session` (caches partagés, l'un OU l'autre par requête).
- **Switching intra-génération (MTP loop)**: SAIN et déjà exercé — chaque run
 auto alterne round chaîne (d2..d6) / pas série (EV=0) / round copy, des
 dizaines de fois par run (logs `[mtp.auto] picks`), avec sorties
 reproductibles rep-à-rep. Test dédié = les runs auto de la table (ci-dessous).
- **Switching inter-moteurs sur la même Session (PLD puis MTP)**: NON SÛR par
 construction — un tour PLD (`Session::generate`) commit des tokens dans les
 caches sans alimentner `mtp_tokens`/`mtp_multi`; un tour MTP ultérieur prime
 la tête sur une historique tronquée (trou token/multi-pairs). Non tenté;
 règle: un seul moteur par session persistante (le routage est par requête).

## 2. Table de décision (27B agnosticeng/Qwen3.8-27B-4bit, ~2k KV, 128 tok,
interleavé ×2 ordre alterné, medians, bench/tools/router_table.sh)

| régime | serial | PLD pur | MTP-auto |
|---|---|---|---|
| écho | 22.6 | **33.5** | 17.3 (PIRE que série) |
| prose | 22.9 | 22.8 (auto-gaté: 2 rounds, 1 hit) | 21.6 |
| mixte | 23.5 | **24.8** | 20.8 |

- PLD pur ne perd JAMAIS: son fallback copy-miss EST un pas série et les
 drafts sont toujours vérifiés (lossless).
- MTP-auto PERD sur écho: chaque round full-copy paie le restart de tête
 (`draft_step`) et les rounds copy sont exclus du pricing EV → EV surestimé
 (EV affiché d2:39 vs réalisé 17.3).
- L'avantage MTP sur prose ne se matérialise pas à ~2k KV; il est documenté à
 long KV (specs/03: auto 41.6 vs série 33.6 à 12.5k).

## 3. Routeur implémenté

- `echo_score` (copy_draft.rs): fraction des positions 4-gram du prompt dont
 le 4-gram est réapparu plus tôt. Séparation FAIBLE sur tokens réels (mesuré:
 écho 0.857, prose 0.707, mixte 0.594 — les 4-grammes de subwords récurrents
 gonflent la prose); tests unitaires sur synthétique (echo>0.9, novel<0.05,
 mixte intermédiaire).
- Règle (`lisa run --router`, `--router-threshold` 0.80 par défaut... voir
 main.rs: PLD si echo ≥ threshold OU prompt < LONG_KV=8192 tokens; sinon
 policy `--depth`). Le seuil ne garde que la branche long-KV (direction
 coûteuse: prose→pld). À court KV, PLD-par-défaut est quasi-optimal.
- Traçage: `[router] echo_score ... -> pld|depth policy`.

## 4. A/B certifié (bench/tools/router_ab.sh; base = `--depth auto` vs
`--router`; ×3 paires interleavées, ordre alterné, même session)

| régime | base (medians) | router | gain | par paire |
|---|---|---|---|---|
| écho | 19.5 | **34.4** | **+76%** | net positif 3/3 |
| prose | 16.7 | 18.7 | +12% | net positif 3/3 |
| mixte | 22.9 | 21.7 | −5% brut = parité (drift thermique) | +0.1/−1.4/+1.7 |

- Note drift: base prose 21.6 (table) → 16.7 (A/B) entre sessions; seules les
 paires même-session comptent (leçon specs/16/27 re-confirmée).
- **Identité de tokens vérifiée**: router≡PLD forcé, texte généré identique
 (écho et mixte) — le routeur ne change que le choix du moteur.
- Objectif "mixte ≥ max(moteurs seuls) par segment": atteint au sens table
 (routeur=PLD=24.8 = meilleur des 3 sur mixte); l'A/B mixte vs MTP-auto est
 une parité dans la fenêtre thermique de la session A/B.

## 5. Vérifications

- Golden 310/310 ×2 (agnosticeng/Qwen3.8-27B-4bit). lib lisa-engine 24/24
 (dont 2 nouveaux tests echo_score). qwen4 golden non runnable sur cette
 machine (ngram.safetensors absent, préexistant).
- RAM: plancher observé ~66 Go libre, cap 120 Go respecté, runs séquentiels.

## 6. Reste

- Routeur côté serve (`lisa-serve` lit cfg.pld/depth process-wide; exposer le
 même routage par requête en respectant la contrainte une-session-un-moteur).
- Branche long-KV non re-mesurée ici (seuil 0.80 garde specs/03 comme
 référence: MTP 41.6 vs PLD≈série sur prose 12.5k; à re-valider par un A/B
 dédié long avant de durcir la règle).
- Étendre pld_decode à capturer les lignes multi (forward_capture) pour
 rendre sûr un escalade PLD→MTP intra-génération (aujourd'hui interdit).
