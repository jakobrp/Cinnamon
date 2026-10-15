# Cinnamon mit differenziell privater Synthese (DP-GAN, PATE-GAN)

Dieser Fork von [KI-AIM/Cinnamon](https://github.com/KI-AIM/Cinnamon) erweitert das
Synthesemodul um zwei differenziell private Verfahren aus SynthCity und enthält die
Skripte für die Experimente der Bachelorarbeit
„Privacy Gain durch Differential Privacy: Bewertung Deep-Learning basierter
Syntheseverfahren für tabellarische Daten" (Universität Trier, 2026).

## Stand und Reproduzierbarkeit

Grundlage ist Cinnamon in der Version von Commit `7a69ae8c` (17.06.2026).
Alle Module außer der Synthese laufen unverändert aus den Original-Images dieses
Stands, festgelegt per Digest in `docker-compose.yml`:

| Modul | Quelle |
|---|---|
| Platform, Evaluation, Anonymization, Risk-Assessment | Original-Images `7a69ae8c`, per Digest |
| Synthetization | lokaler Build aus `cinnamon-synthetization/` mit DP-GAN und PATE-GAN |

Die Python-Pakete der Synthese sind in `cinnamon-synthetization/requirements.lock.txt`
auf die Versionen der Testumgebung festgelegt (u. a. `synthcity==0.2.11`,
`torch==2.2.2`, `opacus==1.5.3`).

Der Tag `abgabe-stand` markiert den Stand zur Abgabe.

## Änderungen gegenüber Cinnamon

**Synthesemodul (`cinnamon-synthetization/`)**
- `synthetic_tabular_data_generator/algorithms/dpgan.py`: Adapter für SynthCity `dpgan`
- `synthetic_tabular_data_generator/algorithms/pategan.py`: Adapter für SynthCity `pategan`
- `synthetic_tabular_data_generator/synthesizer_config/dpgan.yaml`, `pategan.yaml`: Parameter für die Oberfläche
- `synthesizer_classes.py`: Registrierung beider Verfahren
- `app.py`: Callback-Timeout von 30 s auf 300 s erhöht (große Rückgaben nach langem Training)

**Konfiguration (`docker-compose.yml`)**
- `SPRING.SERVLET.MULTIPART.MAX-REQUEST-SIZE` ergänzt, da große Syntheseergebnisse
  sonst das Upload-Limit der Plattform überschreiten und abgelehnt werden
- Healthcheck der Plattform verlängert
- Images per Digest festgelegt, Synthese als lokaler Build mit Volume-Mount

**Experimente (`simulation/`)**: siehe unten.

## Starten

Voraussetzung: Docker Desktop.

Zuerst die Umgebungsdatei anlegen (falls noch nicht vorhanden):

```powershell
Copy-Item .env.example .env
# in .env setzen:
#   PG_PASSWORD=<beliebig>
#   MAX_FILE_SIZE_BYTES=524288000
```

### Variante A: mit Internet

```powershell
docker compose up -d
```

Die Original-Images werden per Digest von ghcr.io geladen, die Synthese wird
lokal mit festgelegten Paketversionen gebaut (beim ersten Start ca. 20–30 Minuten).

### Variante B: offline

```powershell
docker load -i cinnamon-images-abgabe.tar
docker compose up -d
```

`cinnamon-images-abgabe.tar` enthält alle Images fertig gebaut in der getesteten
Version. Es wird nichts heruntergeladen oder gebaut.

Die Oberfläche ist danach unter http://localhost:8080 erreichbar.

## Experimente

Die Experimente laufen headless über die REST-API von Cinnamon, nicht über die
Oberfläche. Ablauf in drei Schritten:

```bash

# 1. Synthese + Evaluation, ein Verfahren mit einem ε (Cinnamon muss laufen)
#    Config-Pfad relativ zu simulation/
py simulation/sim_anon.py --dataset heart_failure --label pategan_eps5 --config dataset_configs/heart_failure/all-configurations-pategan.yaml --holdout 0.2 --runs 25

# 2. Ergebnisse eines Datensatzes zusammenführen (erkennt alle Label-Ordner automatisch)
py simulation/aggregate_results.py --root simulation/experiment_results/heart_failure

# 3a. Grafiken für einen Datensatz
py simulation/vis_results.py --root simulation/experiment_results/heart_failure

# 3b. Übersicht über alle Datensätze
py simulation/vis_results_overview.py --root simulation/experiment_results
```

Mit `--extras` bzw. `--labels` lassen sich einzelne Varianten gezielt auswählen,
z. B. `--labels dpgan_eps1 pategan_eps1`.

**Ablauf je Run in `sim_anon.py`:** neuer Benutzer und neues Projekt, Datenupload,
Holdout-Split, Synthese direkt auf den Rohdaten (Anonymisierung übersprungen),
technische Evaluation und Risk-Evaluation, Download der Ergebnisse nach
`experiment_results/<dataset>/<label>/{files,tech_eval,risk_eval}`.

**Namenskonvention für Labels:** `<verfahren>_eps<ε>`, Dezimaltrenner als `_`,
z. B. `dpgan_eps0_1` für ε = 0,1. Die Auswertungsskripte lesen Verfahren und ε
aus dem Namen.

**Referenzwerte:** Die Ordner `anon/`, `anon_synth/` und `real_risk/` stammen aus
dem Referenzexperiment (Anonymisierung, Anonymisierung + Synthese, Risiko der
Originaldaten) und wurden pro Datensatz übernommen außer `heart_failure` übernommen.
Aufgrund eines geänderten Datentyps der Zielspalte wurde die Baseline für
`heart_failure` auf derselben Grundlage neu berechnet.


### Metriken

| Metrik | Herkunft | Berechnung |
|---|---|---|
| `overall_resemblance` | technische Evaluation | übernommen |
| `overall_utility` | technische Evaluation | übernommen |
| `combined_utility` | `aggregate_results.py` | (Resemblance + Utility) / 2 |
| `privacy_score` | Risk-Evaluation | 1 − `total_risk_score` |

### Konfiguration

Die Parameter je Datensatz und Verfahren stehen in
`simulation/dataset_configs/<dataset>/all-configurations-{dpgan,pategan}.yaml`.
ε wird pro Lauf dort gesetzt, die Dateien enthalten jeweils den Wert des letzten Laufs.

Holdout: 0,3 für `cardiovascular` und `stroke`, sonst 0,2.

Die Ergebnisdaten (`simulation/experiment_results/`, ca. 1 GB) sind nicht im
Repository enthalten, sondern separat abgelegt.

## Bekannte Einschränkungen


- **Modellspeicherung DP-GAN:** Der Adapter gibt beim Speichern ein leeres
  Modell zurück, weil das serialisierte DP-GAN-Modell in der Oberfläche zu
  Fehlern führte. Die synthetischen Daten sind davon nicht betroffen.