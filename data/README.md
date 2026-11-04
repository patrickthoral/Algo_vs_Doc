<img src="https://github.com/patrickthoral/Algo_vs_Doc/blob/main/img/logo_c4i_square.svg?raw=1" alt="Amsterdam UMC Center for Critical Care Computational Intelligence" width=128px><img src="https://github.com/patrickthoral/Algo_vs_Doc/blob/main/img/logo_amds.svg?raw=1" alt="Amsterdam Medical Data Science" width=128px/><img src="https://github.com/patrickthoral/Algo_vs_Doc/blob/main/img/logo_olvg.svg?raw=1" alt="OLVG" width=128px/><img src="https://github.com/patrickthoral/Algo_vs_Doc/blob/main/img/logo_lumc.svg?raw=1" alt="Leiden University Medical Center" width=128px/><img src="https://github.com/patrickthoral/Algo_vs_Doc/blob/main/img/logo_pacmed.svg?raw=1" alt="Pacmed" width=128px/>

# Algo_vs_Doc

# `data` folder

`1-create-analysis-dataset.ipynb` writes the flat analysis files here:

- `LUMC/algo_doc_dataset.csv`
- `VUmc/algo_doc_dataset.csv`

The later analysis notebooks load these CSV files through `algo_vs_doc.prediction_data`.

The notebooks access these files through local paths. Data files in this directory are excluded
from Git by [`.gitignore`](../.gitignore).
