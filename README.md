# BrainDynamics protocol

This protocol is based on our method: **Varga, L.** et al. Brain dynamics supported by a hierarchy of complex correlation patterns defining a robust functional architecture. *Cell Systems*, Volume **15**, Issue **8**, 770 - 786.e5 (2024). DOI: https://doi.org/10.1016/j.cels.2024.07.003

## Requirements

- Python (tested for 3.12 and later versions)
    - igraph (0.11.6 or later)
    - pandas (2.2.2 or later)
    - scipy (1.14.0 or later)
    - matplotlib (3.9.2 or later)
    - cliffs_delta (1.0.0 or later)

- C/C++ compiler (e.g. gcc on Linux, Xcode Developer Tools for MacOS or Visual C++ Build Tools on Windows)

## Setting up

### 1) Set up virtual python environment using python or conda.

For example:

```
python -m venv braindynamics_env
```

OR via Anaconda:

```
conda create --name braindynamics_env python=3.x
```

### 2) Install package from PyPi

```
 pip install braindynamics-starprotocol
```

Link to the PyPi project site for release history: https://pypi.org/project/braindynamics-starprotocol
