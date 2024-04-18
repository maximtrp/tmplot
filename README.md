# tmplot

[![Codacy coverage](https://img.shields.io/codacy/coverage/5939b1cf99bc4f9d91de11c0d3ff9e50)](https://app.codacy.com/gh/maximtrp/tmplot/coverage)
[![Codacy grade](https://img.shields.io/codacy/grade/5939b1cf99bc4f9d91de11c0d3ff9e50)](https://app.codacy.com/gh/maximtrp/tmplot)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/maximtrp/tmplot/python-package.yml?label=tests)](https://github.com/maximtrp/tmplot/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/tmplot/badge/?version=latest)](https://tmplot.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/tmplot)](https://pepy.tech/project/tmplot)
[![PyPI](https://img.shields.io/pypi/v/tmplot)](https://pypi.org/project/tmplot)
[![Issues](https://img.shields.io/github/issues/maximtrp/tmplot.svg)](https://github.com/maximtrp/tmplot/issues)

**tmplot** is a Python package for analysis and visualization of topic modeling results. It provides the interactive report interface that borrows much from LDAvis/pyLDAvis and builds upon it offering a number of metrics for calculating topic distances and a number of algorithms for calculating scatter coordinates of topics. It can be used to select closest and stable topics across multiple models.

![Plots](https://raw.githubusercontent.com/maximtrp/tmplot/main/images/topics_terms_plots.png)

## Features

- Supported models:

  - [tomotopy](https://bab2min.github.io/tomotopy/): `LDAModel`, `LLDAModel`, `CTModel`, `DMRModel`, `HDPModel`, `PTModel`, `SLDAModel`, `GDMRModel`
  - [gensim](https://radimrehurek.com/gensim/): `LdaModel`, `LdaMulticore`
  - [bitermplus](https://github.com/maximtrp/bitermplus): `BTM`

- Supported distance metrics:

  - Kullback-Leibler (symmetric and non-symmetric) divergence
  - Jenson-Shannon divergence
  - Jeffrey's divergence
  - Hellinger distance
  - Bhattacharyya distance
  - Total variation distance
  - Jaccard inversed index

- Supported [algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold) for calculating topics scatter coordinates:

  - t-SNE
  - SpectralEmbedding
  - MDS
  - LocallyLinearEmbedding
  - Isomap

## Donate

If you find this package useful, please consider donating any amount of money. This will help me spend more time on supporting open-source software.

<a href="https://www.buymeacoffee.com/maximtrp" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Installation

The package can be installed from PyPi:

```bash
pip install tmplot
```

Or directly from this repository:

```bash
pip install git+https://github.com/maximtrp/tmplot.git
```

## Dependencies

- `numpy`
- `scipy`
- `scikit-learn`
- `pandas`
- `altair`
- `ipywidgets`
- `tomotopy`, `gensim`, and `bitermplus` (optional)

## Quick example

```python
# Importing packages
import tmplot as tmp
import pickle as pkl
import pandas as pd

# Reading a model from a file
with open('data/model.pkl', 'rb') as file:
    model = pkl.load(file)

# Reading documents from a file
docs = pd.read_csv('data/docs.txt.gz', header=None).values.ravel()

# Plotting topics as a scatter plot
topics_coords = tmp.prepare_coords(model)
tmp.plot_scatter_topics(topics_coords, size_col='size', label_col='label')

# Plotting terms probabilities
terms_probs = tmp.calc_terms_probs_ratio(phi, topic=0, lambda_=1)
tmp.plot_terms(terms_probs)

# Running report interface
tmp.report(model, docs=docs, width=250)
```

You can find more examples in the [tutorial](https://tmplot.readthedocs.io/en/latest/tutorial.html).
