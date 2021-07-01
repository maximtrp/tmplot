# tmplot

[![Documentation Status](https://readthedocs.org/projects/tmplot/badge/?version=latest)](https://tmplot.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/tmplot)](https://pepy.tech/project/tmplot)
![PyPI](https://img.shields.io/pypi/v/tmplot)

**tmplot** is a Python package for visualizing topic modeling results. It provides the interactive report interface that borrows much from LDAvis/pyLDAvis and builds upon it offering a number of metrics for calculating topics distances and a number of algorithms for calculating scatter coordinates of topics.

![Plots](https://raw.githubusercontent.com/maximtrp/tmplot/main/images/topics_terms_plots.png)

## Features

* Supported models:

  * [tomotopy](https://bab2min.github.io/tomotopy/): `LDAModel`, `LLDAModel`, `CTModel`, `DMRModel`, `HDPModel`, `PTModel`, `SLDAModel`, `GDMRModel`
  * [gensim](https://radimrehurek.com/gensim/): `LdaModel`, `LdaMulticore`
  * [bitermplus](https://github.com/maximtrp/bitermplus): `BTM`

* Supported distance metrics:

  * Kullback-Leibler (symmetric and non-symmetric) divergence
  * Jenson-Shannon divergence
  * Jeffrey's divergence
  * Hellinger distance
  * Bhattacharyya distance
  * Total variation distance
  * Jaccard inversed index

* Supported [algorithms](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold) for calculating topics scatter coordinates:

  * t-SNE
  * SpectralEmbedding
  * MDS
  * LocallyLinearEmbedding
  * Isomap

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

* `numpy`
* `scipy`
* `scikit-learn`
* `pandas`
* `altair`
* `ipywidgets`
* `tomotopy`, `gensim`, and `bitermplus`

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
