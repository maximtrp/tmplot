# tmplot

[![Codacy coverage](https://img.shields.io/codacy/coverage/5939b1cf99bc4f9d91de11c0d3ff9e50)](https://app.codacy.com/gh/maximtrp/tmplot/coverage)
[![Codacy grade](https://img.shields.io/codacy/grade/5939b1cf99bc4f9d91de11c0d3ff9e50)](https://app.codacy.com/gh/maximtrp/tmplot)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/maximtrp/tmplot/python-package.yml?label=tests)](https://github.com/maximtrp/tmplot/actions/workflows/python-package.yml)
[![Documentation Status](https://readthedocs.org/projects/tmplot/badge/?version=latest)](https://tmplot.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/tmplot)](https://pepy.tech/project/tmplot)
[![PyPI](https://img.shields.io/pypi/v/tmplot)](https://pypi.org/project/tmplot)
[![Issues](https://img.shields.io/github/issues/maximtrp/tmplot.svg)](https://github.com/maximtrp/tmplot/issues)

**tmplot** is a comprehensive Python package for **topic modeling analysis and visualization**. Built for data scientists and researchers, it provides powerful interactive reports and advanced analytics that extend beyond traditional LDAvis/pyLDAvis capabilities.

**Analyze** • **Visualize** • **Compare** multiple topic models with ease

![Plots](https://raw.githubusercontent.com/maximtrp/tmplot/main/images/topics_terms_plots.png)

## Key Features

### Interactive Visualization

- **Topic scatter plots** with customizable coordinates and sizing
- **Term probability charts** with relevance weighting
- **Document analysis** showing top documents per topic
- **Interactive reports** with real-time parameter adjustment

### Advanced Analytics

- **Topic stability analysis** across multiple model runs
- **Model comparison** with sophisticated distance metrics
- **Saliency calculations** for term importance
- **Entropy metrics** for model optimization

### Model Support

- **[tomotopy](https://bab2min.github.io/tomotopy/)**: `LDAModel`, `LLDAModel`, `CTModel`, `DMRModel`, `HDPModel`, `PTModel`, `SLDAModel`, `GDMRModel`
- **[gensim](https://radimrehurek.com/gensim/)**: `LdaModel`, `LdaMulticore`
- **[bitermplus](https://github.com/maximtrp/bitermplus)**: `BTM`

### Distance Metrics

- **Kullback-Leibler** (symmetric & non-symmetric)
- **Jensen-Shannon divergence**
- **Jeffrey's divergence**
- **Hellinger & Bhattacharyya distances**
- **Total variation distance**
- **Jaccard index**

### Dimensionality Reduction

- **t-SNE** • **SpectralEmbedding** • **MDS**
- **LocallyLinearEmbedding** • **Isomap**

## Donate

If you find this package useful, please consider donating any amount of money. This will help me spend more time on supporting open-source software.

<a href="https://www.buymeacoffee.com/maximtrp" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Quick Start

### Installation

```bash
# From PyPI (recommended)
pip install tmplot

# Development version
pip install git+https://github.com/maximtrp/tmplot.git
```

### Basic Usage

```python
import tmplot as tmp

# Load your topic model and documents
model = your_fitted_model  # tomotopy, gensim, or bitermplus
docs = your_documents

# Create interactive report
tmp.report(model, docs=docs)

# Or create individual visualizations
coords = tmp.prepare_coords(model)
tmp.plot_scatter_topics(coords, size_col='size')
```

## Advanced Examples

### Compare Multiple Models

```python
import tmplot as tmp

# Find stable topics across multiple models
models = [model1, model2, model3, model4]
closest_topics, distances = tmp.get_closest_topics(models)
stable_topics, stable_distances = tmp.get_stable_topics(closest_topics, distances)
```

### Model Optimization

```python
# Calculate entropy for model selection
entropy_score = tmp.entropy(phi_matrix)

# Analyze topic stability
saliency = tmp.get_salient_terms(phi, theta)
```

### Custom Visualizations

```python
# Create topic distance matrix with different metrics
topic_dists = tmp.get_topics_dist(phi, method='jensen-shannon')

# Generate coordinates with custom algorithm
coords = tmp.get_topics_scatter(topic_dists, theta, method='tsne')
tmp.plot_scatter_topics(coords, topic=3)  # Highlight topic 3
```

## Documentation & Examples

- **[Complete Tutorial](https://tmplot.readthedocs.io/en/latest/tutorial.html)** - Step-by-step guide
- **[API Reference](https://tmplot.readthedocs.io/)** - Full documentation
- **[Example Notebooks](https://github.com/maximtrp/tmplot/tree/main/examples)** - Jupyter examples

## Requirements

**Core dependencies:** `numpy`, `scipy`, `scikit-learn`, `pandas`, `altair`, `ipywidgets`
**Optional models:** `tomotopy`, `gensim`, `bitermplus`
