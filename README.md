# 🥷 UpliftNinja

**UpliftNinja(upninja)** is a lightweight Python library for uplift modeling (causal inference for treatment effect estimation), with a focus on practical workflows and extensibility.

---

## Version

**Current version:** `0.0.1`
*(replace with actual version)*

---

## Overview

Uplift modeling aims to estimate the **incremental effect of a treatment** on an outcome, allowing you to answer:

> *Who should we treat to maximize impact?*

This library provides tools for:

* building uplift models
* evaluating uplift performance
* tuning model hyperparameters
* experimenting with different causal approaches

---

## Key Features

* 🌳 **Tree-based uplift models**

  * Custom uplift decision trees
  * DML-based uplift trees and forests

* ⚙️ **Hyperparameter tuning**

  * Built-in integration with `hyperopt`
  * Cross-validated uplift optimization

* 📊 **Visualization utilities**

  * Uplift curves
  * Qini curves
  * Uplift by percentile plots

* 🧪 **Experimental modules**

  * Neural uplift models (research prototypes)
  * Playground for new ideas and approaches

---

## Installation

```bash
pip install upninja
```

Or for development:

```bash
pip install -e .
```

---

## Quick Example

```python
from upninja.trees import UpliftTreeClassifier

model = UpliftTreeClassifier(max_depth=3)

model.fit(X_train, y_train, treatment_train)

uplift = model.predict(X_test)
```

Plot uplift curve:

```python
from upninja.utils import plot_uplift_curve

plot_uplift_curve(y_test, uplift, treatment_test)
```

---

## Project Structure

```text
upninja/
    dml/            # DML-based models (trees, forests)
    trees/          # custom uplift tree implementations
    tune/           # hyperparameter tuning utilities
    utils/          # plotting and evaluation helpers
    experimental/   # research modules (not production-ready)
```

---

## Experimental Modules

The repository includes experimental components (e.g. neural uplift models) located in:

```text
upninja/experimental/
```

⚠️ These modules are:

* not production-ready
* not part of the stable API
* subject to change without notice

---

## Design Principles

* **Simple API** — minimal boilerplate, sklearn-like interface
* **Modular** — easy to extend and experiment
* **Practical** — focused on real-world uplift workflows
* **Transparent** — clear implementations over heavy abstractions

---

## Status

* ✅ Core functionality: usable
* 🚧 Some modules: experimental
* ❗ API may evolve

---

## Contributing

Contributions, ideas, and experiments are welcome.

---

## License
