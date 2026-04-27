# Experimental Neural Uplift Models

This module contains experimental implementations of neural network–based approaches for uplift modeling.

⚠️ **Important:**
This is a **research / playground module**. The code here is not production-ready, not fully validated, and may change or be removed in the future.

---

## Overview

The goal of this module was to explore neural network architectures for estimating individual treatment effects (uplift), as an alternative to tree-based and meta-learning approaches implemented in the main library.

The experiments include:

* **DragonNet**

  * A neural architecture designed for causal inference.
  * Jointly models outcome and treatment assignment.
  * Inspired by:

    > *Dragonnet: Learning to Estimate Treatment Effects in Observational Data*

* **QiniDeep**

  * Experimental attempt to directly optimize uplift-related metrics (e.g., Qini curve / uplift ranking).
  * Focus on improving ranking quality rather than point estimation.

* **SMITE**

  * Prototype approach for structured modeling of treatment effects.
  * Explores alternative representations of treatment-response interactions.

---

## Purpose of This Module

This module was created to:

* Test whether neural networks can outperform tree-based uplift models
* Explore differentiable approximations of uplift metrics
* Experiment with end-to-end uplift learning pipelines

---

## Current Status

* 🚧 Incomplete
* 🧪 Experimental
* ❗ Not fully benchmarked
* ❗ API is unstable
* ❗ Not integrated with the main `upninja` pipeline

---

## Why It Is Kept in the Repository

Although not production-ready, the module is kept because:

* it documents research directions explored during development
* it may be useful as a starting point for future work
* it provides reference implementations of neural uplift ideas

---

## When to Use (and Not Use)

**Use this module if:**

* you want to experiment with neural uplift models
* you are comfortable modifying and debugging research code

**Do NOT use this module if:**

* you need stable and reliable results
* you are building production systems
* you expect sklearn-like API consistency

---

## Future Work (Potential)

* unify API with the rest of the library
* integrate with PyTorch training loops
* add proper evaluation pipelines
* benchmark against tree-based methods
* implement regularization for uplift-specific objectives

---

## Notes

This module intentionally lives outside the core API surface of the library.
The main, stable functionality of the package is implemented in:

* tree-based uplift models
* DML-based approaches
* tuning utilities

---

## Disclaimer

This code is provided **as-is**, without guarantees of correctness, stability, or performance.
