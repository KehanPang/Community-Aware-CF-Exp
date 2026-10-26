# ComCE

## Overview

ComCE provides community-aware counterfactual explanations for GNN vertex classification. Given a target vertex and a trained GNN, it searches for a small set of edges whose removal changes the model's decision. Communities organize the search, bringing related graph structures into view before the procedure settles on individual edits.

![Exp](https://anonymous.4open.science/r/Community-Aware-CF-Exp-77ED/ComCE.png)

The request flow is implemented in `src/comce/pipeline/explainer.py`.

## Setup

Use Linux or WSL with **Python 3.12 or later**. The pinned environment uses PyTorch 2.6.0 with CUDA 12.4 and PyTorch Geometric 2.7.0.

```bash
python -m pip install -r requirements.txt
cp yaml.template config.yaml
```

## Run Explanations

Run a small population on CPU:

```bash
python scripts/run_comce.py --set execution.device=cpu
```

For GPU execution, use `--set execution.device=cuda:0`. You can also select a target with `--target-id`, supply an ID file through `--population`, or choose a result directory with `--output-dir`. Target IDs must belong to the configured graph, and `--split` should match their population.

## Full Version

* The **full version** of the paper can be accessed at this file: [ComCE.pdf](https://anonymous.4open.science/r/Community-Aware-CF-Exp-77ED/ComCE.pdf)
* The **dataset** are available at this link: [Google Drive](https://drive.google.com/file/d/1l2sSmynIQQOgpdXGIhJf0kTfB7rAQtVm)