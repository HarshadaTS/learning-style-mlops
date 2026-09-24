# Learning Style Classification — MLOps Upgrade

This repository modernizes the original final-year learning-style classification project into a reproducible ML/MLOps workflow.

## Current phase
- Original dataset and notebook preserved.
- Reproducible preprocessing extracted from the notebook.
- Decision Tree, Random Forest, SVM and KNN evaluated using the same train/test seed.
- The selected model is saved as a single scikit-learn pipeline/artifact.

## Planned MLOps phases
1. MLflow experiment tracking
2. Flask model-serving API
3. Streamlit client
4. Docker containerization
5. Pytest + GitHub Actions CI
6. AWS deployment
7. Basic production monitoring

## Important reproducibility note
The original notebook trained the Random Forest on the full dataset before evaluating it on the held-out test set, which can introduce evaluation leakage. The cleaned pipeline uses a proper train/test split before fitting models. The original notebook also reports SVM experiments; the reported 91% figure should only be used on the resume after reproducing the exact experiment that produced it.
