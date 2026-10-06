# Smart Agriculture: Soil, Fertilizer and Crop Data Analysis with Machine Learning

Python code for the data analysis and model comparison in the paper:

> Mishra, B. K., Singh, B., & Panwar, R. (2025). **Smart agriculture using deep learning and FDM for the relation of soil health, fertilizer and ground water.** *International Journal of System Assurance Engineering and Management.* https://doi.org/10.1007/s13198-025-03055-1

## Scope

This repository contains the code behind the results in Section 3 of the paper:

| Paper item | Notebook step |
|---|---|
| Table 1: data from various sources | Steps 3 and 4 |
| Table 2: values of different soil parameters | Step 6 |
| Fig. 3: data visualization of soil parameters | Step 8b |
| Fig. 4: accuracy on the whole data | Step 12 |
| Fig. 5: accuracy on the test data | Step 12 |

Six classical machine-learning classifiers (Logistic Regression, K-Nearest Neighbors, Random Forest, XGBoost, Decision Tree, Gaussian Naive Bayes) predict a suitable crop from soil nutrients, soil pH and weather variables for 22 crops.

**Not included:** the conceptual framework (Figs. 1 and 2) and the symbolic Eqs. (1) to (10) of the paper are not implemented as code here. This repository contains no neural network and no finite-difference solver.

## What is in this repository

| Path | Content |
|---|---|
| `notebooks/crop_recommendation_analysis.ipynb` | The full analysis, step by step, with a description and citations for every step |
| `data/` | Dataset instructions (`data/README.md`) and, if included, a small sample file |
| `src/make_sample_data.py` | Creates a small stratified sample from the full dataset |
| `results/` | Figures and tables written by the notebook |
| `requirements.txt` | Python libraries needed |
| `CITATION.cff` | How to cite this repository |

## Method in brief

* **Data:** 2,200 samples of 22 crops; 13 features (N, P, K, S, Cu, Fe, Mn, Zn, B, pH, temperature, humidity, rainfall). Missing values are filled with the column mean.
* **Exploration:** class counts, distributions, feature-versus-crop charts, box plots, correlation matrix, IQR outlier diagnostic.
* **Models:** six classifiers, each in a pipeline with standardisation of the features.
* **Evaluation:** 80/20 train/test split (`random_state=0`); accuracy, weighted F1 and confusion matrices. A stratified 5-fold cross-validation (imputation inside each fold) is included as an additional robustness check (Step 13).

## Results (Figs. 4 and 5 of the paper)

| Model | Test accuracy (%) | Whole-data accuracy (%) |
|---|---|---|
| Logistic Regression | 97.73 | 98.91 |
| K-Nearest Neighbors (k=3) | 95.23 | 96.91 |
| Random Forest | 100.00 | 100.00 |
| XGBoost | 100.00 | 100.00 |
| Decision Tree (depth 5) | 95.91 | 96.18 |
| Gaussian Naive Bayes | 99.32 | 99.00 |

Test set: 440 samples (20 %). Whole-data accuracy includes the training samples and is optimistic; test accuracy is the relevant figure. Results may differ slightly with other library versions. Running the notebook regenerates every figure and table (`results/model_comparison.csv`, `results/model_comparison_cv.csv`).

## Limitations

* The dataset is small and one dataset is used. High accuracy here does not show the same accuracy on real farms in other regions or seasons.
* In the single-split analysis the mean imputation was done before the split, and k for KNN was inspected on the test set. The cross-validation step removes both effects.
* The models predict a suitable crop from measurements and do not establish causes. Fertilizer decisions should be confirmed with local soil tests and agronomic advice.

## How to run

Requires Python 3.9 or newer.

```bash
git clone https://github.com/bkmnkm/soil-health-deep-learning-fdm.git
cd soil-health-deep-learning-fdm
python -m venv venv
# Windows: venv\Scripts\activate      Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/crop_recommendation_analysis.ipynb
```

Put the dataset in `data/` first (see `data/README.md`), then choose **Kernel > Restart & Run All**.

## Data

See [`data/README.md`](data/README.md) for the column descriptions, the source and the licence.

## Citation

If you use this code, please cite the paper above. GitHub's **Cite this repository** button (from `CITATION.cff`) gives the repository citation.

## License

MIT License. See the `LICENSE` file.

## Contact

Brajesh Kumar Mishra, [LinkedIn profile](https://www.linkedin.com/in/brajesh-mishra-338b21266), [brajeshkumar43685@gmail.com]
