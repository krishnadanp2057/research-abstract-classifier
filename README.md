# Project-I: Classifying Research Abstracts into Science, Business and Social Science

**Course:** DAC-507 / DAC-515, Project-I (Classification)
**Student:** Krishnandan Pandit
**Programme:** M.Tech Data Science, IIT Roorkee

---

## 1. What this project does

Given the abstract of a peer-reviewed article, the system predicts which broad field the paper belongs to: **Science**, **Business** or **Social Science**. I collected the abstracts from Scopus, cleaned the text, turned it into TF-IDF features and compared six classical classifiers. The best model is wrapped in a small Streamlit app where a user can upload a file of abstracts and download the predicted labels.

## 2. Repository contents

| File | Description |
|---|---|
| `abstracts_dataset.csv` | The labelled dataset (1,500 abstracts, 500 per class) |
| `scopus_science.csv`, `scopus_business.csv`, `scopus_social.csv` | Raw Scopus exports, one per class |
| `merge_scopus.py` | Cleans the raw exports and builds the balanced dataset |
| `classification_notebook.ipynb` | Preprocessing, model training, evaluation and plots |
| `model.joblib` | The trained pipeline (TF-IDF + Logistic Regression) used by the app |
| `metrics.json` | Results table and confusion matrix saved by the notebook; used by the app's Model insights tab |
| `app.py` | Streamlit demo for uploading abstracts and getting predictions |
| `new_test_abstracts.csv` | 15 unseen abstracts (5 per class) for trying the demo |
| `new_test_hard.csv` | 2 deliberately borderline abstracts |
| `requirements.txt` | Python dependencies |

## 3. Dataset

**Source.** Scopus, accessed through the IIT Roorkee institutional login. I ran one advanced search per class and exported the results as CSV, including the title, year and abstract.

**Search queries.** All searches were limited to English-language journal articles published between 2019 and 2024, and results were sorted by citation count so that the exported papers came from many different journals.

```
Science:        SUBJAREA(PHYS OR CHEM OR BIOC OR EART OR AGRI OR MATH)
                AND NOT SUBJAREA(BUSI OR SOCI OR ECON OR PSYC OR ARTS)
Business:       SUBJAREA(BUSI)
                AND NOT SUBJAREA(SOCI OR PHYS OR CHEM OR BIOC OR EART OR AGRI OR MATH)
Social Science: SUBJAREA(SOCI)
                AND NOT SUBJAREA(BUSI OR PHYS OR CHEM OR BIOC OR EART OR AGRI OR MATH)
(each combined with: AND DOCTYPE(ar) AND LANGUAGE(english) AND PUBYEAR > 2018 AND PUBYEAR < 2025)
```

The `AND NOT` parts exclude papers that Scopus tags with a subject area from another class. This keeps the labels cleaner, because a paper tagged both Business and Social Sciences would otherwise be ambiguous.

**Labels.** The label of each paper is the class whose query returned it, which is based on Scopus's own subject classification. I did not label papers by hand.

**Cleaning and balancing** (done in `merge_scopus.py`):
- removed papers with no abstract or with fewer than 50 words
- removed abstracts written mostly in non-English characters
- removed publisher copyright lines and duplicate abstracts
- randomly sampled **500 abstracts per class**, giving **1,500 abstracts in total**, perfectly balanced

## 4. Method

1. **Text preprocessing.** Lower-casing, removing HTML tags, copyright notices, numbers and punctuation, and collapsing whitespace. English stop words are removed during vectorisation.
2. **Features.** TF-IDF on unigrams and bigrams (sublinear term frequency, minimum document frequency of 2, at most 30,000 features).
3. **Train/test split.** Stratified 80/20 split: 1,200 abstracts for training and 300 for testing. Each model was also evaluated with 5-fold cross-validation on the training set.
4. **Models compared.** Logistic Regression, Decision Tree, Multinomial Naive Bayes, k-Nearest Neighbours (cosine distance, k = 7), linear SVM and Random Forest (300 trees).
5. **Metrics.** Accuracy, macro-averaged precision, recall and F1, confusion matrices and a per-class report. Macro averaging gives each class equal weight, which fits a balanced problem.
6. **Interpretation.** The highest-weighted words per class from the Logistic Regression model, plus a look at misclassified test abstracts.

## 5. Results

| Model | CV accuracy | Test accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.846 | **0.870** | 0.871 | 0.870 | **0.870** |
| Naive Bayes | 0.849 | 0.870 | 0.872 | 0.870 | 0.869 |
| SVM (linear) | 0.847 | 0.867 | 0.868 | 0.867 | 0.866 |
| Random Forest | 0.814 | 0.830 | 0.830 | 0.830 | 0.828 |
| kNN | 0.820 | 0.820 | 0.819 | 0.820 | 0.819 |
| Decision Tree | 0.664 | 0.653 | 0.654 | 0.653 | 0.654 |

### Discussion

- **Logistic Regression, Naive Bayes and the linear SVM are essentially tied** at about 87% accuracy. With only 300 test abstracts and a cross-validation standard deviation of about 1.5 points, the differences between these three are within noise, so I do not claim that one is truly better than the others. Logistic Regression had the highest macro F1 and was chosen for the demo because it also gives usable probability estimates.
- **Linear models work well here** because TF-IDF features are sparse and high-dimensional, and the classes are largely separable by vocabulary (for example "catalyst" or "protein" versus "firm" or "customer" versus "community" or "inequality").
- **Random Forest and kNN scored a few points lower.** Neighbour and tree methods tend to struggle in very high-dimensional sparse spaces, where distances become less informative.
- **The single Decision Tree was clearly the weakest** (65%). A single tree tends to overfit to a handful of individual words, and its cross-validation score is also the least stable of the six (standard deviation 0.031).
- **Where the errors come from.** [Add one or two sentences here after looking at the confusion matrices in the notebook, for example which pair of classes is confused most often and why.]

### Hyperparameter tuning

I ran a 5-fold grid search on the training set only (TF-IDF n-gram range, `min_df`, `max_features` and the Logistic Regression `C`, 60 combinations, scored by macro-F1). It did not improve on the default settings: the best configuration (C = 10, word 1-2-grams, 10,000 features, min_df = 2) reached a cross-validated macro-F1 of 0.844, compared with a cross-validated accuracy of 0.846 for the default settings, and a test accuracy of 0.857 against 0.870 for the default. I therefore kept the default model. This suggests that the remaining errors come mostly from label noise and overlapping fields rather than from model settings.

The figures above come from the held-out test set. The saved `model.joblib` was refit on all 1,500 abstracts after evaluation, so scores computed on `abstracts_dataset.csv` itself would be inflated and should not be reported as accuracy.

## 6. The demo app

`app.py` is a Streamlit app with four tabs:

- **Single abstract:** paste an abstract (or click one of the example buttons) and press Classify. The result appears as an animated card with the predicted field and confidence, animated probability bars for all three classes, and the abstract itself with every word coloured by how strongly it pushed the prediction (green = towards the predicted class, red = against). Low-confidence predictions get a warning.
- **Upload a file:** upload a CSV or Excel file, pick the column that holds the abstracts and press Predict all. The app shows summary tiles, a class chart, a confidence histogram and a results table with a confidence bar per row. The table can be filtered by class, searched by text, or limited to low-confidence rows, and the predictions can be downloaded as a CSV. If the file has a `label` column, the app also reports accuracy and draws a confusion matrix for that file.
- **Model insights:** comparison of all six classifiers on any metric, the confusion matrix of the chosen model on the held-out test set, and the most influential words for each class.
- **How it works:** a short explanation of the pipeline and its limits.

The sidebar has a slider for the confidence level below which a prediction is flagged for manual review.

`new_test_abstracts.csv` contains abstracts I wrote that are not part of the training data, so the upload feature can be tried on unseen text. Scores obtained on `abstracts_dataset.csv` itself are inflated, because the saved model was refit on all of it.

## 7. How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python merge_scopus.py            # builds abstracts_dataset.csv from the Scopus exports
jupyter notebook                  # open classification_notebook.ipynb and run all cells
streamlit run app.py              # starts the demo at http://localhost:8501
```

The notebook creates `model.joblib` in its last cells, and the app needs that file in the same folder.

## 8. Limitations

- Labels come from Scopus subject areas, not from reading each paper. Some papers are borderline between fields, so there is label noise.
- Sorting by citations favours well-known papers, so the dataset is not a random sample of all publications.
- The classes are broad. Fields such as psychology, economics and education overlap with Business and Social Science, which limits the achievable accuracy.
- With 1,500 documents and a 300-document test set, small differences between models are not statistically meaningful.
- The model only sees the abstract text, and it can only choose among three classes. It has no way of saying that a paper belongs to none of them.

## 9. Possible improvements

- Collect more abstracts per class and use random rather than citation-based sampling.
- Try word embeddings or a fine-tuned transformer (for example SciBERT) in place of TF-IDF.
- Report confidence intervals (for example by repeated cross-validation) and tune the other five models as well, not only Logistic Regression.
- Add more classes, such as Medicine or Engineering.
# research-abstract-classifier
