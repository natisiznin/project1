# 🎓 Student Exam Performance Predictor (End-to-End ML Pipeline)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas & Numpy](https://img.shields.io/badge/Pandas_%2F_Numpy-150458?style=for-the-badge&logo=pandas&logoColor=white)
![AWS Elastic Beanstalk](https://img.shields.io/badge/AWS_Elastic_Beanstalk-232F3E?style=for-the-badge&logo=amazon-aws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

An end-to-end Machine Learning pipeline designed to predict student exam performance (math, reading, and writing scores) based on demographic, socioeconomic, and academic background features. Built using a **modular, object-oriented architecture** with custom logging, robust exception handling, and automated hyperparameter tuning, originally engineered for deployment via **AWS Elastic Beanstalk** and containerized with **Docker**.

---

## 💭 The "Why" & Architecture Planning

Anyone can throw `LinearRegression().fit(X, y)` into a Jupyter Notebook and call it a day. But in a real-world production environment, notebooks don't scale. 

The goal of this project was to transition from "experimental data science" to **robust ML Engineering**. That meant building a system where data ingestion, data transformation, model training, and web serving are completely decoupled into reusable, maintainable software components.

### Why a Modular Pipeline over Notebooks?
* **Zero Data Leakage:** By encapsulating preprocessing steps inside a Scikit-Learn `ColumnTransformer` and `Pipeline`, we guarantee that scaling and one-hot encoding parameters are learned *only* from the training set and applied cleanly to validation/inference data.
* **Maintainability:** If the database schema changes or a new categorical feature is added, we only modify `data_transformation.py`, leaving the inference and training APIs completely untouched.
* **Production Readiness:** Custom exception handling (`CustomException`) and structured logging (`logger.py`) track every execution step, making production debugging tracebacks instant and painless.

### End-to-End System Architecture
```text
[ Raw Student Dataset (.csv) ]
             │
             ▼  (src/components/data_ingestion.py)
[ Train / Test Data Split (Artifacts) ]
             │
             ▼  (src/components/data_transformation.py)
[ Scikit-Learn ColumnTransformer (OneHotEncoder + SimpleImputer + StandardScaler) ]
             │
             ├──► Saved Artifact: preprocessor.pkl
             │
             ▼  (src/components/model_trainer.py - GridSearchCV Tuning)
[ Model Evaluation (Random Forest, XGBoost, CatBoost, Linear Regression) ]
             │
             ├──► Saved Artifact: model.pkl (Best Performing Model)
             │
             ▼  (src/pipeline/predict_pipeline.py)
[ Flask / Web Application API ] ──► [ Local Docker Container / AWS Elastic Beanstalk ]
