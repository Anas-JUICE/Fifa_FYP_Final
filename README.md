# FIFA World Cup Match Prediction and Tournament Simulation

A clean, modular final-year project for predicting international football match outcomes using machine learning, comparing three models, and simulating a 48-team World Cup tournament.

## Included improvements
- 3-model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Time-based train/test split
- Explainability plots and feature rankings
- Best-model selection
- Processed team profiles for future match prediction
- Improved Streamlit app with team profile comparison and matchup insights
- Tournament simulation with qualification, quarterfinal, semifinal, final, and champion probabilities

## Project structure
```text
Fifa_FYP_Final/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
├── reports/
│   └── figures/
├── src/
│   ├── app/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── predict/
│   ├── simulation/
│   └── utils/
├── requirements.txt
├── README.md
└── run_pipeline.py
```

## Main files
- `run_pipeline.py` -> runs training, evaluation, best-model selection, and simulation
- `src/models/train_logistic.py`
- `src/models/train_random_forest.py`
- `src/models/train_xgboost.py`
- `src/evaluation/evaluate_model.py`
- `src/simulation/tournament_simulation.py`
- `src/app/app.py`

## How to run

### 1) Install requirements
```bash
pip install -r requirements.txt
```

### 2) Run the full pipeline
```bash
python run_pipeline.py
```

### 3) Predict one match
```bash
python src/predict/predict_match.py "Brazil" "France"
```

### 4) Launch the app
```bash
streamlit run src/app/app.py
```

## Best-model selection rule
The project selects the final model using:
1. highest balanced accuracy
2. highest accuracy
3. lowest log loss

This makes the final choice more fair for the draw class, which is usually the hardest football outcome to predict.

## Notes
- The raw dataset is kept in `data/raw/`.
- The processed team profiles are created automatically in `data/processed/`.
- The simulation uses the selected best model by default.
- The World Cup simulation assumes a 48-team format with 12 groups, top 2 from each group, plus the best 8 third-place teams.
