# Project Summary

## Objective
Predict international football match outcomes and use the trained model for:
- single match prediction
- interactive app usage
- tournament simulation

## Models compared
- Logistic Regression
- Random Forest
- XGBoost

## Why this version is stronger
- compares multiple models instead of one
- uses a fair time-based split
- includes explainability outputs
- selects the best model automatically
- provides a cleaner app interface
- extends simulation from only champion prediction to multi-stage tournament probabilities

## Selection rule
The final model is selected using:
1. highest balanced accuracy
2. highest accuracy
3. lowest log loss

This helps because football draws are harder to predict and plain accuracy alone can be misleading.
