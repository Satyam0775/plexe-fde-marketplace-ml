# Plexe FDE – ML for a Marketplace  
## Low Review Prediction System

---

## 1. Business Context

The marketplace is experiencing:

- Seller dissatisfaction  
- Margin pressure  
- Negative customer reviews  
- Lack of clarity on operational focus  

From the customer conversation:

> "Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus."

The objective of this project is to identify a high-impact ML problem and build a production-ready solution that directly improves business outcomes.

---

## 2. Selected ML Problem

### Predict Low Review Scores (≤ 2 Stars)

**Target Definition:**

```
low_rating = 1 if review_score <= 2 else 0
```

### Why This Problem?

- Directly aligned with seller dissatisfaction  
- Measurable and actionable  
- Enables proactive operational intervention  
- Impacts both seller retention and margins  

### Why Not Other Problems?

- **Revenue prediction** → Does not directly address dissatisfaction  
- **Customer churn** → Not directly measurable from dataset  
- **Delivery delay prediction** → Operational signal, not final outcome  

Low review prediction provides the strongest business leverage.

---

## 3. Data & Feature Engineering

Dataset: Brazilian E-Commerce Public Dataset (Olist)

### Key Features Used

- `delivery_delay_days`
- `price`
- `freight_value`
- `seller_historical_risk` (computed from past data only)

### Methodology

- Chronological 80/20 split to simulate real-world deployment
- Seller historical risk computed using training data only (prevents leakage)
- Median imputation using training statistics
- Class imbalance handled via `class_weight="balanced"`

---

## 4. Modeling

### Models Tested

- Logistic Regression (baseline)
- Random Forest (final model)

### Final Model

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)
```

---

## 5. Evaluation Results
| Metric     | Value |
|------------|--------|
| ROC-AUC    | ~0.65  |
| Recall     | ~0.46  |
| Precision  | ~0.21  |
| F1 Score   | ~0.29  |

The model captures approximately **46% of low-rating orders before the review is posted**.
Threshold tuning was explored to increase recall (~69%), but resulted in lower precision. The default threshold (0.5) was selected for practical balance.

### Error Analysis
Confusion Matrix:

```
[[13754  4054]
 [ 1247  1059]]
```

Observations:
- Moderate predictive power (ROC-AUC ~0.65)
- Some dissatisfaction drivers not captured (e.g., product quality)
- False positives acceptable if intervention cost is low

### Practical Limitations
- No NLP analysis of review text
- No product category features
- Seller risk not rolling-window based
- Model should be used as a risk flag, not an automated decision system

---

## 6. Deployment
Model is served via **FastAPI**.

### Run API

```bash
uvicorn src.api.serve:app --reload
```

Visit:
```
http://127.0.0.1:8000/docs
```

### API Example
**Request**

```json
{
  "delivery_delay_days": -5,
  "price": 200,
  "freight_value": 25,
  "seller_historical_risk": 0.18
}
```

**Response**

```json
{
  "prediction": 1,
  "probability": 0.50,
  "top_contributing_features": [
    ["delivery_delay_days", 0.41],
    ["seller_historical_risk", 0.33]
  ]
}
```

Feature importance is derived from the trained Random Forest model.
---

### Sample cURL Request

You can test the API directly from terminal:

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "delivery_delay_days": -5,
  "price": 200,
  "freight_value": 25,
  "seller_historical_risk": 0.18
}'
```

---

## 7. Project Structure

```
├── notebooks/
├── src/
├── models/
├── experiments/
├── ai_chat_logs/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 8. Docker

### Build

```bash
docker build -t marketplace-ml .
```

### Run

```bash
docker run -p 8000:8000 marketplace-ml
```

---

## 9. AI Usage
AI tools (ChatGPT) were used for:
- Structuring the ML pipeline
- Debugging pandas and sklearn issues
- Refining production-style code structure
- Iterative improvement of evaluation and deployment components

All AI-generated outputs were reviewed, tested, and corrected before integration.

Full conversation logs are included in `ai_chat_logs/`.

---

## 10. Future Improvements
- Add SHAP-based explanations
- Include review text embeddings
- Add product category and customer history features
- Implement rolling-window seller risk
- Add monitoring and retraining pipeline

---

This project demonstrates end-to-end ML problem framing, modeling, evaluation, and production-style deployment aligned with real marketplace business needs.
