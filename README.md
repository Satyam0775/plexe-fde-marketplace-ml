# Plexe FDE – ML for a Marketplace  
## Low Review Prediction System

---

## 1. Business Context

The marketplace is experiencing:
- Seller dissatisfaction  
- Margin pressure  
- Negative customer reviews  
- Lack of clarity on where to focus operational improvements  

From the customer conversation:

> "Sellers are complaining, buyers leave bad reviews, and we don't really know where to focus."

The objective of this project is to identify an ML problem that creates measurable business value and build a production-ready solution.

---

## 2. Selected ML Problem

### Predict Low Review Scores (≤ 2 Stars)

Target:
low_rating = 1 if review_score <= 2 else 0


### Why This Problem?
- Directly aligned with stated business pain
- Measurable outcome
- Enables proactive intervention
- Impacts both seller satisfaction and margins

### Why Not Other Problems?

- **Revenue prediction** → Does not directly address dissatisfaction.
- **Customer churn** → Not directly measurable from available dataset.
- **Delivery delay prediction** → Operational signal, but not the actual business outcome.
  
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

- Chronological 80/20 split to prevent data leakage
- Seller risk computed using historical training data only
- Median imputation (train statistics only)
- Class imbalance handled using `class_weight="balanced"`

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
5. Evaluation Results
Metric	Value
ROC-AUC	~0.65
Recall	~0.46
Precision	~0.21
F1 Score	~0.29

The model captures approximately 46% of low-rating orders before review is posted.
Threshold tuning was explored to increase recall (~69%) but resulted in reduced precision. The default threshold (0.5) was selected for practical balance.

6. Error Analysis
Confusion Matrix:
[[13754  4054]
 [ 1247  1059]]

Observations:
~46% of low-rating cases detected
Some dissatisfaction drivers not captured (e.g., product quality)
False positives may be acceptable if intervention cost is low
Limitations:
No NLP analysis of review text
No product category features
Seller risk not rolling-window based

7. Deployment
Model is served via FastAPI.

Run API
uvicorn src.api.serve:app --reload

Visit:
http://127.0.0.1:8000/docs
API Example

Request:
{
  "delivery_delay_days": -5,
  "price": 200,
  "freight_value": 25,
  "seller_historical_risk": 0.18
}

Response:
{
  "prediction": 1,
  "probability": 0.50,
  "top_contributing_features": [
    ["delivery_delay_days", 0.41],
    ["seller_historical_risk", 0.33]
  ]
}

Feature importance is derived from the trained Random Forest model.

8. Project Structure
├── notebooks/
├── src/
├── models/
├── experiments/
├── ai_chat_logs/
├── Dockerfile
├── requirements.txt
└── README.md
9. Docker

Build:
docker build -t marketplace-ml .

Run:
docker run -p 8000:8000 marketplace-ml
10. AI Usage

AI tools (ChatGPT) were used for:
Structuring the ML pipeline
Debugging pandas and sklearn issues
Refining
Improvig production code structure
All AI outputs were reviewed, tested, and corrected before integration.

Full conversation logs are included in ai_chat_logs/.

11. Future Improvements
Add SHAP-based explanations
Include review text embeddings
Add product category features
Implement r
Add monitoring & model retraining pipeline
This project demonstrates end-to-end problem framing, ML modeling, evaluation, and production-style deployment aligned with real business needs.

## AI Usage
AI tools (ChatGPT) were used for iterative debugging, code structuring, and refinement. All outputs were reviewed and validated before integration.
