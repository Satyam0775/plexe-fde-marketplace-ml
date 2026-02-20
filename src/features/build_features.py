import pandas as pd


def build_base_dataset(orders, reviews, order_items):
    # Merge base tables
    df = orders.merge(reviews, on="order_id", how="inner")
    df["low_rating"] = (df["review_score"] <= 2).astype(int)

    # ---------------------------
    # Delivery Delay Feature
    # ---------------------------

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["order_delivered_customer_date"] = pd.to_datetime(orders["order_delivered_customer_date"])
    orders["order_estimated_delivery_date"] = pd.to_datetime(orders["order_estimated_delivery_date"])

    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"]
        - orders["order_estimated_delivery_date"]
    ).dt.days

    df = df.merge(
        orders[["order_id", "delivery_delay_days"]],
        on="order_id",
        how="left"
    )

    df["delivery_delay_days"] = df["delivery_delay_days"].clip(-60, 30)

    # ---------------------------
    # Monetary Features
    # ---------------------------

    price_df = (
        order_items.groupby("order_id")[["price", "freight_value"]]
        .sum()
        .reset_index()
    )

    df = df.merge(price_df, on="order_id", how="left")

    # ---------------------------
    # Seller Historical Risk
    # ---------------------------

    # Attach seller_id
    order_seller = order_items[["order_id", "seller_id"]].drop_duplicates()
    df = df.merge(order_seller, on="order_id", how="left")

    # Sort chronologically
    df = df.sort_values("order_purchase_timestamp")

    # Split index for historical risk calculation
    split_index = int(len(df) * 0.8)
    train_part = df.iloc[:split_index]

    seller_risk = (
        train_part.groupby("seller_id")["low_rating"]
        .mean()
        .reset_index()
    )

    seller_risk.columns = ["seller_id", "seller_historical_risk"]

    df = df.merge(seller_risk, on="seller_id", how="left")

    # Fill missing seller risk with global mean
    global_mean = train_part["low_rating"].mean()
    df["seller_historical_risk"] = df["seller_historical_risk"].fillna(global_mean)

    return df