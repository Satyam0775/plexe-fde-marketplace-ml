import pandas as pd
from src.config import DATA_PATH

def load_raw_data():
    orders = pd.read_csv(DATA_PATH + "olist_orders_dataset.csv")
    reviews = pd.read_csv(DATA_PATH + "olist_order_reviews_dataset.csv")
    order_items = pd.read_csv(DATA_PATH + "olist_order_items_dataset.csv")

    return orders, reviews, order_items