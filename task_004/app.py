import streamlit as st
import pandas as pd
import yaml
from decimal import Decimal
import dateparser
from recordlinkage.index import Full
import recordlinkage
import matplotlib.pyplot as plt

# Data loading functions
def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file into a DataFrame."""
    return pd.read_csv(file_path)

def load_parquet_data(file_path: str) -> pd.DataFrame:
    """Load data from a Parquet file into a DataFrame."""
    return pd.read_parquet(file_path, engine='fastparquet')

def load_yaml_data(file_path: str) -> pd.DataFrame:
    """Load data from a YAML file into a DataFrame."""
    with open(file_path, 'r') as file:
        return pd.DataFrame(yaml.safe_load(file))

# Data cleaning and transformation functions
def extract_price(value: str) -> tuple[Decimal, str] | None:
    """Extract numeric price and currency from a string."""
    price = str(value).strip()
    price = price.lower()
    
    euro_symbols = ['€', 'eur', 'euro']
    dollar_symbols = ['$', 'usd', 'dollar']
    if any(sym in price for sym in euro_symbols):
        curr = 'EUR'
    elif any(sym in price for sym in dollar_symbols):
        curr = 'USD'
    else:
        curr = 'UNKNOWN'
        
    for symbol in euro_symbols + dollar_symbols:
        price = price.replace(symbol, '')
        
    price = price.replace('¢', '.')
    price = price.replace(',', '.')
    price = price.replace(' ', '')

    try:
        return (Decimal(price), curr)
    except ValueError:
        return None

# Currency equalization function
def equalize_currency_eur_to_usd(df: pd.DataFrame, exchange_rate: float) -> pd.Series:
    """Convert all prices in EUR to USD using the given exchange rate."""
    def convert(row):
        if row['currency'] == 'EUR':
            row['unit_price'] = round(row['unit_price'] * Decimal(exchange_rate), 2)
            row['currency'] = 'USD'
        return row
    
    return df.apply(convert, axis=1)

# Date parsing function
def parse_df_dates(df: pd.DataFrame, column: str):
    """Parse date strings in the specified column to datetime objects."""
    df[column] = df[column].apply(lambda x: dateparser.parse(x))
    return df

# Extract date from datetime function
def extract_date(df: pd.DataFrame, datetime_column: str, date_column: str):
    """Extract date part from datetime column."""
    df[date_column] = df[datetime_column].dt.date
    return df

# User deduplication function
def find_real_users(df, min_matching_fields=3):
    u = df.set_index('id')
    
    pairs = Full().index(u)
    
    compare = recordlinkage.Compare()
    for col in ['name', 'email', 'phone', 'address']:
        compare.exact(col, col)
    
    features = compare.compute(pairs, u)
    matching_pairs = features[features.sum(axis=1) >= min_matching_fields].index
    
    cluster_list = recordlinkage.ConnectedComponents().compute(matching_pairs)
    
    ids_in_clusters = set()
    for cluster in cluster_list:
        ids_in_clusters.update(cluster.get_level_values(0))
        ids_in_clusters.update(cluster.get_level_values(1))
        
    user_to_cluster = {}
    for cluster in cluster_list:
        cluster_ids = set(cluster.get_level_values(0)) | set(cluster.get_level_values(1))
        for user_id in cluster_ids:
            user_to_cluster[user_id] = cluster_ids
    
    return len(cluster_list) + len(set(u.index) - ids_in_clusters), user_to_cluster

# Find how how many unique sets of authors there are
def count_author_sets(books_df):
    author_sets = books_df[':author'].apply(lambda x: frozenset(x.split(', '))).unique()
    return len(author_sets)

# Find the most popular author
def most_popular_author(books_df, orders_df):
    merged = orders_df.merge(books_df, left_on='book_id', right_on=':id')
    sales = merged.groupby(':author')['quantity'].sum()
    return sales.idxmax()

# Get top user IDs function
def get_top_user_ids(user_claster, top_id):
    if user_claster.get(top_id):
        return list(user_claster.get(top_id))
    else:
        return [top_id]

# Prepare data for analysis function
def prepare_data_for_analysis(dfs):
    dfs["orders_df"][["unit_price", "currency"]] = dfs["orders_df"]["unit_price"].apply(lambda x: pd.Series(extract_price(x)))
    equalize_currency_eur_to_usd(dfs["orders_df"], exchange_rate=1.2)
    dfs["orders_df"]["paid_price"] = dfs["orders_df"]["unit_price"] * dfs["orders_df"]["quantity"]
    dfs["orders_df"]["paid_price"] = dfs["orders_df"]["paid_price"].astype(float)
    parse_df_dates(dfs["orders_df"], "timestamp")
    extract_date(dfs["orders_df"], "timestamp", "date")
    return dfs

# Wrap up function
def wrap_up(dfs):
    dfs = prepare_data_for_analysis(dfs)
    daily_revenue = dfs["orders_df"].groupby('date', as_index=False)['paid_price'].sum()
    top_5_days = daily_revenue.nlargest(5, 'paid_price')
    
    unique_user_count, user_claster = find_real_users(dfs["users_df"], min_matching_fields=3)
    unique_author_sets_count = count_author_sets(dfs["books_df"])
    popular_author = most_popular_author(dfs["books_df"], dfs["orders_df"])
    
    top_id = dfs["orders_df"].groupby('user_id')['paid_price'].sum().idxmax()
    top_customer_ids = get_top_user_ids(user_claster, top_id)
    
    return {
        "top_5_days": top_5_days,
        "unique_user_count": unique_user_count,
        "unique_author_sets_count": unique_author_sets_count,
        "popular_author": popular_author,
        "top_customer_ids": top_customer_ids,
        "daily_revenue": daily_revenue
    }

# Load data for specific dataset
def load_dataset(dataset_num):
    users_df = load_csv_data(f"data/DATA{dataset_num}/users.csv")
    orders_df = load_parquet_data(f"data/DATA{dataset_num}/orders.parquet")
    books_df = load_yaml_data(f"data/DATA{dataset_num}/books.yaml")
    dfs = {"users_df": users_df, "orders_df": orders_df, "books_df": books_df}
    return wrap_up(dfs)

# Streamlit App
st.set_page_config(page_title="Bookstore Analytics Dashboard", layout="wide")
st.title("📊 Bookstore Analytics Dashboard")

# Create tabs for each dataset
tab1, tab2, tab3 = st.tabs(["DATA1", "DATA2", "DATA3"])

def display_dataset(tab, dataset_num):
    with tab:
        try:
            results = load_dataset(dataset_num)
            
            st.header(f"Dataset {dataset_num} Analysis")
            
            # Metrics row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Unique Users", results["unique_user_count"])
            with col2:
                st.metric("Unique Author Sets", results["unique_author_sets_count"])
            with col3:
                st.metric("Most Popular Author", results["popular_author"])
            
            # Top 5 days
            st.subheader("Top 5 Days by Revenue")
            top_5_formatted = results["top_5_days"].copy()
            top_5_formatted['date'] = top_5_formatted['date'].astype(str)
            top_5_formatted['paid_price'] = top_5_formatted['paid_price'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(top_5_formatted, hide_index=True, use_container_width=True)
            
            # Top customer
            st.subheader("Best Buyer (All User IDs)")
            st.write(results["top_customer_ids"])
            
            # Daily revenue chart
            st.subheader("Daily Revenue Chart")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results["daily_revenue"]['date'], results["daily_revenue"]['paid_price'], linewidth=2)
            ax.set_xlabel('Date')
            ax.set_ylabel('Revenue ($)')
            ax.set_title('Daily Revenue Over Time')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading DATA{dataset_num}: {str(e)}")

# Display each dataset in its tab
display_dataset(tab1, 1)
display_dataset(tab2, 2)
display_dataset(tab3, 3)