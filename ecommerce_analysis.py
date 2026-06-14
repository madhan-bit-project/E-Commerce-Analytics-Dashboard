import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

# =========================================================
# VISUALIZATION STYLE
# =========================================================

sns.set_style("whitegrid")

plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

# =========================================================
# DISPLAY SETTINGS
# =========================================================

pd.set_option('display.max_columns', None)

print("Libraries Imported Successfully")

# =========================================================
# LOAD DATASETS
# =========================================================

customers = pd.read_csv('data/olist_customers_dataset.csv')
orders = pd.read_csv('data/olist_orders_dataset.csv')
order_items = pd.read_csv('data/olist_order_items_dataset.csv')
products = pd.read_csv('data/olist_products_dataset.csv')
payments = pd.read_csv('data/olist_order_payments_dataset.csv')
reviews = pd.read_csv('data/olist_order_reviews_dataset.csv')

print("Datasets Loaded Successfully")

# =========================================================
# DATASET SHAPES
# =========================================================

print("\nDataset Shapes")

print("Customers Shape:", customers.shape)
print("Orders Shape:", orders.shape)
print("Order Items Shape:", order_items.shape)
print("Products Shape:", products.shape)
print("Payments Shape:", payments.shape)
print("Reviews Shape:", reviews.shape)

# =========================================================
# DATA PREVIEW
# =========================================================

print("\nCustomers Dataset")
print(customers.head())

print("\nOrders Dataset")
print(orders.head())

print("\nProducts Dataset")
print(products.head())

# =========================================================
# MISSING VALUE ANALYSIS
# =========================================================

print("\nMissing Values in Customers Dataset")
print(customers.isnull().sum())

print("\nMissing Values in Orders Dataset")
print(orders.isnull().sum())

print("\nMissing Values in Products Dataset")
print(products.isnull().sum())

print("\nMissing Values in Payments Dataset")
print(payments.isnull().sum())

print("\nMissing Values in Reviews Dataset")
print(reviews.isnull().sum())

# =========================================================
# DUPLICATE CHECKING
# =========================================================

print("\nDuplicate Rows")

print("Customers:", customers.duplicated().sum())
print("Orders:", orders.duplicated().sum())
print("Products:", products.duplicated().sum())
print("Payments:", payments.duplicated().sum())
print("Reviews:", reviews.duplicated().sum())

# =========================================================
# HANDLE MISSING VALUES - PRODUCTS DATASET
# =========================================================

products['product_category_name'] = products[
    'product_category_name'
].fillna('Unknown')

products['product_name_lenght'] = products[
    'product_name_lenght'
].fillna(
    products['product_name_lenght'].median()
)

products['product_description_lenght'] = products[
    'product_description_lenght'
].fillna(
    products['product_description_lenght'].median()
)

products['product_photos_qty'] = products[
    'product_photos_qty'
].fillna(
    products['product_photos_qty'].median()
)

products['product_weight_g'] = products[
    'product_weight_g'
].fillna(
    products['product_weight_g'].median()
)

products['product_length_cm'] = products[
    'product_length_cm'
].fillna(
    products['product_length_cm'].median()
)

products['product_height_cm'] = products[
    'product_height_cm'
].fillna(
    products['product_height_cm'].median()
)

products['product_width_cm'] = products[
    'product_width_cm'
].fillna(
    products['product_width_cm'].median()
)

print("\nRemaining Missing Values in Products Dataset")
print(products.isnull().sum())

# =========================================================
# DATE CONVERSION
# =========================================================

orders['order_purchase_timestamp'] = pd.to_datetime(
    orders['order_purchase_timestamp']
)

orders['order_approved_at'] = pd.to_datetime(
    orders['order_approved_at']
)

orders['order_delivered_carrier_date'] = pd.to_datetime(
    orders['order_delivered_carrier_date']
)

orders['order_delivered_customer_date'] = pd.to_datetime(
    orders['order_delivered_customer_date']
)

orders['order_estimated_delivery_date'] = pd.to_datetime(
    orders['order_estimated_delivery_date']
)

print("\nDate Columns Converted Successfully")

print("\nOrders Dataset Data Types")
print(orders.dtypes)

# =========================================================
# FEATURE ENGINEERING
# =========================================================

orders['order_year'] = (
    orders['order_purchase_timestamp'].dt.year
)

orders['order_month'] = (
    orders['order_purchase_timestamp'].dt.month
)

orders['order_day'] = (
    orders['order_purchase_timestamp'].dt.day
)

print("\nFeature Columns Created Successfully")

print(
    orders[
        [
            'order_purchase_timestamp',
            'order_year',
            'order_month',
            'order_day'
        ]
    ].head()
)

# =========================================================
# DELIVERY TIME FEATURE
# =========================================================

orders['delivery_time_days'] = (
    orders['order_delivered_customer_date']
    - orders['order_purchase_timestamp']
).dt.days

print("\nDelivery Time Feature Created")

print(
    orders[
        [
            'order_purchase_timestamp',
            'order_delivered_customer_date',
            'delivery_time_days'
        ]
    ].head()
)

# =========================================================
# MONTHLY ORDERS ANALYSIS
# =========================================================

monthly_orders = (
    orders.groupby('order_month')['order_id']
    .count()
)

print("\nMonthly Orders")
print(monthly_orders)

# =========================================================
# MONTHLY ORDERS VISUALIZATION
# =========================================================

plt.figure(figsize=(10,5))

sns.barplot(
    x=monthly_orders.index,
    y=monthly_orders.values,
    hue=monthly_orders.index,
    palette='viridis',
    legend=False
)

plt.title('Monthly Orders Trend')
plt.xlabel('Month')
plt.ylabel('Number of Orders')

plt.tight_layout()

plt.savefig('screenshots/monthly_orders_trend.png')

plt.show()

# =========================================================
# MERGE ORDER ITEMS WITH PRODUCTS
# =========================================================

merged_products = pd.merge(
    order_items,
    products,
    on='product_id',
    how='inner'
)

print("\nMerged Dataset Shape")
print(merged_products.shape)

print("\nMerged Dataset Preview")
print(merged_products.head())

# =========================================================
# TOP PRODUCT CATEGORIES
# =========================================================

top_categories = (
    merged_products['product_category_name']
    .value_counts()
    .head(10)
)

print("\nTop Product Categories")
print(top_categories)

# =========================================================
# TOP PRODUCT CATEGORIES VISUALIZATION
# =========================================================

plt.figure(figsize=(12,6))

sns.barplot(
    x=top_categories.values,
    y=top_categories.index,
    hue=top_categories.index,
    palette='magma',
    legend=False
)

plt.title('Top 10 Product Categories')
plt.xlabel('Number of Products Sold')
plt.ylabel('Product Category')

plt.tight_layout()

plt.savefig('screenshots/top_categories_chart.png')

plt.show()

# =========================================================
# CUSTOMER STATE ANALYSIS
# =========================================================

state_orders = (
    customers['customer_state']
    .value_counts()
    .head(10)
)

print("\nTop Customer States")
print(state_orders)

# =========================================================
# CUSTOMER STATE VISUALIZATION
# =========================================================

plt.figure(figsize=(12,6))

sns.barplot(
    x=state_orders.values,
    y=state_orders.index,
    hue=state_orders.index,
    palette='coolwarm',
    legend=False
)

plt.title('Top 10 Customer States')
plt.xlabel('Number of Customers')
plt.ylabel('State')

plt.tight_layout()

plt.savefig('screenshots/customer_state_analysis.png')

plt.show()

# =========================================================
# TOTAL REVENUE
# =========================================================

total_revenue = payments['payment_value'].sum()

print("\nTotal Revenue")
print(round(total_revenue, 2))

# =========================================================
# MERGE ORDERS WITH PAYMENTS
# =========================================================

revenue_data = pd.merge(
    orders,
    payments,
    on='order_id',
    how='inner'
)

print("\nRevenue Dataset Shape")
print(revenue_data.shape)

# =========================================================
# MONTHLY REVENUE ANALYSIS
# =========================================================

monthly_revenue = (
    revenue_data.groupby('order_month')['payment_value']
    .sum()
)

print("\nMonthly Revenue")
print(monthly_revenue)

# =========================================================
# MONTHLY REVENUE VISUALIZATION
# =========================================================

plt.figure(figsize=(10,5))

sns.lineplot(
    x=monthly_revenue.index,
    y=monthly_revenue.values,
    marker='o',
    color='purple',
    linewidth=3
)

plt.fill_between(
    monthly_revenue.index,
    monthly_revenue.values,
    alpha=0.3,
    color='purple'
)

plt.title('Monthly Revenue Trend')
plt.xlabel('Month')
plt.ylabel('Revenue')

plt.tight_layout()

plt.savefig('screenshots/monthly_revenue_trend.png')

plt.show()

# =========================================================
# DELIVERY PERFORMANCE ANALYSIS
# =========================================================

average_delivery = (
    orders['delivery_time_days']
    .mean()
)

print("\nAverage Delivery Time")
print(round(average_delivery, 2), "days")

fastest_delivery = (
    orders['delivery_time_days']
    .min()
)

slowest_delivery = (
    orders['delivery_time_days']
    .max()
)

print("\nFastest Delivery Time:", fastest_delivery, "days")
print("Slowest Delivery Time:", slowest_delivery, "days")

# =========================================================
# DELIVERY DISTRIBUTION VISUALIZATION
# =========================================================

plt.figure(figsize=(10,5))

sns.histplot(
    orders['delivery_time_days'],
    bins=30,
    kde=True,
    color='teal'
)

plt.title('Delivery Time Distribution')
plt.xlabel('Delivery Time (Days)')
plt.ylabel('Frequency')

plt.tight_layout()

plt.savefig('screenshots/delivery_time_distribution.png')

plt.show()

# =========================================================
# PAYMENT METHOD ANALYSIS
# =========================================================

payment_methods = (
    payments['payment_type']
    .value_counts()
)

print("\nPayment Method Usage")
print(payment_methods)

# =========================================================
# PAYMENT METHOD DONUT CHART
# =========================================================

plt.figure(figsize=(8,8))

colors = sns.color_palette('Set2', len(payment_methods))

wedges, texts, autotexts = plt.pie(
    payment_methods.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'width': 0.4},
    textprops={'fontsize': 11}
)

plt.legend(
    wedges,
    payment_methods.index,
    title="Payment Methods",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.title('Payment Method Distribution')

plt.tight_layout()

plt.savefig('screenshots/payment_method_analysis.png')

plt.show()

# =========================================================
# INSTALLMENT ANALYSIS
# =========================================================

average_installments = (
    payments['payment_installments']
    .mean()
)

print("\nAverage Installments Used")
print(round(average_installments, 2))

# =========================================================
# REVIEW SCORE ANALYSIS
# =========================================================

review_scores = (
    reviews['review_score']
    .value_counts()
    .sort_index()
)

print("\nReview Score Distribution")
print(review_scores)

# =========================================================
# AVERAGE REVIEW SCORE
# =========================================================

average_review = (
    reviews['review_score']
    .mean()
)

print("\nAverage Review Score")
print(round(average_review, 2))

# =========================================================
# REVIEW SCORE DONUT CHART
# =========================================================

plt.figure(figsize=(8,8))

colors = sns.color_palette('viridis', len(review_scores))

wedges, texts, autotexts = plt.pie(
    review_scores.values,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    wedgeprops={'width': 0.4},
    textprops={'fontsize': 11}
)

plt.legend(
    wedges,
    review_scores.index,
    title="Review Scores",
    loc="center left",
    bbox_to_anchor=(1, 0.5)
)

plt.title('Customer Review Score Distribution')

plt.tight_layout()

plt.savefig('screenshots/review_score_donut_chart.png')

plt.show()

# =========================================================
# CORRELATION DATASET
# =========================================================

correlation_data = revenue_data[
    [
        'payment_value',
        'payment_installments',
        'delivery_time_days'
    ]
]

print("\nCorrelation Dataset")
print(correlation_data.head())

# =========================================================
# CORRELATION MATRIX
# =========================================================

correlation_matrix = correlation_data.corr()

print("\nCorrelation Matrix")
print(correlation_matrix)

# =========================================================
# CORRELATION HEATMAP
# =========================================================

plt.figure(figsize=(8,5))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    linewidths=0.5
)

plt.title('Correlation Heatmap')

plt.tight_layout()

plt.savefig('screenshots/correlation_heatmap.png')

plt.show()

# =========================================================
# CREATE SQLITE DATABASE
# =========================================================

connection = sqlite3.connect(
    'ecommerce_database.db'
)

print("\nDatabase Created Successfully")

# =========================================================
# EXPORT DATAFRAMES TO SQL TABLES
# =========================================================

customers.to_sql(
    'customers',
    connection,
    if_exists='replace',
    index=False
)

orders.to_sql(
    'orders',
    connection,
    if_exists='replace',
    index=False
)

products.to_sql(
    'products',
    connection,
    if_exists='replace',
    index=False
)

payments.to_sql(
    'payments',
    connection,
    if_exists='replace',
    index=False
)

reviews.to_sql(
    'reviews',
    connection,
    if_exists='replace',
    index=False
)

print("\nDatasets Exported to SQL Database")

# =========================================================
# SQL QUERY - TOP CUSTOMER STATES
# =========================================================

query = """

SELECT
    customer_state,
    COUNT(customer_id) AS total_customers

FROM customers

GROUP BY customer_state

ORDER BY total_customers DESC

LIMIT 10

"""

top_states_sql = pd.read_sql_query(
    query,
    connection
)

print("\nTop States From SQL")
print(top_states_sql)

# =========================================================
# SQL VISUALIZATION
# =========================================================

plt.figure(figsize=(10,6))

sns.barplot(
    x='total_customers',
    y='customer_state',
    hue='customer_state',
    data=top_states_sql,
    palette='rocket',
    legend=False
)

plt.title('Top Customer States (SQL Analysis)')
plt.xlabel('Total Customers')
plt.ylabel('State')

plt.tight_layout()

plt.savefig('screenshots/sql_customer_state_analysis.png')

plt.show()

# =========================================================
# SQL REVENUE ANALYSIS
# =========================================================

query_revenue = """

SELECT
    strftime('%m', order_purchase_timestamp) AS order_month,
    ROUND(SUM(payment_value), 2) AS total_revenue

FROM orders

INNER JOIN payments
ON orders.order_id = payments.order_id

GROUP BY order_month

ORDER BY order_month

"""

sql_revenue = pd.read_sql_query(
    query_revenue,
    connection
)

print("\nMonthly Revenue From SQL")
print(sql_revenue)

# =========================================================
# SQL REVENUE VISUALIZATION
# =========================================================

plt.figure(figsize=(10,5))

sns.lineplot(
    x='order_month',
    y='total_revenue',
    data=sql_revenue,
    marker='o',
    linewidth=3,
    color='darkorange'
)

plt.fill_between(
    range(len(sql_revenue)),
    sql_revenue['total_revenue'],
    alpha=0.3,
    color='orange'
)

plt.title('Monthly Revenue Trend (SQL Analysis)')
plt.xlabel('Month')
plt.ylabel('Revenue')

plt.tight_layout()

plt.savefig('screenshots/sql_revenue_analysis.png')

plt.show()

# =========================================================
# SQL DELIVERY ANALYSIS
# =========================================================

query_delivery = """

SELECT
    ROUND(
        AVG(
            julianday(order_delivered_customer_date)
            -
            julianday(order_purchase_timestamp)
        ),
        2
    ) AS average_delivery_days

FROM orders

"""

sql_delivery = pd.read_sql_query(
    query_delivery,
    connection
)

print("\nAverage Delivery Time From SQL")
print(sql_delivery)

# =========================================================
# SQL REVIEW ANALYSIS
# =========================================================

query_reviews = """

SELECT
    review_score,
    COUNT(review_id) AS total_reviews

FROM reviews

GROUP BY review_score

ORDER BY review_score

"""

sql_reviews = pd.read_sql_query(
    query_reviews,
    connection
)

print("\nReview Analysis From SQL")
print(sql_reviews)

# =========================================================
# SQL REVIEW VISUALIZATION
# =========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    x='review_score',
    y='total_reviews',
    hue='review_score',
    data=sql_reviews,
    palette='viridis',
    legend=False
)

plt.title('Customer Review Analysis (SQL)')
plt.xlabel('Review Score')
plt.ylabel('Total Reviews')

plt.tight_layout()

plt.savefig('screenshots/sql_review_analysis.png')

plt.show()

# =========================================================
# FINAL ORDERS DATASET
# =========================================================

final_orders = revenue_data[
    [
        'order_id',
        'customer_id',
        'order_status',
        'order_purchase_timestamp',
        'order_month',
        'payment_value',
        'payment_installments',
        'delivery_time_days'
    ]
]

print("\nFinal Orders Dataset")
print(final_orders.head())

# =========================================================
# CUSTOMER ANALYTICS DATASET
# =========================================================

customer_analysis = pd.merge(
    customers,
    final_orders,
    on='customer_id',
    how='inner'
)

print("\nCustomer Analytics Dataset")
print(customer_analysis.head())

# =========================================================
# EXPORT CLEANED DATASETS
# =========================================================

final_orders.to_csv(
    'powerbi/final_orders.csv',
    index=False
)

customer_analysis.to_csv(
    'powerbi/customer_analysis.csv',
    index=False
)

top_states_sql.to_csv(
    'powerbi/top_states_sql.csv',
    index=False
)

sql_revenue.to_csv(
    'powerbi/sql_revenue.csv',
    index=False
)

sql_reviews.to_csv(
    'powerbi/sql_reviews.csv',
    index=False
)

print("\nPower BI Files Exported Successfully")

# =========================================================
# KPI SUMMARY TABLE
# =========================================================

kpi_summary = pd.DataFrame({

    'Metric': [
        'Total Revenue',
        'Average Delivery Time',
        'Average Review Score',
        'Average Installments'
    ],

    'Value': [
        round(total_revenue, 2),
        round(average_delivery, 2),
        round(average_review, 2),
        round(average_installments, 2)
    ]

})

print("\nKPI Summary")
print(kpi_summary)

# =========================================================
# EXPORT KPI SUMMARY
# =========================================================

kpi_summary.to_csv(
    'powerbi/kpi_summary.csv',
    index=False
)

print("\nKPI Summary Exported")

# =========================================================
# CLOSE DATABASE CONNECTION
# =========================================================

connection.close()

print("\nDatabase Connection Closed")