import pandas as pd
import numpy as np
import os
import shutil
import json
import math


def create_mini_datalake_departments(
    base_path: str,
    num_customers: int = 20,
    split_v1: int = 2,
    num_sales_chunks: int = 4,
):
    """
    Generate a mini data lake with two departments:
    - Department A (Sales) with customers_v1 and sales
    - Department B (Support) with customers_v2 and product reviews
    """

    # Clean up previous folder
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)

    # ------------------------
    # Department A: Sales
    # ------------------------
    dept_a_path = os.path.join(base_path, "department_a")
    os.makedirs(dept_a_path, exist_ok=True)

    # Customers v1 (split)
    first_names = [
        "Alice",
        "Bob",
        "Charlie",
        "Diana",
        "Ethan",
        "Fiona",
        "George",
        "Hannah",
        "Ian",
        "Julia",
        "Kevin",
        "Laura",
        "Michael",
        "Nina",
        "Oliver",
        "Paula",
        "Quentin",
        "Rachel",
        "Sam",
        "Tina",
    ]
    last_names = [
        "Smith",
        "Brown",
        "Johnson",
        "Williams",
        "Jones",
        "Miller",
        "Davis",
        "Garcia",
        "Martinez",
        "Taylor",
        "Anderson",
        "Thomas",
        "Jackson",
        "White",
        "Harris",
        "Martin",
        "Thompson",
        "Moore",
        "Lee",
        "Walker",
    ]
    countries_short = ["US", "UK", "FR", "DE", "ES"]

    customers_v1 = pd.DataFrame(
        {
            "customer_id": range(1, num_customers + 1),
            "first_name": first_names[:num_customers],
            #"last_name": last_names[:num_customers],
            "email": [f"user{i}@example.com" for i in range(1, num_customers + 1)],
            "country": np.random.choice(countries_short, num_customers),
            "telephone": np.random.randint(10000, 99999, num_customers),
        }
    )

    os.makedirs(os.path.join(dept_a_path, "customers"), exist_ok=True)
    chunk_size = int(np.ceil(len(customers_v1) / split_v1))
    for i in range(split_v1):
        chunk = customers_v1.iloc[i * chunk_size : (i + 1) * chunk_size]
        if not chunk.empty:
            chunk.to_csv(
                os.path.join(dept_a_path, f"customers/customers_part{i+1}.csv"),
                index=False,
            )

    # Sales data (nested in sales folder)
    np.random.seed(42)
    sales_full = pd.DataFrame(
        {
            "order_key": range(1001, 1001 + num_customers * 5),
            "cust_id": np.random.choice(range(1, num_customers + 1), num_customers * 5),
            "product_id": np.random.choice(range(2001, 2006), num_customers * 5),
            "quantity": np.random.randint(1, 5, num_customers * 5),
            "date": pd.date_range("2023-01-01", periods=num_customers * 5).astype(str),
            "amount_usd": np.random.randint(20, 200, num_customers * 5),
        }
    )

    sales_path = os.path.join(dept_a_path, "sales")
    os.makedirs(sales_path, exist_ok=True)
    chunk_size = math.ceil(len(sales_full) / num_sales_chunks)
    for i in range(num_sales_chunks):
        chunk = sales_full.iloc[i * chunk_size : (i + 1) * chunk_size]
        if not chunk.empty:
            chunk.to_csv(os.path.join(sales_path, f"sales_part{i+1}.csv"), index=False)

    # Department A products
    products_path_a = os.path.join(dept_a_path, "products")
    os.makedirs(products_path_a, exist_ok=True)
    products_a = [
        {
            "product_id": 2001,
            "name": "Laptop",
            "type": "Electronics",
            "price": 1200.50,
            "currency": "USD",
        },
        {
            "product_id": 2002,
            "name": "Phone",
            "type": "Electronics",
            "price": 799.99,
            "currency": "USD",
        },
    ]
    products_v2 = [
        {
            "product_id": 3003,
            "name": "Headphones",
            "type": "Accessories",
            "price": 199.99,
            "currency": "USD",
            "stock": 250,
        },
        {
            "product_id": 3004,
            "name": "Shoes",
            "type": "Fashion",
            "price": 89.99,
            "currency": "USD",
            "color": "Black",
        },
        {
            "product_id": 3005,
            "name": "Backpack",
            "type": "Fashion",
            "price": 59.99,
            "currency": "USD",
            "material": "Nylon",
        },
    ]
    with open(os.path.join(products_path_a, "products_a.json"), "w") as f:
        json.dump(products_a, f, indent=4)
    with open(os.path.join(products_path_a, "products_b.json"), "w") as f:
        json.dump(products_v2, f, indent=4)
    # Geography
    geo = pd.DataFrame(
        {
            "code": ["US", "UK", "FR", "DE", "ES"],
            "country_name": ["United States", "United Kingdom", "France", "Germany", "Spain"],
            "continent": ["North America", "Europe", "Europe", "Europe", "Europe"],
            "capital": ["Washington, D.C.", "London", "Paris", "Berlin", "Madrid"],
            "population": [331000000, 67000000, 65000000, 83000000, 47000000],
        }
    )
    geo.to_excel(os.path.join(dept_a_path, "countries.xlsx"), index=False)

    # ------------------------
    # Department B: Support + Reviews
    # ------------------------
    dept_b_path = os.path.join(base_path, "department_b")
    os.makedirs(dept_b_path, exist_ok=True)

    # Customers
    countries_full = ["United States", "United Kingdom", "France", "Germany", "Spain"]
    fullnames = [
        f"{fn} {ln}"
        for fn, ln in zip(first_names[:num_customers], last_names[:num_customers])
    ]
    customers_v2 = pd.DataFrame(
        {
            "custID": range(1, num_customers + 1),
            # "fullname": fullnames,
            "name": first_names[:num_customers],
            # "surname": last_names[:num_customers],
            "location": np.random.choice(countries_full, num_customers),
            "telephone": np.random.randint(10000, 99999, num_customers),
        }
    )
    os.makedirs(os.path.join(dept_b_path, "customers"), exist_ok=True)
    customers_v2.to_csv(
        os.path.join(dept_b_path, "customers/customers_v2.csv"), index=False
    )

    # Products (joinable with Dept A)
    products_b = [
        {
            "product_id": 2001,
            "name": "Laptop",
            "type": "Electronics",
            "price": 1200.50,
            "currency": "USD",
        },
        {
            "product_id": 2002,
            "name": "Phone",
            "type": "Electronics",
            "price": 799.99,
            "currency": "USD",
        },
        {
            "product_id": 2003,
            "name": "Headphones",
            "type": "Accessories",
            "price": 199.99,
            "currency": "USD",
        },
        {
            "product_id": 2004,
            "name": "Shoes",
            "type": "Fashion",
            "price": 89.99,
            "currency": "USD",
        },
        {
            "product_id": 2005,
            "name": "Backpack",
            "type": "Fashion",
            "price": 59.99,
            "currency": "USD",
        },
    ]
    products_path_b = os.path.join(dept_b_path, "products")
    os.makedirs(products_path_b, exist_ok=True)
    with open(os.path.join(products_path_b, "products_c.json"), "w") as f:
        json.dump(products_b, f, indent=4)

    # Reviews for products
    reviews = []
    review_texts = ["Excellent!", "Good", "Average", "Poor", "Terrible"]
    for pid in range(2001, 2006):
        for _ in range(3):  # 3 reviews per product
            reviews.append(
                {
                    "product_id": pid,
                    "review_text": np.random.choice(review_texts),
                    "rating": np.random.randint(1, 6),
                }
            )
    reviews_df = pd.DataFrame(reviews)
    reviews_path = os.path.join(dept_b_path, "reviews")
    os.makedirs(reviews_path, exist_ok=True)
    reviews_df.to_csv(os.path.join(reviews_path, "product_reviews.csv"), index=False)

    # ------------------------
    # Zip the whole lake
    # ------------------------
    #zip_path = f"{base_path}.zip"
    #shutil.make_archive(base_path, "zip", base_path)
    #print(
    #    f"Mini data lake created with {split_v1} customer files (Sales) and {num_sales_chunks} sales chunks."
    #)
    #print(f"Department Support contains 1 customer file, products and reviews.")
    #return zip_path


# ------------------------
# Example usage
# ------------------------
create_mini_datalake_departments(
    "/home/data/bronze",
    num_customers=20,
    split_v1=2,
    num_sales_chunks=5,
)
