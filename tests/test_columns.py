import pandas as pd

from analysis.columns import detect_columns, safe_summary


def test_sample_file_roles(sample):
    roles = sample.roles
    assert roles.date == "order_date"
    assert roles.metric == "total"
    assert roles.ids == ["order_id"]
    assert roles.personal == ["customer_email"]
    assert set(roles.segments) == {"device", "payment_method", "region", "sales_channel"}


def test_personal_columns_detected_by_name_and_value():
    df = pd.DataFrame(
        {
            "Created at": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "Amount": [10.0, 20.0, 30.0],
            "Phone": ["555-123-4567", "555-987-6543", "555-111-2222"],
            "contact": ["a@x.com", "b@y.com", "c@z.com"],  # email by value, not name
            "Billing Address": ["1 Main St", "2 Oak Ave", "3 Elm Rd"],
            "product_name": ["Mug", "Lamp", "Mug"],  # 'name' alone is not personal
        }
    )
    roles = detect_columns(df)
    assert set(roles.personal) == {"Phone", "contact", "Billing Address"}
    assert roles.date == "Created at"
    assert roles.metric == "Amount"
    assert "product_name" in roles.segments


def test_metric_prefers_revenue_like_names():
    df = pd.DataFrame({"date": ["2026-01-01"] * 3, "quantity": [1, 2, 3], "net_sales": [5.0, 6.0, 7.0]})
    roles = detect_columns(df)
    assert roles.metric == "net_sales"
    assert roles.other_metrics == ["quantity"]


def test_safe_summary_never_shows_personal_values(sample):
    raw = pd.read_csv("data/sample_shop.csv")
    summary = safe_summary(raw, detect_columns(raw))
    text = str(summary)
    assert "@example.com" not in text
    assert summary["columns"]["customer_email"]["examples"] == "hidden"
    assert len(summary["columns"]["device"]["examples"]) <= 5


def test_dataset_drops_personal_columns(sample):
    assert "customer_email" not in sample.df.columns
