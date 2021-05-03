import datetime
import pandas as pd
import plotly.express as px
import re


def load_financial_data(
    file: str = r"C:\Users\dm635\Downloads\transactions (1).csv",
) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


def clean_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all "bogus" columns, convert the `Date` column to timestamp,
    add a `Month` (timestamp-typed) column, properly adjust the `Amount` values.
    """
    # https://stackoverflow.com/a/45147491/2543689
    df = df.dropna(axis=1, how="all")
    df["Date"] = pd.to_datetime(df["Date"])
    # https://github.com/pandas-dev/pandas/issues/15303
    df["Month"] = df["Date"].apply(lambda d: datetime.datetime(d.year, d.month, 1))
    df["Amount"] = df.apply(
        lambda row: -row["Amount"]
        if row["Transaction Type"] == "debit"
        else row["Amount"],
        axis=1,
    )
    return df


def credit_card_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to transactions from credit cards that are not bill payments.
    """
    rv = df[df["Account Name"].str.contains("Bank of America|credit", case=False)]
    rv = rv[rv["Category"] != "Credit Card Payment"]
    # Sanity check: no large input monies.
    assert rv[rv["Amount"] > 500].empty
    return rv


def summarize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    top_categories = set(
        df.groupby("Category").sum().reset_index().sort_values(by="Amount").Category[:5]
    )

    def squashed_category(row):
        category = row["Category"]
        if (
            category in top_categories
            or re.match("pet|vet", category, re.IGNORECASE)
            or abs(row["Amount"]) > 200
        ):
            return category
        else:
            return "other"

    df["Category"] = df.apply(squashed_category, axis=1)
    df = df.groupby(by=["Month", "Category"]).sum().reset_index()
    return df


if __name__ == "__main__":
    # Ivy is magic.
    raw_data = load_financial_data()
    clean_data = clean_financial_data(raw_data)
    cct = credit_card_transactions(clean_data)
    st = summarize_transactions(cct)

    # https://plotly.com/python/bar-charts/
    fig = px.bar(st, x="Month", y="Amount", color="Category")
    fig.show()
