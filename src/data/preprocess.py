import pandas as pd

def preprocess_netflix_data(df):
    """Preprocess Netflix data"""
    df = df.copy()

    # Handle missing values
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["duration"] = df["duration"].fillna("Unknown")

    # Handle mode safely for 'rating' column
    rating_mode = df["rating"].mode()
    default_rating = rating_mode.iloc[0] if not rating_mode.empty else "TV-MA"
    df["rating"] = df["rating"].fillna(default_rating)

    # Convert date
    df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month

    # Feature engineering for content-based filtering / vectorization
    text_cols = ["director", "cast", "listed_in", "description"]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    df["combined_features"] = df.apply(
        lambda r: " ".join(
            [str(r[c]) for c in text_cols if c in r and str(r[c]).strip() != "Unknown"]
        ),
        axis=1,
    )

    return df
