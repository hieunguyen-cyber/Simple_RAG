# src/retrieval/keyword_filter.py

import pandas as pd
from typing import Dict, Any

def filter_restaurants(df: pd.DataFrame, parsed_query: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter restaurants based on extracted features from the query.
    
    Args:
        df (pd.DataFrame): DataFrame containing restaurant data.
        parsed_query (Dict[str, Any]): Parsed query with features.
    
    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_df = df.copy()
    
    if parsed_query.get("cuisine"):
        filtered_df = filtered_df[filtered_df["cuisine"].str.lower() == parsed_query["cuisine"].lower()]
    
    if parsed_query.get("menu"):
        filtered_df = filtered_df[filtered_df["dishes"].apply(
            lambda dishes: any(item.lower() in [d.lower() for d in dishes] for item in parsed_query["menu"])
        )]
    
    if parsed_query.get("price_range"):
        filtered_df = filtered_df[filtered_df["price_range"].str.lower() == parsed_query["price_range"].lower()]
    
    distance = parsed_query.get("distance")
    if isinstance(distance, (int, float)):
        filtered_df = filtered_df[filtered_df["distance"] <= distance]
    elif distance in ["nearby", "close"]:
        filtered_df = filtered_df[filtered_df["distance"] <= 2.0]
    elif distance == "far":
        filtered_df = filtered_df[filtered_df["distance"] <= 10.0]
    
    if parsed_query.get("rating"):
        filtered_df = filtered_df[filtered_df["rating"] >= parsed_query["rating"]]
    
    return filtered_df