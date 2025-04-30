# src/utils/data_loader.py

import pandas as pd
import json

def load_restaurant_data(file_path: str) -> pd.DataFrame:
    """
    Load restaurant data from JSON file into a DataFrame.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        pd.DataFrame: DataFrame containing restaurant data.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    # Create a text field for embedding and BM25
    df['text'] = df.apply(
        lambda row: f"{row['name']} ({row['cuisine']}): {', '.join(row['dishes'])}. "
                    f"Price: {row['price_range']}, Distance: {row['distance']} km, "
                    f"Rating: {row['rating']}. Description: {row['description']}",
        axis=1
    )
    return df
if __name__ == "__main__":
    print(load_restaurant_data("./data/restaurants.json"))