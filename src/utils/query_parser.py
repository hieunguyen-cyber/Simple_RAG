# src/utils/query_parser.py
import pandas as pd
import json
from typing import Dict, Any
from src.generation.llm import LLM

class QueryParser:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the query parser with restaurant data.
        
        Args:
            df (pd.DataFrame): DataFrame containing restaurant data.
        """
        self.llm = LLM()
        self.df = df
        self.valid_cuisines = sorted(self.df['cuisine'].unique().tolist())
        self.valid_price_ranges = sorted(self.df['price_range'].unique().tolist())
        self.valid_dishes = sorted(set([dish for dishes in self.df['dishes'] for dish in dishes]))
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse the query to extract features.
        
        Args:
            query (str): User query.
        
        Returns:
            Dict[str, Any]: Parsed features.
        """
        # Format prompt using LLM's prompt template
        prompt = self.llm.format_query_prompt(
            query=query,
            cuisines=self.valid_cuisines,
            dishes=self.valid_dishes,
            price_ranges=self.valid_price_ranges
        )
        
        # Generate response
        response = self.llm.generate(prompt)
        
        # Parse JSON response
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            parsed = json.loads(response[json_start:json_end])
            return parsed
        except json.JSONDecodeError:
            return {
                "cuisine": None,
                "menu": [],
                "price_range": None,
                "distance": None,
                "rating": None,
                "description": query
            }


# Quick test block for QueryParser
if __name__ == "__main__":
    import pandas as pd

    sample_data = {
        "cuisine": ["Italian", "Japanese"],
        "price_range": ["$", "$$"],
        "dishes": [["pizza", "pasta"], ["sushi", "ramen"]]
    }
    df = pd.DataFrame(sample_data)
    parser = QueryParser(df)

    user_query = "I want cheap sushi"
    result = parser.parse_query(user_query)

    print("Parsed Query Result:")
    print(result)
            