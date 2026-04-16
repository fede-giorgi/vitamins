import pandas as pd

def load_and_preprocess_data(file_path):
    """
    Load the file (Excel or CSV) and preprocess the data.
    """
    print(f"Loading data from {file_path}...")
    if str(file_path).endswith('.csv'):
        df = pd.read_csv(file_path, dtype=str)
    else:
        df = pd.read_excel(file_path, sheet_name='ALL (DO NOT EDIT)', dtype=str)
    print(f"Loaded {len(df)} rows.")
    
    # Calculate Ground_Truth based on product codes
    # 1927, 1931, 1932 are the specific vitamin codes
    VITAMIN_CODES = ['1927', '1931', '1932']
    df['Ground_Truth'] = (
        df[['Product_1', 'Product_2', 'Product_3']]
        .isin(VITAMIN_CODES)
        .any(axis=1)
        .astype(int)
    )
    
    return df