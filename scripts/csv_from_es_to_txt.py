import pandas as pd
import os
from pathlib import Path

def convert_csv_to_txt(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Print available columns for debugging
    print("Available columns in CSV:")
    print(df.columns.tolist())
    
    # Create input directory in the same folder as the CSV
    csv_dir = os.path.dirname(csv_path)
    input_dir = os.path.join(csv_dir, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Group data by original file name
    grouped = df.groupby('metadata.original_file_name')
    
    for file_name, group in grouped:
        # Sort by chunk_id to maintain paragraph order
        group = group.sort_values('metadata.chunk_id')
        
        # Create the output file path
        output_path = os.path.join(input_dir, f"{file_name}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            current_chapter = None
            
            for _, row in group.iterrows():
                # If chapter changes, add chapter title
                if row['metadata.chapter'] != current_chapter:
                    current_chapter = row['metadata.chapter']
                    if current_chapter:
                        f.write(f"# {current_chapter}\n\n")
                
                # Write paragraph
                f.write(f"{row['paragraph']}\n\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV to organized text files')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
    
    args = parser.parse_args()
    convert_csv_to_txt(args.csv_path)
