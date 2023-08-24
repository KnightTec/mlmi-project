import os
import pandas as pd

def split_csv_file(csv_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path, header=None)
    
    # Calculate midpoint for splitting
    midpoint = len(df) // 2
    
    # Split the dataframe into two equal halves
    first_half = df.iloc[:midpoint]
    second_half = df.iloc[midpoint:]
    
    # Save the halves to new CSV files
    base_name = os.path.basename(csv_file_path).rsplit('.', 1)[0]
    directory = os.path.dirname(csv_file_path)
    
    first_half.to_csv(os.path.join(directory, f"{base_name}0.csv"), index=False, header=False)
    second_half.to_csv(os.path.join(directory, f"{base_name}1.csv"), index=False, header=False)

def split_all_csv_files_in_folder(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            split_csv_file(os.path.join(folder_path, file_name))

if __name__ == '__main__':
    folder_path = input("Enter the folder path containing the CSV files: ")
    split_all_csv_files_in_folder(folder_path)
    print("CSV files split successfully.")
