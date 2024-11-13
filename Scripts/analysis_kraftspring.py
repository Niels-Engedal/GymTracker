import pandas as pd

def load_trc_file_with_pandas(file_path):
    """
    Loads a .trc motion capture file into a Pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the .trc file.
    
    Returns:
    tuple: A tuple containing the header information (as a dict) and the data (as a DataFrame).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Parse header information (adjust based on your .trc file's structure)
    headers = {}
    for i in range(4):  # First 4 lines typically contain metadata
        headers[f'Header line {i+1}'] = lines[i].strip()
    
    # Load the data starting from the 6th line (index 5), assuming tab-separated values
    data_df = pd.read_csv(file_path, sep='\t', skiprows=5)
    
    return headers, data_df

# Example usage
file_path = '/Users/niels/Desktop/University/Third Semester/Perception and Action/Exam/Gymnastics Motion Tracking/Code for Gym Tracking/Analyzed Videos/7Nov24/Kraftspring/kraft_bad_id1/Sequence 01_Sports2D_px_person00.trc'
headers, data_df = load_trc_file_with_pandas(file_path)

# Display the parsed header info and data preview
print("Header Information:")
for key, value in headers.items():
    print(f"{key}: {value}")

print("\nData Preview:")
print(data_df.head())
