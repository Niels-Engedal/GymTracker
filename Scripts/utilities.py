import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def list_filepaths(directory, extensions):
    """
    Lists all files with the specified extensions in the directory and its subdirectories.
    
    Parameters:
    directory (str): The path to the directory.
    extensions (str or list/tuple): The file extension(s) to look for.
    
    Returns:
    list: A list of file paths to files with the specified extensions in the directory and subdirectories.
    """
    # Ensure extensions is a tuple for consistency
    if isinstance(extensions, str):
        extensions = (extensions,)
    
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            if file.endswith(tuple(extensions)):
                files.append(os.path.join(root, file))
    return files


def load_trc_file(file_path):
    """
    Loads a .trc motion capture file into a Pandas DataFrame, handling multi-line headers and pairing marker names with coordinates.
    
    Parameters:
    file_path (str): The path to the .trc file.
    
    Returns:
    tuple: A tuple containing the header information (as a dict) and the data (as a DataFrame).
    """
    # Read the entire file to access header lines
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    # Extract header information from the first three lines
    headers = {
        'PathFileType': lines[0].strip(),
        'Data Info': lines[1].strip(),
        'Rates and Frames': lines[2].strip()
    }
    
    # Extract marker names and coordinate labels from the last two header lines
    marker_names = lines[3].strip().split()  # e.g., ['Frame#', 'Time', 'Hip', 'RHip', ...]
    coordinate_labels = lines[4].strip().split()  # e.g., ['X1', 'Y1', 'Z1', 'X2', 'Y2', 'Z2', ...]

    # Combine marker names with coordinate labels to form complete column names
    column_names = []
    marker_index = 2  # Start at 2 to skip 'Frame#' and 'Time'
    for coord in coordinate_labels:
        if coord.startswith('X'):
            column_names.append(f"{marker_names[marker_index]}_X")
        elif coord.startswith('Y'):
            column_names.append(f"{marker_names[marker_index]}_Y")
        elif coord.startswith('Z'):
            column_names.append(f"{marker_names[marker_index]}_Z")
            marker_index += 1

    # Prepend 'Frame#' and 'Time' to the column names
    column_names = ['Frame#', 'Time'] + column_names
    
    # Create the DataFrame starting from the data rows (after the header)
    data_df = pd.read_csv(file_path, sep='\s+', skiprows=5, names=column_names)
    
    return headers, data_df


def load_mot_file(file_path):
    """
    Loads a .mot file containing joint angles into a Pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the .mot file.
    
    Returns:
    tuple: A tuple containing metadata (as a dict) and the data (as a DataFrame).
    """
    metadata = {}
    data_lines = []
    
    with open(file_path, 'r') as file:
        # Read and parse the header
        line = file.readline().strip()
        while line != "endheader":
            if '=' in line:
                key, value = line.split('=')
                metadata[key.strip()] = value.strip()
            line = file.readline().strip()
        
        # Read the column headers line
        column_headers_line = file.readline().strip()
        column_headers = column_headers_line.split('\t')  # Split by tabs if columns are tab-separated
        column_headers = [col.replace(' ', '_') for col in column_headers]  # Replace spaces in column names with underscores
        
        # Read the data lines
        for line in file:
            data_lines.append(line.strip().split('\t'))  # Split by tabs if data is tab-separated

    # Convert data to DataFrame
    data_df = pd.DataFrame(data_lines, columns=column_headers)
    
    # Convert numerical columns from string to float
    data_df = data_df.apply(pd.to_numeric, errors='coerce')

    return metadata, data_df


def extract_identifiers(file_path):
    """
    Extracts move, evaluation, participant ID, video number, and person tracked from the file path.
    
    Parameters:
    file_path (str): The path to the file.
    
    Returns:
    tuple: A tuple containing move, evaluation, participant ID, video number, and person tracked.
    """
    # Regex to extract identifiers from the path after "Sports2D/"
    match = re.search(r"Sports2D/([^_]+)_(good|bad)_id(\d+)_(\d+)_.*_(person\d+)", file_path)
    if match:
        move, evaluation, participant_id, video_number, person_tracked = match.groups()
        return move, evaluation, f"id{participant_id}_{person_tracked}", video_number
    else:
        raise ValueError(f"Cannot extract identifiers from the file path: {file_path}")

def load_multiple_files(file_paths, file_type='trc'):
    """
    Loads multiple .trc or .mot files into a combined DataFrame with participant identifiers.

    Parameters:
    file_paths (list of str): List of paths to the .trc or .mot files.
    file_type (str): Type of file to load ('trc' or 'mot').

    Returns:
    pd.DataFrame: A combined DataFrame with participant ID as a key and other metadata as columns.
    """
    data_list = []
    
    for path in file_paths:
        try:
            move, evaluation, participant_key, video_number = extract_identifiers(path)
            
            if file_type == 'trc':
                _, data_df = load_trc_file(path)
            elif file_type == 'mot':
                metadata, data_df = load_mot_file(path)
                # Optionally add metadata columns if needed
                for key, value in metadata.items():
                    data_df[key] = value
            
            # Add participant-specific information as new columns
            data_df['participant_id'] = participant_key
            data_df['move'] = move
            data_df['evaluation'] = evaluation
            data_df['video_number'] = video_number
            
            data_list.append(data_df)
        
        except ValueError as e:
            print(e)

    # Combine all DataFrames into a single DataFrame
    combined_df = pd.concat(data_list, ignore_index=True)
    return combined_df
