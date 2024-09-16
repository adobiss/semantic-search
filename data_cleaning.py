import pandas as pd
import chardet
from io import StringIO

def string_pattern_search(df, params, cols=None, summary='all'):
    """
    Search for a string pattern in the specified columns of a DataFrame and print a summary.
    
    Args:
    df (pd.DataFrame): The DataFrame to search within.
    params (tuple): A tuple containing the following parameters:
        - pattern (str): The pattern to search for.
        - regex_setting (bool): Whether to interpret the pattern as a regular expression.
    cols (list, optional): The list of columns to search within. If None, all columns are searched. Defaults to None.
    summary (str, optional): Specifies the type of summary to print. 
        - 'all' to print the total occurrences.
        - 'unique' to print the unique occurrences.
        - Defaults to 'all'.
    
    Returns:
    pd.DataFrame: A DataFrame containing columns where the pattern was found, with only the rows that contain the pattern.
    """
    pattern, regex_setting = params

    sliced_dfs = []

    if cols is None:
        cols = df.columns

    for col in cols:
        pattern_mask = df[col].str.contains(pattern, regex=regex_setting, na=False)
        pattern_df_next = df.loc[pattern_mask, col] # Returns Series
        sliced_dfs.append(pattern_df_next)

    pattern_df_full = pd.concat(sliced_dfs, axis=1) # Concatenates Series
    pattern_df_full.dropna(axis=1, how='all', inplace=True)

    if pattern_df_full.empty:
        print(f"Either no occurrences of the pattern '{pattern}' found or all have been replaced.")
    elif summary == 'all':
        print(f"Total occurrences of the pattern '{pattern}':\n", pattern_df_full.count(), sep='')
    elif summary == 'unique':
        print(f"Unique occurrences of the pattern '{pattern}':\n", pattern_df_full.nunique(), sep='')

    return pattern_df_full

def string_pattern_replace(df, params, replace_with, cols=None):
    """
    Replace a string pattern in the specified columns of a DataFrame and print a summary of the replacements.
    
    Args:
    df (pd.DataFrame): The DataFrame to perform the replacement within.
    params (tuple): A tuple containing the following parameters:
        - pattern (str): The pattern to search for and replace.
        - regex_setting (bool): Whether to interpret the pattern as a regular expression.
    replace_with (str): The string to replace the pattern with.
    cols (list, optional): The list of columns to perform the replacement within. If None, all columns are used. Defaults to None.
    
    Returns:
    None: The DataFrame is modified in place. After replacement, a summary of the replacements is printed.
    """
    pattern, regex_setting = params

    if cols is None:
        cols = df.columns    

    for col in cols:
        df[col] = df[col].str.replace(pattern, replace_with, regex=regex_setting)   

    string_pattern_search(df, params, cols)

### Dynamic decoding section ###

def detect_and_decode_line(line):
    """
    Attempts to decode a given line using UTF-8 encoding first.
    If UTF-8 decoding fails, it uses the `chardet` library to detect the encoding and tries to decode with the detected encoding.

    Parameters:
    line (bytes): The line of text to decode.

    Returns:
    tuple: A tuple containing the decoded line (str), the detected encoding (str), and the confidence level (float).
           If decoding fails, returns None for the decoded line.
    """
    try:
        # Try decoding the line using UTF-8 encoding
        decoded_line = line.decode('utf-8')
        return decoded_line, 'utf-8', None
    except UnicodeDecodeError:
        # If UTF-8 decoding fails, use chardet to detect the encoding
        result = chardet.detect(line)
        encoding = result['encoding']
        confidence = result['confidence']
        try:
            # Try decoding the line using the detected encoding
            decoded_line = line.decode(encoding)
            return decoded_line, encoding, confidence
        except UnicodeDecodeError:
            # If decoding fails again, return None for the decoded line
            return None, encoding, confidence

def read_file_by_lines(file_path):
    """
    Reads the entire file line by line and returns a list of lines.

    Parameters:
    file_path (str): The path to the file to read.

    Returns:
    list: A list of lines read from the file.
    """
    with open(file_path, 'rb') as f:
        lines = f.readlines()
    return lines

def single_threaded_read_and_decode_lines(file_path):
    """
    Reads the file line by line, attempts to decode each line, and collects the decoded lines.
    Also collects the encoding and confidence level for each line.

    Parameters:
    file_path (str): The path to the file to read and decode.

    Returns:
    tuple: A tuple containing the decoded text (str) and a list of encodings tried (list of tuples).
           Each tuple in the list contains the encoding (str) and confidence level (float).
    """
    lines = read_file_by_lines(file_path)
    total_lines = len(lines)
    print(f"Total lines to process: {total_lines}")

    decoded_lines = []
    encodings_tried = []
    for line_index, line in enumerate(lines):
        decoded_line, encoding, confidence = detect_and_decode_line(line)
        if decoded_line is not None:
            decoded_lines.append(decoded_line)
        if encoding is not None:
            encodings_tried.append((encoding, confidence))
        # Print progress
        print(f"Processed line {line_index + 1} of {total_lines}")

    # Join decoded lines and ensure no extra trailing newlines
    decoded_text = ''.join(decoded_lines).strip()
    print(f"Total decoded lines: {len(decoded_lines)}")
    return decoded_text, encodings_tried

def load_data_into_dataframe(decoded_text):
    """
    Loads the decoded text into a Pandas DataFrame.

    Parameters:
    decoded_text (str): The decoded text to load into the DataFrame.

    Returns:
    DataFrame: A Pandas DataFrame containing the data from the decoded text.
    """
    data = StringIO(decoded_text)
    df = pd.read_csv(data)
    return df

def detect_encoding_for_row(df, index):
    """
    Detect the character encoding for a specific row in a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        index (int): The index of the row to be checked.
    
    Returns:
        tuple: A tuple containing the detected encoding and confidence level.
    """
    # Convert the row to bytes
    row_bytes = df.iloc[index].to_string().encode('utf-8')
    
    # Detect encoding using chardet
    result = chardet.detect(row_bytes)
    encoding = result['encoding']
    confidence = result['confidence']
    
    return encoding, confidence