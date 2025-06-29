import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import sys

def consolidate_tickets(file_path, similarity_threshold=80):
    """
    Scans a CSV file, identifies rows with similar subjects, and consolidates them.

    Args:
        file_path (str): The path to the input CSV file.
        similarity_threshold (int): The minimum fuzzy matching score (0-100)
                                    to consider two subjects similar.

    Returns:
        pd.DataFrame: A new DataFrame with consolidated rows.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found. Please ensure the file exists.", file=sys.stderr)
        return pd.DataFrame()
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        return pd.DataFrame()

    # Ensure 'subject' column exists
    if 'subject' not in df.columns:
        print("Error: 'subject' column not found in the CSV file.", file=sys.stderr)
        return pd.DataFrame()

    # Initialize a list to store consolidated rows
    consolidated_rows = []
    # Keep track of indices already processed
    processed_indices = set()

    # Get a list of subjects and their original indices
    subjects_with_indices = [(sub, i) for i, sub in enumerate(df['subject'])]

    for i, (current_subject, current_idx) in enumerate(subjects_with_indices):
        if current_idx in processed_indices:
            continue # Skip if this row has already been processed as part of a group

        # Initialize a group for similar subjects
        current_group_indices = {current_idx}
        current_group_subjects = [current_subject]

        # Compare the current subject with all other unprocessed subjects
        for j, (other_subject, other_idx) in enumerate(subjects_with_indices):
            if other_idx in processed_indices or other_idx == current_idx:
                continue

            # Calculate similarity using token_sort_ratio for better robustness to word order
            score = fuzz.token_sort_ratio(str(current_subject), str(other_subject))

            if score >= similarity_threshold:
                current_group_indices.add(other_idx)
                current_group_subjects.append(other_subject)

        # Mark all found similar subjects as processed
        for idx in current_group_indices:
            processed_indices.add(idx)

        # Consolidate data for the current group
        if current_group_indices:
            # Select all rows belonging to this group
            group_df = df.loc[list(current_group_indices)]

            consolidated_row = {}
            # Use the subject of the first entry in the group as the master subject
            consolidated_row['subject'] = current_subject # Representative subject

            # Consolidate other columns
            for col in df.columns:
                if col == 'subject':
                    continue # Skip subject, already handled

                # Get unique values for the column within the group
                unique_values = group_df[col].astype(str).unique()
                # Join them with a semicolon for consolidation
                consolidated_row[col] = "; ".join(unique_values)

            consolidated_rows.append(consolidated_row)

    # Create a new DataFrame from consolidated rows
    consolidated_df = pd.DataFrame(consolidated_rows)
    return consolidated_df

# --- Main execution ---
# Define the path to the uploaded CSV file
file_path = 'C:/Users/riya.shukla/Downloads/062920091239Sheet.csv'
output_file_path = 'Fullconsolidated_tickets.csv'

# Set your desired similarity threshold (e.g., 75, 80, 85, etc.)
# A higher threshold means stricter similarity.
SIMILARITY_THRESHOLD = 80

print(f"Consolidating tickets from '{file_path}' with a similarity threshold of {SIMILARITY_THRESHOLD}...")
consolidated_df = consolidate_tickets(file_path, SIMILARITY_THRESHOLD)

if not consolidated_df.empty:
    try:
        consolidated_df.to_csv(output_file_path, index=False)
        print(f"Consolidation complete! Consolidated data saved to '{output_file_path}'")
        # Removed the .to_markdown() line to avoid the 'tabulate' dependency error
        print(f"\nTotal original rows: {len(pd.read_csv(file_path))}")
        print(f"Total consolidated rows: {len(consolidated_df)}")
    except Exception as e:
        print(f"Error saving consolidated data to CSV: {e}", file=sys.stderr)
else:
    print("No data to consolidate or an error occurred during processing.")
