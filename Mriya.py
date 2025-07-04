import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import numpy as np
import sys
from collections import defaultdict
import re # Import regex module
import os # Import os module for file operations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK imports
import nltk

# --- Mock convosense_utilities Library (for general text refinement) ---
# This mock is retained as it provides general text normalization beyond signature removal.
class MockConvosenseUtilities:
    def refine_text(self, text):
        """
        A mock function to simulate text refinement from convosense_utilities.
        This now includes more aggressive normalization patterns and uses NLTK for tokenization.
        """
        if not isinstance(text, str):
            return ""

        # Use NLTK for word tokenization
        tokens = nltk.word_tokenize(text.lower())

        # Define patterns for words/phrases to remove or normalize
        # These will be applied to individual tokens or re-joined text

        # Normalize common phrasing variations (applied before general noise removal)
        replacements = {
            'calls': 'calling',
            'not showing': 'not_showing',
            'not working': 'not_working',
            'reset password': 'password_reset',
            'password reset': 'password_reset',
            'rtmt password': 'password_reset',
            'call landing': 'call_landing',
        }

        # Apply replacements to tokens
        processed_tokens = []
        skip_next = 0 # For multi-word replacements
        for i, token in enumerate(tokens):
            if skip_next > 0:
                skip_next -= 1
                continue

            found_replacement = False
            for phrase, replacement in replacements.items():
                phrase_tokens = nltk.word_tokenize(phrase)
                if len(phrase_tokens) > 1:
                    # Check for multi-word phrases
                    if i + len(phrase_tokens) <= len(tokens) and \
                       [tokens[j] for j in range(i, i + len(phrase_tokens))] == phrase_tokens:
                        processed_tokens.append(replacement)
                        skip_next = len(phrase_tokens) - 1
                        found_replacement = True
                        break
                elif token == phrase:
                    processed_tokens.append(replacement)
                    found_replacement = True
                    break
            if not found_replacement:
                processed_tokens.append(token)

        cleaned_text = " ".join(processed_tokens)


        # --- Additional generic words/phrases to remove for better problem_reported consolidation ---
        # These are applied after initial tokenization and multi-word replacements
        generic_noise_patterns = [
            r'\bfor\b', r'\bof\b', r'\bstatus showing red\b', r'\brobo\b', r'\bcircle\b', r'\bissue\b',
            r'\bproblem with\b', r'\bissue with\b', r'\bunable to\b', r'\berror in\b', r'\bfacing issue with\b',
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bbe\b', r'\bbeen\b', r'\bbeing\b', # Forms of 'to be'
            r'\bthe\b', r'\ba\b', r'\ban\b', # Articles
            r'\band\b', r'\bor\b', r'\bbut\b', r'\bso\b', r'\bnor\b', # Conjunctions
            r'\bwith\b', r'\bin\b', r'\bon\b', r'\bat\b', r'\bto\b', r'\bfrom\b', r'\bby\b', r'\babout\b', r'\bthrough\b', # Prepositions
            r'\bget\b', r'\bgot\b', r'\bgetting\b', # Common verbs
            r'\bshowing\b', r'\bdisplaying\b', r'\bappearing\b', # Display-related verbs
            r'\bsystem\b', r'\bapplication\b', r'\bserver\b', r'\bdatabase\b', # Generic IT terms
            r'\bplease\b', r'\bkindly\b', r'\bcan you\b', r'\bwe are\b', r'\bwe have\b', # Common request/status phrases
            r'\bof\b', r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b', # Demonstratives
            r'\bhas\b', r'\bhave\b', r'\bhad\b', # Forms of 'to have'
            r'\bdo\b', r'\bdoes\b', r'\bdid\b', # Forms of 'to do'
            r'\bcan\b', r'\bwill\b', r'\bwould\b', r'\bshould\b', r'\bcould\b', # Modals
            r'\bvery\b', r'\bquite\b', r'\bjust\b', r'\balmost\b', # Focus adverbs
            r'\bthere\b', r'\bhere\b', # Adverbs of place
            r'\bwhen\b', r'\bwhere\b', r'\bwhy\b', r'\bhow\b', # Wh-words
            r'\beven\b', r'\bonly\b', r'\bjust\b', r'\balmost\b', # Focus adverbs
            r'\bable\b', r'\bunable\b', # Ability
            r'\bnot\b', # Negation
            r'\bcall\b', r'\bcalls\b', # Specific to call issues
            r'\breport\b', r'\breports\b', # Specific to report issues
            r'\bpassword\b', r'\bpasswords\b', # Specific to password issues
            r'\breset\b', r'\bresetting\b', # Specific to reset actions
            r'\bthe\s+number\b', r'\bnumber\s+is\b', # Number related phrases
            r'\d+' # Remove standalone numbers (not part of a standardized phrase like IT-XXXX)
        ]
        for pattern in generic_noise_patterns:
            cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)

        # Remove any remaining non-alphanumeric characters (except spaces and underscore for standardized phrases)
        cleaned_text = re.sub(r'[^a-z0-9\s_]', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text

convosense_utilities = MockConvosenseUtilities()
# --- End Mock Convosense Utilities ---


# --- Helper Formatting Functions ---

def format_simple_numbered_list(values):
    """
    Formats a list of values into a numbered string (e.g., "1. ValueA\n2. ValueB").
    Handles cases where 'values' might be a numpy array.
    Each item will be on a new line.
    """
    if len(values) == 0:
        return ""
    formatted_items = []
    for i, val in enumerate(values):
        # Ensure value is treated as string to avoid errors with non-string types.
        formatted_items.append(f"{i + 1}. {str(val).strip()}") # .strip() to clean whitespace
    return "\n".join(formatted_items)

def create_docket_sequence_map(unique_dockets_in_group):
    """
    Creates a mapping from actual docket numbers to their sequence numbers (1, 2, 3...)
    within a consolidated group, and returns the formatted docket string.
    This function now only returns the sequence map and the docket string without a category prefix.
    """
    docket_to_seq_num_map = {}
    formatted_dockets = []
    sorted_dockets = sorted(list(unique_dockets_in_group)) # Ensure consistent order

    for i, docket in enumerate(sorted_dockets):
        seq_num = i + 1
        docket_to_seq_num_map[docket] = str(seq_num)
        # Format the docket string for the 'docket_no' column with just the sequence number
        formatted_dockets.append(f"{seq_num}. {docket}")
    return "\n".join(formatted_dockets), docket_to_seq_num_map

def clean_decoded_body(text):
    """
    Cleans the email body by aggressively truncating at the first occurrence of
    "Thanks", "Regards", or "Thanks and regards", then applying general signature
    removal, and finally normalizing whitespace.
    If the body becomes empty after all cleaning, it is considered "not meaningful".
    """
    if not isinstance(text, str):
        return ""

    cleaned_text = text.strip() # Start with original text

    # --- Phase 0: Initial Replacements for Newline Feature (before aggressive truncation) ---
    # Replace "snip" and "Disclaimer" with "Thanks and regards" (case-insensitive)
    # This ensures they are caught by the later truncation logic.
    cleaned_text = re.sub(r'\bsnip\b', 'Thanks and regards', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\bDisclaimer\b', 'Thanks and regards', cleaned_text, flags=re.IGNORECASE)

    # --- Aggressive Truncation: Remove everything from "Thanks" / "Regards" onwards ---
    # Find the first occurrence of any of these keywords (case-insensitive)
    truncation_patterns = [
        r'\bThanks\s*&\s*Regards\b',
        r'\bThanks\s+and\s+Regards\b',
        r'\bThanks\b',
        r'\bRegards\b',
    ]

    # Combine patterns into a single regex for efficiency
    combined_pattern = '|'.join(truncation_patterns)

    match = re.search(combined_pattern, cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        # Truncate the text at the start of the matched keyword
        cleaned_text = cleaned_text[:match.start()].strip()

    # --- Integrated Signature and System Report Removal Logic ---
    # These patterns were previously in MockEmailSignatureRemover.
    # They are now directly applied here to remove reply chains, disclaimers, and system reports.

    lines = cleaned_text.split('\n')

    # Patterns for common reply headers and signature start markers
    # These are usually at the beginning of a block to be removed
    reply_and_signature_block_starters = [
        r"^\s*On\s+.+?wrote:",
        r"^\s*-----Original Message-----",
        r"^\s*[\-_=]{5,}\s*Original Message[\-_=]{5,}",
        r"^\s*From:[\s\S]*?Sent:[\s\S]*?To:[\s\S]*?Subject:", # Outlook style header
        r"^\s*[\-_=]{3,}\s*Reply above this line\.",
        r"^\s*View request\s*·\s*Turn off this request's notifications",
        r"^\s*This is shared with",
        r"^\s*Powered by Jira Service Management",
        r"^\s*Sent on\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*::IST",
        r"^\s*A message that you sent could not be delivered",
        r"^\s*A message that you sent has not yet been delivered",
        r"^\s*CONFIDENTIALITY NOTICE:",
        r"^\s*NOTICE: This e-mail and any files transmitted with it are confidential and legally privileged",
        r"^\s*This email and any files transmitted with it are confidential and intended solely",
        r"^\s*LEGAL DISCLAIMER: The information in this email is confidential and may be legally privileged",
        r"^\s*----- Disclaimer -----",
        r"^\s*Hyper-V Environment Report:", # Added for system reports
        r"^\s*VCenter Environment Report:", # Added for system reports
        r"^\s*Cluster Overview \(VcenterCluster\)", # Added for system reports
        r"^\s*[\-_=]{5,}\s*$", # Generic separator line
    ]

    # Patterns for lines that are typically part of a signature, but might not be a "start" marker
    # These are used to identify lines to remove from the bottom up
    signature_content_patterns = [
        r"^\s*Best\s*Regards",
        r"^\s*With\s*Regards",
        r"^\s*Sincerely",
        r"^\s*Contact:",
        r"^\s*Email:",
        r"^\s*Mobile\s*number:",
        r"^\s*Handset:",
        r"^\s*TOC\s*Engineer",
        r"^\s*Shift\s*Lead",
        r"^\s*Operations",
        r"^\s*Central TOC",
        r"^\s*\[cid:image\d+\.png@\d+\.\d+\]", # Common image placeholders in signatures
        r"^\s*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", # URLs
        r"^\s*\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{4,9}\b", # Phone numbers
        r"^\s*[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", # Email addresses
        r"^\s*Piyush Kant|Chandan Maiti|Sagar Luthra|Arvind Mishra|Suraj Kumar Padhy|Manish Rai|SANJEEV BHARTI|Naveen Kumar", # Specific names
        # Added patterns for system report data
        r"^\s*Ipv4\s*:",
        r"^\s*Mac\s*:",
        r"^\s*Hard disk \d+=\d+\s*GB",
        r"^\s*DNS\s*:",
        r"^\s*poweredOn\s*\d+\s*Days",
        r"^\s*poweredOff\s*\d+\s*Days",
        r"^\s*\d+\s*CPU\s*\d+\s*GB",
        r"^\s*Update Available",
        r"^\s*UPTO Date",
        r"^\s*Total Nodes",
        r"^\s*Running Nodes",
        r"^\s*Logical Processors",
        r"^\s*Total Memory",
        r"^\s*Free Memory",
        r"^\s*Total Storage",
        r"^\s*Free Storage",
        r"^\s*Total VM",
        r"^\s*Running VM",
        r"^\s*vProcessor",
        r"^\s*vMemory",
        r"^\s*vStorage Used",
        r"^\s*vStorage",
        r"^\s*Clustered Disks/Volumes",
        r"^\s*Virtual Machines",
        r"^\s*Name\s+State\s+Uptime\s+Host\s+vCPU\s+vRAM\s+VMWare Tool Status", # Table header
        r"^\s*Generated on\s+.*IST\s+\d{4}", # Timestamp for reports
    ]

    # First pass: Find the highest line that indicates a reply/signature block start
    # Everything from this line downwards will be considered for removal.
    first_block_start_index = len(lines)
    for i, line in enumerate(lines):
        for pattern in reply_and_signature_block_starters:
            if re.search(pattern, line, flags=re.IGNORECASE | re.DOTALL):
                first_block_start_index = i
                break
        if first_block_start_index != len(lines):
            break

    # Consider lines up to this block start for further processing
    candidate_lines = lines[:first_block_start_index]

    # Second pass: From the end of the candidate lines, remove signature content
    # Iterate backwards to find the actual end of the message
    actual_message_end_index = len(candidate_lines) - 1
    for i in range(len(candidate_lines) - 1, -1, -1):
        line = candidate_lines[i]
        is_signature_content = False
        for pattern in signature_content_patterns:
            if re.search(pattern, line, flags=re.IGNORECASE | re.DOTALL):
                is_signature_content = True
                break

        # If this line is signature content OR it's an empty/whitespace line
        # AND it's at or near the end of the message, consider it part of the signature to be removed.
        # Stop when we hit a line that is clearly message content.
        if is_signature_content or not line.strip():
            actual_message_end_index = i - 1
        else:
            break # Found a non-signature, non-empty line, so this is likely the end of the message

    # Ensure actual_message_end_index doesn't go below -1
    cleaned_text = "\n".join(candidate_lines[:actual_message_end_index + 1])

    # --- Final Normalization (whitespace, newlines) ---
    # Normalize multiple newlines to at most two (representing a paragraph break)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)

    # Strip leading/trailing whitespace from each line, then from the whole text
    text_lines_processed = []
    for line in cleaned_text.split('\n'):
        stripped_line = line.strip()
        if stripped_line:
            text_lines_processed.append(stripped_line)
        elif line:
            text_lines_processed.append('')

    cleaned_text = "\n".join(text_lines_processed)
    cleaned_text = cleaned_text.strip()

    # --- Final check for "meaningful" content (after all cleaning) ---
    # If the text is now empty or only contains whitespace, return an "empty" string.
    # This is the ultimate filter for "mail body does not exist".
    if not cleaned_text:
        return ""

    return cleaned_text

def normalize_text_for_comparison(text):
    """
    Cleans and normalizes text for comparison purposes (used for both subjects and problem_reported).
    1. Removes common ticket/docket number patterns.
    2. Applies convosense_utilities refinement for further normalization.
    """
    if not isinstance(text, str):
        return ""

    cleaned_text = text.strip()

    # Regex to remove common ticket/docket number patterns
    ticket_number_patterns = [
        r'\bIT-\d{6,}\b',
        r'\bIT\d{6,}\b',
        r'\b\d{2}-\d{6}-\d{6}\b',
        r'\b\d{6,}\b',
        r'\b(?:ticket|docket|case)\s*#?\s*\d+\b',
        r'\b(?:sr|inc|req)\s*#?\s*\d+\b',
    ]

    for pattern in ticket_number_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # --- NEW: Apply convosense_utilities refinement for comprehensive normalization ---
    cleaned_text = convosense_utilities.refine_text(cleaned_text)

    return cleaned_text


def format_indexed_content_list(contents):
    """
    Formats a list of content strings with simple sequential numbering (1., 2., 3., etc.).
    It deduplicates the content before numbering.
    e.g., "1. Content A\n2. Content B"
    """
    if not contents:
        return ""

    unique_contents = sorted(list(set(filter(None, contents)))) # Deduplicate and sort

    formatted_items = []
    for i, content in enumerate(unique_contents):
        formatted_items.append(f"{i + 1}. {content}")
    return "\n".join(formatted_items)


def format_docket_specific_content_docket_centric(group_df, col_name, docket_to_seq_num_map, dockets_to_include_in_output):
    """
    Merges content (DecodedBody, Problem Reported) based on individual docket numbers,
    prefixing with the docket's sequence number (e.g., "1. Content").
    This function ensures only dockets present in `dockets_to_include_in_output` are included
    in the numbered list, and their content is formatted.
    This is for columns where each docket's content should be listed, not unique content across dockets.
    """
    docket_content_map = defaultdict(list)

    # Iterate through each row in the group DataFrame (rows from the similar-subject group)
    for index, row in group_df.iterrows():
        # Get individual dockets from this row's 'docket_no' cell
        row_dockets_str = str(row['docket_no'])
        row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

        # Apply specific cleaning based on column name
        content = str(row[col_name])
        # Note: clean_decoded_body is already applied to create 'DecodedBody_cleaned' upstream.
        # Re-applying it here for 'DecodedBody' ensures consistency if it wasn't pre-cleaned.
        if col_name == 'DecodedBody':
            content = clean_decoded_body(content)
        elif col_name == 'subject': # Subject is handled separately for single entry
            content = content.strip()
        else: # For other columns (like problem_reported), just strip whitespace
            content = content.strip()

        # Associate content with each individual docket in this row
        for docket in row_dockets:
            # Only add content if the docket is in the final set of dockets to include
            if docket in docket_to_seq_num_map and docket in dockets_to_include_in_output:
                docket_content_map[docket].append(content)

    formatted_output = []
    # Sort by docket sequence number for consistent output, but only for dockets that are in the final set
    sorted_dockets_for_output = sorted(list(dockets_to_include_in_output), key=lambda d: int(docket_to_seq_num_map[d]))

    for docket in sorted_dockets_for_output:
        seq_num = docket_to_seq_num_map[docket]
        # Get unique content for this specific docket and join them with spaces
        unique_content_for_docket = " ".join(sorted(list(set(filter(None, docket_content_map[docket])))))

        # Always append the docket number, even if content for this specific column is empty
        # because the docket itself has been determined to be meaningful by its DecodedBody.
        # Now formatted as just the sequential number (e.g., "1. Content")
        formatted_output.append(f"{seq_num}. {unique_content_for_docket}")

    return "\n".join(formatted_output)

def format_docket_content_grouping(group_df, content_col_name, docket_to_seq_num_map, dockets_to_include_in_output):
    """
    Formats content columns (like DecodedBody_cleaned) by grouping dockets that share
    the exact same content, prefixing with (docket_seq_nums) like attributes.
    Only includes dockets that are present in `dockets_to_include_in_output`.
    Format: "(1,2,3) Cleaned Content Text"
    """
    content_to_docket_seq_nums = defaultdict(set)

    for index, row in group_df.iterrows():
        row_dockets_str = str(row['docket_no'])
        row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

        # Directly use the content from the specified column, already assumed to be cleaned if it's 'DecodedBody_cleaned'
        content_value = str(row[content_col_name]).strip()

        if content_value: # Only consider non-empty content
            for docket in row_dockets:
                if docket in docket_to_seq_num_map and docket in dockets_to_include_in_output:
                    content_to_docket_seq_nums[content_value].add(docket_to_seq_num_map[docket])

    formatted_contents = []
    # Sort content values for consistent output order (e.g., alphabetically)
    # Filter out empty content_values that might have been added if a docket had no meaningful content
    sorted_content_values = sorted([k for k in content_to_docket_seq_nums.keys() if k])

    for content_val in sorted_content_values:
        docket_seq_nums_sorted = sorted(list(content_to_docket_seq_nums[content_val]), key=int)
        docket_nums_str = ", ".join(docket_seq_nums_sorted)
        formatted_contents.append(f"({docket_nums_str}) {content_val}")

    return "\n".join(formatted_contents)


def format_docket_attribute_grouping(group_df, attribute_col_name, docket_to_seq_num_map, dockets_to_include_in_output):
    """
    Formaats Priority, Disposition, Sub-disposition with associated docket sequence numbers.
    Only includes dockets that are present in `dockets_to_include_in_output`.
    Format: "(1,2,3) Semicritical\n(4,5,6) Non Critical"
    """
    attribute_to_docket_seq_nums = defaultdict(set) # Maps attribute value to a set of docket sequence numbers

    for index, row in group_df.iterrows():
        # Get individual dockets from this row's 'docket_no' cell
        row_dockets_str = str(row['docket_no'])
        row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

        attribute_value = str(row[attribute_col_name]).strip()

        # If attribute value is not 'nan' or empty, associate dockets with it
        if attribute_value and attribute_value.lower() != 'nan':
            for docket in row_dockets:
                # ONLY include this docket if it's in the master list of dockets with content
                if docket in docket_to_seq_num_map and docket in dockets_to_include_in_output:
                    attribute_to_docket_seq_nums[attribute_value].add(docket_to_seq_num_map[docket])

    formatted_attributes = []
    # Sort attribute values for consistent output order
    sorted_attribute_values = sorted(attribute_to_docket_seq_nums.keys())

    for attr_val in sorted_attribute_values:
        # Sort docket sequence numbers numerically
        docket_seq_nums_sorted = sorted(list(attribute_to_docket_seq_nums[attr_val]), key=int)
        docket_nums_str = ", ".join(docket_seq_nums_sorted)
        formatted_attributes.append(f"({docket_nums_str}){attr_val}")

    return "\n".join(formatted_attributes)


def consolidate_tickets(file_path, similarity_threshold=80, cosine_similarity_threshold=0.7):
    """
    Scans a CSV file, identifies rows with similar subjects (primary grouping),
    and then consolidates other columns based on docket numbers within those groups.
    It ensures that dockets with no meaningful DecodedBody content are entirely removed
    from the consolidated output row.

    Args:
        file_path (str): The path to the input CSV file.
        similarity_threshold (int): The minimum fuzzy matching score (0-100)
                                    to consider two subjects similar in the first pass.
        cosine_similarity_threshold (float): The minimum cosine similarity score (0-1)
                                             to consider two subjects similar in the second pass.

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

    # Ensure required columns exist, add if missing for robustness
    required_cols = ['docket_no', 'subject', 'problem_reported', 'priority_name',
                     'disposition_name', 'sub_disposition_name', 'DecodedBody',
                     'mail_list_id', 'mail_id', 'ticket_id', 'assigned_to_dept_name', 'assigned_to_user_name',
                     'DecodedBody_cleaned', 'normalized_problem', 'normalized_subject', 'full_comparison_text'] # Include cleaned/normalized columns
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding empty column.", file=sys.stderr)
            df[col] = ''

    # Fill any NaN values in the input DataFrame with empty strings before processing
    df = df.fillna('')

    # --- REMOVED: Duplicate filtering for "Re:/Fw:" subjects from here ---
    # This filtering now happens only in the main execution block before saving to temp_filtered_file_path.

    df = df.reset_index(drop=True)

    consolidated_rows_output = []
    processed_indices = set()

    consolidated_group_counter = 0 # Initialize consolidated_group_counter here

    # TF-IDF Vectorizer initialization (moved here as it's used for grouping)
    print("Calculating TF-IDF vectors for full comparison text...")
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=1, ngram_range=(1, 2))

    # Fit and transform only if there's meaningful text to vectorize
    if df['full_comparison_text'].astype(bool).any():
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['full_comparison_text'])
    else:
        print("Warning: All 'full_comparison_text' entries are empty after cleaning. Each row will form its own group.")
        from scipy.sparse import csr_matrix
        tfidf_matrix = None # Set to None to indicate no meaningful vectors for similarity

    print(f"Starting data consolidation based on cosine similarity (threshold: {cosine_similarity_threshold})...")
    groups = []

    for i in range(len(df)):
        if i in processed_indices:
            continue

        current_group_indices = {i}
        processed_indices.add(i)

        # Only perform similarity calculations if tfidf_matrix is not None and has features
        if tfidf_matrix is not None and tfidf_matrix.shape[1] > 0:
            current_vector = tfidf_matrix[i:i+1]

            unassigned_indices_list = [j for j in range(len(df)) if j not in processed_indices]
            if unassigned_indices_list:
                unassigned_vectors = tfidf_matrix[unassigned_indices_list]
                similarities = cosine_similarity(current_vector, unassigned_vectors).flatten()

                similar_items_indices_in_unassigned = np.where(similarities >= cosine_similarity_threshold)[0]

                for idx_in_unassigned_list in similar_items_indices_in_unassigned:
                    original_idx = unassigned_indices_list[idx_in_unassigned_list]
                    if original_idx not in processed_indices:
                        current_group_indices.add(original_idx)
                        processed_indices.add(original_idx)

        groups.append(df.loc[list(current_group_indices)])
    print(f"Finished data consolidation. Created {len(groups)} groups.")

    # Consolidate each group into a single row
    print("Aggregating consolidated groups...")
    consolidated_rows_output = []

    for group in groups:
        # Fill NaN values with empty strings before passing to formatting functions
        group_filled = group.fillna('')

        consolidated_row_data = {}

        # --- Generate Initial Docket Numbering and Map for ALL dockets in this group ---
        all_dockets_in_group = []
        for docket_cell in group_filled['docket_no'].astype(str):
            all_dockets_in_group.extend([d.strip() for d in docket_cell.split(';') if d.strip()])
        unique_dockets_in_group = set(all_dockets_in_group)

        _, final_docket_to_seq_num_map = create_docket_sequence_map(unique_dockets_in_group)

        # --- Determine the master filter set: dockets that have meaningful DecodedBody content ---
        # This determines which dockets are actually included in the final consolidated row.
        dockets_with_meaningful_body = set()

        for index, row in group_filled.iterrows():
            row_dockets_str = str(row['docket_no'])
            row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

            # Use the already cleaned body from the 'DecodedBody_cleaned' column
            cleaned_body = str(row['DecodedBody_cleaned'])

            if cleaned_body: # Only consider if the cleaned body is not empty
                for docket in row_dockets:
                    if docket in final_docket_to_seq_num_map:
                        dockets_with_meaningful_body.add(docket)

        # If no dockets in this group have a meaningful decoded body, skip this entire consolidated row
        if not dockets_with_meaningful_body:
            continue

        # --- Consolidate Subject: Show only one subject for the group, numbered as category index ---
        # Use the cleaned subject of the first item in the group as the representative subject
        representative_subject_normalized = group_filled.iloc[0]['normalized_subject']
        # Use a new consolidated group counter for the output subject numbering
        consolidated_group_counter += 1
        consolidated_row_data['subject'] = f"{consolidated_group_counter}. {representative_subject_normalized.capitalize()}"

        # --- Re-create the docket_no string using the final_docket_to_seq_num_map ---
        docket_no_output_lines = []
        sorted_dockets_for_output = sorted(list(dockets_with_meaningful_body), key=lambda d: int(final_docket_to_seq_num_map[d]))
        for docket in sorted_dockets_for_output:
            seq_num = final_docket_to_seq_num_map[docket]
            docket_no_output_lines.append(f"{seq_num}. {docket}")
        consolidated_row_data['docket_no'] = "\n".join(docket_no_output_lines)

        # --- Generate the DecodedBody string (now docket-centric numbering) ---
        # Collect ALL original DecodedBody entries from the group_df, filtered by meaningful dockets,
        # and format them using the docket-centric function.
        consolidated_row_data['DecodedBody'] = format_docket_specific_content_docket_centric(
            group_filled, 'DecodedBody', final_docket_to_seq_num_map, dockets_with_meaningful_body
        )

        # --- Consolidate DecodedBody_cleaned (now docket-centric numbering as well) ---
        consolidated_row_data['DecodedBody_cleaned'] = format_docket_content_grouping(
            group_filled, 'DecodedBody_cleaned', final_docket_to_seq_num_map, dockets_with_meaningful_body
        )

        # --- Consolidate Problem Reported (now using full two-stage similarity grouping) ---
        all_problems_for_group_raw = []
        # Collect all problem reports from the current subject-consolidated group
        for index, row in group_filled.iterrows():
            row_dockets_str = str(row['docket_no'])
            row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

            problem_reported_content = str(row['problem_reported']).strip()

            # Only add if the problem is meaningful and associated with a meaningful docket
            if problem_reported_content and any(docket in dockets_with_meaningful_body for docket in row_dockets):
                all_problems_for_group_raw.append(problem_reported_content)

        unique_problem_reports_after_grouping = []
        if all_problems_for_group_raw:
            # Normalize all problem reports for internal grouping
            # Store original index to retrieve original text later
            normalized_problems_with_original_indices = [
                (normalize_text_for_comparison(p), idx)
                for idx, p in enumerate(all_problems_for_group_raw)
            ]

            # Filter out empty strings after normalization
            non_empty_normalized_problems_with_indices = [
                (p_norm, orig_idx) for p_norm, orig_idx in normalized_problems_with_original_indices if p_norm
            ]

            if non_empty_normalized_problems_with_indices:
                # Extract just the normalized texts for TF-IDF
                normalized_texts_for_tfidf = [p_norm for p_norm, _ in non_empty_normalized_problems_with_indices]

                problem_vectorizer = TfidfVectorizer()
                problem_tfidf_matrix = problem_vectorizer.fit_transform(normalized_texts_for_tfidf)

                problem_processed_indices_for_grouping = set()

                for prob_idx, (current_prob_normalized, original_raw_idx) in enumerate(non_empty_normalized_problems_with_indices):
                    if prob_idx in problem_processed_indices_for_grouping:
                        continue

                    # Start a new sub-group for problem reports
                    current_problem_sub_group_indices = {prob_idx}

                    # Fuzzy matching for problem reports
                    for other_prob_idx in range(len(non_empty_normalized_problems_with_indices)):
                        if other_prob_idx in problem_processed_indices_for_grouping or other_prob_idx == prob_idx:
                            continue

                        other_prob_normalized = non_empty_normalized_problems_with_indices[other_prob_idx][0]
                        fuzzy_score_prob = fuzz.token_sort_ratio(current_prob_normalized, other_prob_normalized)
                        if fuzzy_score_prob >= similarity_threshold: # Using the same similarity_threshold as subjects
                            current_problem_sub_group_indices.add(other_prob_idx)

                    # Cosine similarity for problem reports (for remaining unprocessed in this sub-group)
                    unprocessed_in_problem_pass = [
                        idx for idx in range(len(non_empty_normalized_problems_with_indices))
                        if idx not in problem_processed_indices_for_grouping and idx != prob_idx
                    ]

                    if unprocessed_in_problem_pass:
                        current_prob_vector = problem_tfidf_matrix[prob_idx:prob_idx+1]

                        for other_prob_idx_raw_in_normalized_list in unprocessed_in_problem_pass:
                            other_prob_vector = problem_tfidf_matrix[other_prob_idx_raw_in_normalized_list:other_prob_idx_raw_in_normalized_list+1]
                            cos_sim_prob = cosine_similarity(current_prob_vector, other_prob_vector)[0][0]

                            if cos_sim_prob >= cosine_similarity_threshold: # Using the same cosine_similarity_threshold
                                current_problem_sub_group_indices.add(other_prob_idx_raw_in_normalized_list)

                    # Select a representative problem report from this sub-group
                    # Choose the original problem report corresponding to the first index in the sub-group
                    # (based on the original order of all_problems_for_group_raw, after filtering to non_empty)

                    # Get the original indices from the non_empty_normalized_problems_with_indices list
                    original_indices_in_raw_for_sub_group = sorted([
                        non_empty_normalized_problems_with_indices[idx][1] for idx in current_problem_sub_group_indices
                    ])

                    if original_indices_in_raw_for_sub_group:
                        representative_original_problem = all_problems_for_group_raw[original_indices_in_raw_for_sub_group[0]]
                        unique_problem_reports_after_grouping.append(representative_original_problem)

                    for idx_to_mark in current_problem_sub_group_indices:
                        problem_processed_indices_for_grouping.add(idx_to_mark)

            consolidated_row_data['problem_reported'] = format_indexed_content_list(unique_problem_reports_after_grouping)


            # --- Consolidate attributes with (seq,seq)Value format, filtering by dockets with content ---
            # These remain docket-centric as per previous discussions.
            consolidated_row_data['priority_name'] = format_docket_attribute_grouping(group_filled, 'priority_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)
            consolidated_row_data['disposition_name'] = format_docket_attribute_grouping(group_filled, 'disposition_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)
            consolidated_row_data['sub_disposition_name'] = format_docket_attribute_grouping(group_filled, 'sub_disposition_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)

            # --- Handle remaining columns with simple 1. 2. numbering, filtering by dockets with content ---
            all_original_columns = df.columns.tolist()
            for col in all_original_columns:
                # Skip columns already processed above
                if col not in consolidated_row_data:
                    temp_col_values_map = defaultdict(list)
                    for idx_in_group, row_in_group in group_filled.iterrows():
                        row_dockets_str = str(row_in_group['docket_no'])
                        row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]

                        for docket in row_dockets:
                            if docket in dockets_with_meaningful_body: # Only collect for dockets with meaningful body
                                value = str(row_in_group[col]).strip()
                                if value: # Only add non-empty values
                                    temp_col_values_map[docket].append(value)

                    # Now format based on the final_docket_to_seq_num_map
                    col_output_lines = []
                    sorted_dockets_for_col_output = sorted(list(dockets_with_meaningful_body),
                                                           key=lambda d: int(final_docket_to_seq_num_map[d]))
                    for docket in sorted_dockets_for_col_output:
                        seq_num = final_docket_to_seq_num_map[docket]
                        unique_content = " ".join(sorted(list(set(filter(None, temp_col_values_map[docket])))))
                        # Always append the docket number, even if content for this specific column is empty
                        # because the docket itself has been determined to be meaningful by its DecodedBody.
                        # Now formatted as just the sequential number (e.g., "1. Content")
                        col_output_lines.append(f"{seq_num}. {unique_content}")

                    consolidated_row_data[col] = "\n".join(col_output_lines)

            # Add the fully formed consolidated row to the output list
            # Ensure all original columns are present in the final row, filling with empty if not consolidated
            final_row_dict = {col: consolidated_row_data.get(col, '') for col in all_original_columns}
            consolidated_rows_output.append(final_row_dict)

    # Create a new DataFrame from the consolidated rows
    all_original_columns = df.columns.tolist() # Re-get columns in case any were added
    consolidated_df = pd.DataFrame(consolidated_rows_output, columns=all_original_columns)

    # Final fillna for any cells that might still be NaN (should be rare with current logic)
    consolidated_df = consolidated_df.fillna('')

    return consolidated_df

# --- Main execution ---
# Define the path to the uploaded CSV file
file_path = 'fulldata.csv'
output_file_path = 'full_processed_Data.csv'
temp_filtered_file_path = 'temp_filtered_tickets.csv' # Temporary file for filtered data

# Set your desired similarity thresholds
# A higher threshold means stricter similarity.
SIMILARITY_THRESHOLD = 80 # For fuzzy matching (token_sort_ratio)
COSINE_SIMILARITY_THRESHOLD = 0.7 # For cosine similarity (0.0 to 1.0)

print(f"Loading original tickets from '{file_path}'...")
try:
    original_df = pd.read_csv(file_path)
    # Fill any NaN values in the original DataFrame with empty strings before initial filtering
    original_df = original_df.fillna('')
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please ensure the file exists.", file=sys.stderr)
    sys.exit(1) # Exit if original file not found
except Exception as e:
    print(f"Error reading original CSV file: {e}", file=sys.stderr)
    sys.exit(1)

# --- NEW: Filter out rows containing "This message was created automatically by mail delivery software." ---
mail_delivery_software_pattern = r"This message was created automatically by mail delivery software\."
initial_rows_before_mail_filter = len(original_df)
original_df = original_df[~original_df['DecodedBody'].astype(str).str.contains(mail_delivery_software_pattern, case=False, na=False)].copy()
mail_filtered_rows_count = initial_rows_before_mail_filter - len(original_df)
if mail_filtered_rows_count > 0:
    print(f"Filtered out {mail_filtered_rows_count} rows due to 'mail delivery software' messages.")
else:
    print("No rows found with 'mail delivery software' messages to filter.")


# --- Step 1: Filter out rows where subject starts with "Re:", "Fw:", etc. ---
print("Filtering out rows with 'Re:'/'Fw:' subjects...")
initial_row_count_after_mail_filter = len(original_df)

# Regex to match subjects starting with common reply/forward prefixes (case-insensitive)
reply_forward_pattern = r"^(Re|Fw|REF|FWD):\s*"

# Filter rows where the subject does NOT start with these patterns
filtered_df = original_df[~original_df['subject'].astype(str).str.contains(reply_forward_pattern, case=False, na=False)].copy()

if len(filtered_df) < initial_row_count_after_mail_filter:
    print(f"Filtered out {initial_row_count_after_mail_filter - len(filtered_df)} rows due to 'Re:'/'Fw:' subjects.")
else:
    print("No rows found with 'Re:'/'Fw:' subjects to filter.")

# --- Step 2: Save the filtered data to a temporary file ---
print(f"Saving filtered data to temporary file: '{temp_filtered_file_path}'...")
try:
    # Apply cleaning to DecodedBody and normalize subject/problem_reported for comparison
    # These columns are needed for the TF-IDF and cosine similarity in consolidate_tickets
    filtered_df['DecodedBody_cleaned'] = filtered_df['DecodedBody'].apply(clean_decoded_body)
    filtered_df['normalized_problem'] = filtered_df['problem_reported'].apply(normalize_text_for_comparison)
    filtered_df['normalized_subject'] = filtered_df['subject'].apply(normalize_text_for_comparison)

    # Combine normalized texts for comprehensive similarity comparison
    filtered_df['full_comparison_text'] = filtered_df.apply(
        lambda row: ' '.join(filter(None, [row['normalized_problem'], row['normalized_subject'], row['DecodedBody_cleaned']])),
        axis=1
    )
    filtered_df['full_comparison_text'] = filtered_df['full_comparison_text'].replace('', '__EMPTY_TEXT__')

    filtered_df.to_csv(temp_filtered_file_path, index=False)
    print("Temporary file saved successfully.")
except Exception as e:
    print(f"Error saving temporary filtered data to CSV: {e}", file=sys.stderr)
    sys.exit(1) # Exit if temporary file cannot be saved

# --- Step 3: Process the temporary file for consolidation ---
print(f"Consolidating tickets from temporary file '{temp_filtered_file_path}' based on similar subjects (fuzzy threshold={SIMILARITY_THRESHOLD}, cosine threshold={COSINE_SIMILARITY_THRESHOLD})...")
consolidated_df = consolidate_tickets(temp_filtered_file_path, SIMILARITY_THRESHOLD, COSINE_SIMILARITY_THRESHOLD)

if not consolidated_df.empty:
    try:
        consolidated_df.to_csv(output_file_path, index=False)
        print(f"Consolidation complete! Consolidated data saved to '{output_file_path}'")
        print(f"\nTotal original rows (before 'mail delivery software' filter): {initial_rows_before_mail_filter}")
        print(f"Total rows after 'mail delivery software' filter: {initial_row_count_after_mail_filter}")
        print(f"Total rows after 'Re:'/'Fw:' filter: {len(filtered_df)}")
        print(f"Total consolidated rows: {len(consolidated_df)}")
        print("\n--- First 5 rows of Consolidated Data (Preview of Key Columns) ---")

        # Define key columns for preview
        preview_cols = [
            'docket_no',
            'subject',
            'problem_reported',
            'priority_name',
            'disposition_name',
            'sub_disposition_name',
            'DecodedBody', # Show original DecodedBody
            'DecodedBody_cleaned' # Show cleaned DecodedBody
        ]

        # Filter to only show columns that actually exist in the consolidated DataFrame
        existing_preview_cols = [col for col in preview_cols if col in consolidated_df.columns]

        if existing_preview_cols:
            print(consolidated_df[existing_preview_cols].head().to_markdown(index=False))
        else:
             print(consolidated_df.head().to_markdown(index=False)) # Fallback if none of the key columns exist
        print("\n(Note: Full content for all columns, including DecodedBody, is in the CSV file. Enable 'Wrap Text' in spreadsheet for full view.)")

    except Exception as e:
        print(f"Error saving consolidated data to CSV: {e}", file=sys.stderr)
else:
    print("No data to consolidate or an error occurred during processing.")

# --- Cleanup: Remove the temporary file ---
# if os.path.exists(temp_filtered_file_path):
#     try:
#         os.remove(temp_filtered_file_path)
#         print(f"Temporary file '{temp_filtered_file_path}' removed.")
#     except Exception as e:
#         print(f"Error removing temporary file '{temp_filtered_file_path}': {e}", file=sys.stderr)

# --- Test Block for clean_decoded_body function ---
print("\n--- Testing clean_decoded_body with provided examples ---")

# Example 1: Merged Mail Body
email_message_test_1 = '''
Dear C-zentrix Team, I'm pleased to inform you that the issue has been resolved. Please close the ticket. Thanks
----- Original Message ----- From: Helpdesk Tvt To: Sent: Wed, 31 Jan 2024 14:18:47 +0530 Subject: Urgent: Admin Panel Login Issue G
'''
cleaned_text_test_1 = clean_decoded_body(email_message_test_1)
print("\nInput Email 1:")
print(email_message_test_1)
print("\nCleaned Text 1:")
print(cleaned_text_test_1)

# Example 2: Another Sample Mail Body
email_message_test_2 = '''
Hi Team, Please extend inbound call timings till 11 PM today and confirm asap. Regards
, Naveen Kumar Manager – Setu Desk HO Mobile No - 8447172738 Maxlifeinsurace Our Values drive Our Culture Caring | Collaboration Customer Obsession | Growth Mindset Thanks
and Regards
: This email contains confidential information intended solely for the recipient, and if received by mistake, it should not be distributed or copied; please notify the sender and delete all copies. WARNING: Computer viruses can be transmitted via email. You are advised to check for viruses in the email and attachments, with the company disclaiming liability for any resulting damage.
'''
cleaned_text_test_2 = clean_decoded_body(email_message_test_2)
print("\nInput Email 2:")
print(email_message_test_2)
print("\nCleaned Text 2:")
print(cleaned_text_test_2)

# Example 3: System Report Data
email_message_test_3 = '''
192.168.99.78 Mac : 00:50:56:8f:80:4f Ipv4 : 172.16.99.78 Mac : 00:50:56:8f:22:65 Mac : 00:50:56:8f:91:e9 Hard disk 1=200 GB 35980-BLR-01-V01-0062 DNS : localhost.localdomain poweredOn 114 Days PD05VPC-TIER0001-03 5 CPU 4 GB Update Available Update Available NA Mac : 02:42:8f:75:ec:7d Ipv4 : 172.18.0.1 Mac : 00:50:56:8f:78:49 Ipv4 : 192.168.99.54 Mac : 00:50:56:8f:c7:40 Ipv4 : 172.16.99.54 Mac : 00:50:56:8f:52:7f Mac : 00:50:56:8f:fb:ea Hard disk 1=100 GB 35980-BLR-01-V23-0212 DNS : JANUS-04 poweredOn 114 Days PD05VPC-TIER0001-03 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:67:38 Ipv4 : 192.168.99.91 Mac : 00:50:56:8f:d1:f1 Ipv4 : 172.16.99.91 Mac : 00:50:56:8f:b3:97 Mac : 00:50:56:8f:a0:10 Hard disk 1=200 GB 35980-BLR-01-V23-0197 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-03 4 CPU 8 GB UPTO Date Update Available NA NA Hard disk 1=100 GB 35980-BLR-01-V23-0035 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-03 4 CPU 8 GB UPTO Date UPTO Date NA NA Hard disk 1=100 GB Hard disk 2=500 GB 35980-BLR-01-V23-0034 DNS : ntt-ftp-blr-node poweredOn 103 Days PD05VPC-TIER0001-03 2 CPU 12 GB Update Available Update Available NA Mac : 00:50:56:8f:32:4a Ipv4 : 192.168.99.26 Mac : 00:50:56:8f:81:e9 Ipv4 : 172.16.99.26 Mac : 00:50:56:8f:31:ff Ipv4 : 192.168.31.24 Mac : 00:50:56:8f:25:d2 Hard disk 1=200 GB 35980-BLR-01-V23-0047 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-03 4 CPU 8 GB UPTO Date Update Available NA NA Hard disk 1=100 GB Hard disk 2=100 GB This email and all contents are subject to the following disclaimer: https://services.global.ntt/en-us/email-disclaimer Hyper-V Environment Report: Towards Vision Technologies Private Limited VCenter Environment Report: Towards Vision Technologies Private Limited Generated on Fri Jan 12 05:45:36 IST 2024 Cluster Overview (VcenterCluster) Name Total Nodes Running Nodes Logical Processors Total Memory Free Memory Total Storage Free Storage Total VM Running VM vProcessor Total vMemory Total vStorage Used vStorage 35980-SMVPC-BLRDC2-TVTPL-01 4 4 80 509 GB 166 GB 23.06 TB 10.41 TB 58 48 152 518 GB 27534 GB 14472 GB Cluster Nodes Name State Uptime Domain Total VM Running VM Active vProcessor Logical Processor Used Memory Free Memory Total Memory Free Memory (%) pd05vpc-tier0001-02 Up 103 Days pd05vpc-tier0001-02.smplcldblrgrd.com 13 12 20 40 83 GB 44 GB 127 GB 34 % PD05VPC-TIER0001-01 Up 133 Days PD05VPC-TIER0001-01.SMPLCLDBLRGRD.COM 11 11 20 40 88 GB 39 GB 127 GB 30 % PD05VPC-TIER0001-04 Up 103 Days PD05VPC-TIER0001-04.SMPLCLDBLRGRD.COM 14 8 20 40 90 GB 37 GB 127 GB 29 % PD05VPC-TIER0001-03 Up 103 Days PD05VPC-TIER0001-03.SMPLCLDBLRGRD.COM 20 17 20 40 91 GB 36 GB 127 GB 28 % Clustered Disks/Volumes Name State File System Total VM Used Size Free Size Total Size Free Size (%) 35980-SMVPC-BLRDC2-TVTPL-DS01 normal VMFS 58 12934 GB 9593 GB 22527 GB 42 % Virtual Machines Name State Uptime Host vCPU vRAM VMWare Tool Status VMWare Hardware Status Snapshot Status IP/Mac Address Disk 35980-BLR-01-V23-0068 DNS : NA poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 16 GB UPTO Date Update Available NA NA Hard disk 1=300 GB 35980-BLR-01-V24-0214 DNS : omni-asia-chat-01 poweredOn 51 Days pd05vpc-tier0001-02 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:71:11 Ipv4 : 192.168.99.93 Mac : 00:50:56:8f:85:ed Ipv4 : 172.16.99.93 Mac : 00:50:56:8f:f6:0e Ipv4 : 192.168.31.48 Mac : 00:50:56:8f:3b:da Ipv4 : 192.168.41.48 Hard disk 1=100 GB 35980-BLR-01-V23-0065 DNS : ocms poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:a9:97 Ipv4 : 192.168.99.95 Mac : 00:50:56:8f:f9:1d Ipv4 : 172.16.99.95 Mac : 00:50:56:8f:1b:b1 Mac : 00:50:56:8f:ed:cd Hard disk 1=100 GB 35980-BLR-01-V23-0028 DNS : cz-cloud-ws-02 poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:1a:b8 Ipv4 : 192.168.99.21 Mac : 00:50:56:8f:25:62 Ipv4 : 172.16.99.21 Mac : 00:50:56:8f:7b:2a Ipv4 : 192.168.31.19 Mac : 00:50:56:8f:3d:fb Ipv4 : 192.168.41.19 Hard disk 1=100 GB 35980-BLR-01-V23-0006 DNS : janus-02 poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:1f:d5 Ipv4 : 192.168.99.202 Mac : 00:50:56:8f:47:49 Ipv4 : 172.16.99.202 Mac : 00:50:56:8f:8f:2f Mac : 00:50:56:8f:eb:87 Hard disk 1=200 GB 35980-BLR-01-V01-0001 DNS : HAproxy-WebRTC poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:42:ee Ipv4 : 192.168.99.3 Ipv4 : 192.168.99.203 Mac : 00:50:56:8f:5b:14 Ipv4 : 172.16.99.3 Mac : 00:50:56:8f:d2:94 Mac : 00:50:56:8f:d6:f7 Hard disk 1=100 GB 35980-BLR-01-V23-0213 DNS : vlcc-cz-cloud-02 poweredOn 90 Days pd05vpc-tier0001-02 1 CPU 2 GB Update Available Update Available NA Mac : 00:50:56:8f:42:51 Ipv4 : 192.168.99.92 Mac : 00:50:56:8f:7c:cf Ipv4 : 172.16.99.92 Mac : 00:50:56:8f:b7:35 Mac : 00:50:56:8f:48:0d Hard disk 1=100 GB Hard disk 2=300 GB 35980-BLR-01-V23-0058 DNS : cz-cloud-ws-03 poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:b6:ae Ipv4 : 192.168.99.50 Mac : 00:50:56:8f:e6:f4 Ipv4 : 172.16.99.50 Mac : 00:50:56:8f:84:2e Mac : 00:50:56:8f:f4:2a Hard disk 1=100 GB 35980-BLR-01-V23-0159 DNS : NA poweredOff 0 Days pd05vpc-tier0001-02 2 CPU 4 GB Update Available Update Available NA NA Hard disk 1=100 GB 35980-BLR-01-V23-0067 DNS : service poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:a9:3d Ipv4 : 192.168.99.59 Mac : 00:50:56:8f:0e:aa Ipv4 : 172.16.99.59 Mac : 00:50:56:8f:f8:e0 Mac : 00:50:56:8f:37:09 Hard disk 1=100 GB 35980-BLR-01-V23-0061 DNS : vlcc-cz-cloud poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:b7:57 Ipv4 : 192.168.99.53 Mac : 00:50:56:8f:9e:59 Ipv4 : 172.16.99.53 Mac : 00:50:56:8f:f0:c0 Mac : 00:50:56:8f:31:9d Hard disk 1=100 GB Hard disk 2=300 GB 35980-BLR-01-V23-0007 DNS : cz-te-db-01 poweredOn 103 Days pd05vpc-tier0001-02 3 CPU 32 GB Update Available Update Available NA Mac : 00:50:56:8f:b9:1a Ipv4 : 192.168.99.101 Mac : 00:50:56:8f:0f:01 Ipv4 : 172.16.99.101 Mac : 00:50:56:8f:2b:b3 Mac : 00:50:56:8f:59:aa Hard disk 1=200 GB Hard disk 2=500 GB 35980-BLR-01-V23-0016 DNS : Bang_ACD_GW8 poweredOn 103 Days pd05vpc-tier0001-02 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:52:ed Ipv4 : 192.168.99.16 Mac : 00:50:56:8f:1c:4f Ipv4 : 172.16.99.16 Mac : 00:50:56:8f:9d:eb Mac : 00:50:56:8f:72:98 Hard disk 1=100 GB Hard disk 2=500 GB 35980-BLR-01-V23-0172 DNS : voice-bot-proxy poweredOn 119 Days PD05VPC-TIER0001-01 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:ee:3f Ipv4 : 192.168.99.79 Mac : 00:50:56:8f:0b:6a Ipv4 : 172.16.99.79 Mac : 00:50:56:8f:e8:0f Mac : 00:50:56:8f:51:58 Hard disk 1=100 GB 35980-BLR-01-V23-0056 DNS : Bang_ACD_GW1 poweredOn 133 Days PD05VPC-TIER0001-01 1 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:31:f2 Ipv4 : 192.168.99.42 Mac : 00:50:56:8f:c5:e9 Ipv4 : 172.16.99.42 Mac : 00:50:56:8f:8a:c6 Mac : 00:50:56:8f:f7:24 Hard disk 1=200 GB Hard disk 2=300 GB 35980-BLR-01-V23-0055 DNS : cz-te-db-03 poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 32 GB Update Available Update Available NA Mac : 00:50:56:8f:de:08 Ipv4 : 192.168.99.103 Mac : 00:50:56:8f:ec:af Ipv4 : 172.16.99.103 Mac : 00:50:56:8f:2f:71 Mac : 00:50:56:8f:71:ca Hard disk 1=200 GB Hard disk 2=500 GB 35980-BLR-01-V23-0075 DNS : cz-guide-app-01 poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:73:bd Ipv4 : 192.168.99.62 Mac : 00:50:56:8f:78:71 Ipv4 : 172.16.99.62 Mac : 00:50:56:8f:ae:32 Mac : 00:50:56:8f:03:3c Hard disk 1=250 GB 35980-BLR-01-V23-0027 DNS : cz-cloud-ws-01 poweredOn 133 Days PD05VPC-TIER0001-01 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:fe:06 Ipv4 : 192.168.99.20 Mac : 00:50:56:8f:15:1b Ipv4 : 172.16.99.20 Mac : 00:50:56:8f:b6:5a Ipv4 : 192.168.31.18 Mac : 00:50:56:8f:81:49 Ipv4 : 192.168.41.18 Hard disk 1=100 GB 35980-BLR-01-V23-0014 DNS : hawebrtc_2 poweredOn 119 Days PD05VPC-TIER0001-01 1 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:9e:81 Ipv4 : 192.168.99.15 Mac : 00:50:56:8f:e7:04 Ipv4 : 172.16.99.15 Mac : 00:50:56:8f:36:9a Mac : 00:50:56:8f:93:65 Hard disk 1=300 GB 35980-BLR-01-V23-0026 DNS : czentrix-uat-vbot-asr-01 poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:c0:b1 Ipv4 : 192.168.99.19 Mac : 00:50:56:8f:11:fb Ipv4 : 172.16.99.19 Mac : 00:50:56:8f:5a:59 Ipv4 : 192.168.31.17 Mac : 00:50:56:8f:93:39 Ipv4 : 192.168.41.17 Hard disk 1=100 GB 35980-BLR-01-V23-0069 DNS : DB4-Shared poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:70:d4 Ipv4 : 192.168.99.61 Mac : 00:50:56:8f:0c:f4 Ipv4 : 172.16.99.61 Mac : 00:50:56:8f:d2:66 Mac : 00:50:56:8f:69:2e Hard disk 1=100 GB Hard disk 2=500 GB 35980-BLR-01-V23-0066 DNS : omni-asia-chat-01 poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:45:4f Ipv4 : 192.168.99.58 Mac : 00:50:56:8f:fe:95 Ipv4 : 172.16.99.58 Mac : 00:50:56:8f:e1:e0 Mac : 00:50:56:8f:4f:35 Hard disk 1=100 GB 35980-BLR-01-V23-0173 DNS : bot-nlp poweredOn 83 Days PD05VPC-TIER0001-01 2 CPU 14 GB Update Available Update Available NA Mac : 00:50:56:8f:a2:36 Ipv4 : 192.168.99.80 Mac : 00:50:56:8f:7b:9c Ipv4 : 172.16.99.80 Mac : 00:50:56:8f:6c:0d Mac : 00:50:56:8f:ce:67 Hard disk 1=100 GB 35980-BLR-01-V23-0018 DNS : czentrix-uat-vbot-ast18-01 poweredOn 103 Days PD05VPC-TIER0001-01 2 CPU 4 GB Update Available Update Available NA Mac : 00:50:56:8f:84:15 Ipv4 : 192.168.99.18 Mac : 00:50:56:8f:c7:05 Ipv4 : 172.16.99.18 Mac : 00:50:56:8f:19:a2 Mac : 00:50:56:8f:bc:e7 Hard disk 1=100 GB 35980-BLR-01-V23-0215 DNS : ocms poweredOn 16 Days PD05VPC-TIER0001-04 2 CPU 6 GB Update Available Update Available NA Mac : 00:50:56:8f:0c:3e Ipv4 : 192.168.99.57 Mac : 00:50:56:8f:01:5a Ipv4 : 172.16.99.57 Mac : 00:50:56:8f:88:7b Mac : 00:50:56:8f:23:53 Hard disk 1=100 GB 35980-BLR-01-V23-0215_old_not-working DNS : NA poweredOff 0 Days PD05VPC-TIER0001-04 2 CPU 4 GB Update Available Update Available NA NA Hard disk 1=100 GB 35980-BLR-01-V23-0196 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-04 4 CPU 8 GB UPTO Date Update Available NA NA Hard disk 1=100 GB 35980-BLR-01-V23-0195 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-04 16 CPU 16 GB UPTO Date Update Available NA NA Hard disk 1=200 GB 35980-BLR-01-V23-0194 DNS : NA poweredOff 0 Days PD05VPC-TIER0001-04 16 CPU 16 GB UPTO Date Update Available NA NA Hard disk 1=200 GB 35980-BLR-01-V23-0174 DNS : OBD-CMS poweredOn 93 Days PD05VPC-TIER0001-04 2 CPU 8 GB Update Available Update Available NA Mac : 00:50:56:8f:de:a4 Ipv4 : 192.168.99.71 Mac : 00:50:56:8f:7c:07 Ipv4 : 172.16.99.71 Mac : 00:50:56:8f:00:6d Ipv4 : 192.168.31.28 Mac : 00:50:56:8f:13:7b Hard disk 1=200 GB 35980-BLR-01-V23-0052 DNS : cz-cloud-dialer poweredOn 103 Days PD05VPC-TIER0001-04 2 CPU 16 GB Update Available Update Available NA Mac : 00:50:56:8f:93:3a Ipv4 : 192.168.99.45 Mac : 00:50:56:8f:b8:29 Ipv4 : 172.16.99.45 Mac : 00:50:56:8f:15:bb Mac : 00:50:56:8f:5b:80 Hard disk 1=100 GB Hard disk 2=300 GB 35980-BLR-01-V23-0059 DNS : cz-clo
'''
