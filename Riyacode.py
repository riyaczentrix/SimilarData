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

# Attempt to import talon components. If this fails, the clean_decoded_body will use regex fallback.
try:
    from talon import quotations
    from talon.signature.bruteforce import extract_signature
    TALON_AVAILABLE = True
    # print("Talon library (quotations and signature modules) imported successfully. Will use for cleaning.")
except ImportError:
    TALON_AVAILABLE = False
    # print("Talon library not available. Falling back to regex-based email body cleaning.")
except Exception as e:
    TALON_AVAILABLE = False
    # print(f"Error importing Talon library: {e}. Falling back to regex-based email body cleaning.")


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
    Cleans the email body by first attempting to remove reply chains, signatures, and disclaimers
    using Talon. If Talon is unavailable or ineffective for a specific block, it falls back to
    aggressive regex patterns for common signature/disclaimer markers, removing all subsequent text.
    Finally, as a last resort, explicit PII redaction regexes are applied to any remaining text.
    Whitespace is normalized throughout.
    If the body becomes empty after all cleaning, it is considered "not meaningful".
    """
    if not isinstance(text, str):
        return ""
    
    original_text = text.strip()
    cleaned_text = original_text # Start with original text

    # --- Phase 1: Attempt with Talon for signatures and quoted replies ---
    text_after_talon_attempt = cleaned_text
    if TALON_AVAILABLE:
        try:
            # Use talon.quotations to remove quoted replies
            msg_without_quotes = quotations.extract_from(cleaned_text, 'text/plain')
            
            # Use talon.signature to extract and remove signature/disclaimer
            stripped_text_by_talon, signature = extract_signature(msg_without_quotes)
            
            # Prioritize talon's stripped text if it actually removed something
            # or if it identified a signature.
            if stripped_text_by_talon is not None and stripped_text_by_talon.strip() != msg_without_quotes.strip():
                text_after_talon_attempt = stripped_text_by_talon
            elif signature and signature.strip(): # If talon found a signature, use its stripped text
                text_after_talon_attempt = stripped_text_by_talon
            else: # If talon didn't remove signature, use text without quotes
                text_after_talon_attempt = msg_without_quotes
        except Exception as e:
            # If talon errors, fall back to the original text for regex processing
            text_after_talon_attempt = original_text 
    else:
        # Talon not available, proceed with original text for regex fallback
        text_after_talon_attempt = original_text 

    # --- Phase 2: Aggressive Regex Fallback for Signatures/Disclaimers and Auto-replies ---
    # This phase is executed if Talon was not available or did not remove the signature/disclaimer.
    # These patterns are designed to match the start of a signature/disclaimer/auto-reply block
    # and implicitly remove everything after it by setting the cut_off_index.
    
    lines = text_after_talon_attempt.split('\n')
    cut_off_index = len(lines) # Default to no cut-off

    aggressive_signature_disclaimer_patterns = [
        # --- Specific Auto-reply/Jira patterns ---
        r"^\s*[\-_=]{3,}\s*Reply above this line\.[\s\S]*?$", # Matches "--- Reply above this line." and everything after
        r"^\s*View request\s*·\s*Turn off this request's notifications[\s\S]*?$", # Matches Jira notification links
        r"^\s*This is shared with[\s\S]*?$", # Matches Jira sharing info
        r"^\s*Powered by Jira Service Management[\s\S]*?$", # Matches Jira footer
        r"^\s*Sent on January\s*::IST[\s\S]*?$", # Matches date/time stamp of auto-reply
        r"^\s*Sent on\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s*::IST[\s\S]*?$", # More robust date/time stamp

        # --- Mail Delivery Software / Bounce Message patterns ---
        # NOTE: These patterns are primarily for *cleaning* the body if the row wasn't filtered out earlier.
        # The main filtering of entire rows based on "This message was created automatically by mail delivery software"
        # happens at the DataFrame level in the main execution block.
        r"^\s*A message that you sent could not be delivered to one or more of its recipients\.[\s\S]*?$",
        r"^\s*A message that you sent has not yet been delivered to one or more of its recipients after more than.*?hours on the queue[\s\S]*?$",
        r"^\s*SMTP error from remote mail server after RCPT TO:[\s\S]*?$",
        r"^\s*Recipient disabled[\s\S]*?$",
        r"^\s*retry timeout exceeded[\s\S]*?$",
        r"^\s*Delay reason:[\s\S]*?$",
        r"^\s*The message identifier is:[\s\S]*?$",
        r"^\s*The date of the message is:[\s\S]*?$",
        r"^\s*The subject of the message is:[\s\S]*?$",
        r"^\s*The address to which the message has not yet been delivered is:[\s\S]*?$",


        # --- General Signature/Disclaimer patterns ---
        # Catches "Snap:- Thanks & Regards" and "Thanks & Regards" and everything after
        r"^\s*(?:Snap:-\s*)?Thanks\s*&\s*Regards[\s\S]*?$", 
        # Catches "Best Regards," and everything after
        r"^\s*Best\s*Regards,[\s\S]*?$", 
        # Catches "With Regards" and everything after
        r"^\s*With\s*Regards[\s\S]*?$",
        # Catches "Disclaimer" and everything after (case-insensitive, allowing leading non-alphanumeric chars)
        r"^\s*[\W_]*Disclaimer[\s\S]*?$", 
        # Common email client headers/separators
        r"^\s*On\s+.+?\s+wrote:[\s]*$", # Standard "On [Date] [Sender] wrote:"
        r"^\s*From:\s*.+?\s*Sent:\s*.+?\s*To:\s*.+?\s*Subject:\s*.+?\s*$", # Generic email client header (e.g., from Outlook)
        r"^\s*-----Original Message-----\s*$", # Standard original message separator
        r"^\s*[\-_=]{5,}\s*Original Message[\-_=]{5,}[\s\S]*?$", # More robust original message block
        # General disclaimer/system report patterns
        r"CONFIDENTIALITY NOTICE:[\s\S]*?$",
        r"NOTICE: This e-mail and any files transmitted with it are confidential and legally privileged[\s\S]*?$",
        r"This email and any files transmitted with it are confidential and intended solely[\s\S]*?$",
        r"LEGAL DISCLAIMER: The information in this email is confidential and may be legally privileged[\s\S]*?$",
        r"----- Disclaimer -----.+$", # Generic disclaimer line
        r"Hyper-V Environment Report:[\s\S]*?$",
        r"VCenter Environment Report:[\s\S]*?$",
        r"Cluster Overview \(VcenterCluster\)[\s\S]*?$",
        # Final fallback for lines with many separators, indicating a break before replies/footers
        r"^\s*[\-_=]{5,}\s*$", 
    ]
    
    # Iterate backwards through the lines to find the first match from the bottom.
    # This ensures we cut off at the highest point if multiple patterns could match.
    for i in range(len(lines) - 1, -1, -1):
        sub_text_from_current_line = "\n".join(lines[i:])
        
        for pattern in aggressive_signature_disclaimer_patterns:
            if re.search(pattern, sub_text_from_current_line, flags=re.IGNORECASE | re.DOTALL | re.MULTILINE):
                cut_off_index = i
                break # Found a match, cut here and stop checking patterns for this line
        if cut_off_index != len(lines): # If a cut-off was found for this line, no need to check earlier lines
            break 

    cleaned_text = "\n".join(lines[:cut_off_index])

    # --- Phase 3: PII Redaction (Last Resort) ---
    # This phase is only reached if Talon and the aggressive regexes didn't remove the PII.
    # It will remove specific PII patterns wherever they are found in the *remaining* text.

    # 1. Email Addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    cleaned_text = re.sub(email_pattern, '', cleaned_text, flags=re.IGNORECASE)

    # 2. Phone Numbers (various formats)
    phone_pattern = r'\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{4,9}\b|\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{4}[-.\s]?\d{6}\b'
    cleaned_text = re.sub(phone_pattern, '', cleaned_text)

    # 3. Specific Name patterns (more aggressive now as a last resort for PII)
    # This will remove names if they appear in common signature-like contexts
    name_patterns = [
        r'\b(?:Thanks\s*&\s*Regards|Best\s*Regards|With\s*Regards)\s*[\s\S]*?(?:Mobile|Email|Contact|@|✆|✉)?\s*([A-Za-z\s\.]+)(?:\s*\|[\s\S]*?)?',
        r'\bName:\s*([A-Za-z\s\.]+)(?:\s*TOC engineer)?',
        # Explicit names from examples - use with caution as these are very specific
        r'\b(?:Piyush Kant|Chandan Maiti|Sagar Luthra|Arvind Mishra|Suraj Kumar Padhy|Manish Rai|SANJEEV BHARTI)\b', 
    ]
    for pattern in name_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

    # 4. Address components (if they remain)
    address_component_patterns = [
        r'\bPlot No\.\s*\d+',
        r'\b(?:Udyog Vihar|Okhla Phase)\s*(?:Phase)?\s*[\d\w-]*',
        r'\b(?:New Delhi|Gurugram|Haryana|Jaipur|Sitapura|India)\b(?:,\s*\d{6})?',
        r'\b(?:A-?\d+|UGF|[\d]{1,4})\s*(?:,\s*\d{1,4}\s*Okhla Phase)?\s*(?:,\s*IT-?\d{3})?',
    ]
    for pattern in address_component_patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)


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
    1. Converts to lowercase.
    2. Removes common ticket/docket number patterns.
    3. Normalizes common phrasing variations (e.g., "calls" to "calling", "for"/"of" removal, etc.).
    4. Removes more generic problem-related words/phrases.
    5. Removes extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    cleaned_text = text.strip().lower()

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

    # More aggressive normalization of common phrasing variations and generic words
    cleaned_text = re.sub(r'\bcalls\b', 'calling', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\bfor\b|\bof\b', ' ', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bstatus showing red\b', '', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\brobo\b', '', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bcircle\b', '', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bissue\b', '', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bnot showing\b', 'not_showing', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bnot working\b', 'not_working', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\breset password\b', 'password_reset', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bpassword reset\b', 'password_reset', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\brtmt password\b', 'password_reset', cleaned_text, flags=re.IGNORECASE) 
    cleaned_text = re.sub(r'\bcall landing\b', 'call_landing', cleaned_text, flags=re.IGNORECASE) 

    # --- NEW: Additional generic words/phrases to remove for better problem_reported consolidation ---
    generic_noise_patterns = [
        r'\bproblem with\b', r'\bissue with\b', r'\bunable to\b', r'\berror in\b', r'\bfacing issue with\b',
        r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b', r'\bbe\b', r'\bbeen\b', r'\bbeing\b', # Forms of 'to be'
        r'\bthe\b', r'\ba\b', r'\ban\b', # Articles
        r'\band\b', r'\bor\b', r'\bbut\b', r'\bso\b', r'\bnor\b', # Conjunctions
        r'\bwith\b', r'\bin\b', r'\bon\b', r'\bat\b', r'\bto\b', r'\bfrom\b', r'\bby\b', r'\babout\b', r'\bthrough\b', # Prepositions
        r'\bget\b', r'\bgot\b', r'\bgetting\b', # Common verbs
        r'\bshowing\b', r'\bdisplaying\b', r'\bappearing\b', # Display-related verbs
        r'\bsystem\b', r'\bapplication\b', r'\bserver\b', r'\bdatabase\b', # Generic IT terms
        r'\bplease\b', r'\bkindly\b', r'\bcan you\b', r'\bwe are\b', r'\bwe have\b', # Common request/status phrases
        r'\bfor\b', r'\bof\b', # Re-iterate for robustness if not caught earlier
        r'\bof\b', r'\bthis\b', r'\bthat\b', r'\bthese\b', r'\bthose\b', # Demonstratives
        r'\bhas\b', r'\bhave\b', r'\bhad\b', # Forms of 'to have'
        r'\bdo\b', r'\bdoes\b', r'\bdid\b', # Forms of 'to do'
        r'\bcan\b', r'\bwill\b', r'\bwould\b', r'\bshould\b', r'\bcould\b', # Modals
        r'\bvery\b', r'\bquite\b', r'\bjust\b', r'\balso\b', r'\bnow\b', r'\bthen\b', # Adverbs
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
        
        # Always append the docket number, even if content is empty for this specific column,
        # because the docket itself has been determined to be meaningful by its DecodedBody.
        # Now formatted as just the sequential number (e.g., "1. Content")
        formatted_output.append(f"{seq_num}. {unique_content_for_docket}")
    
    return "\n".join(formatted_output)

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
                     'disposition_name', 'sub_disposition_name', 'DecodedBody']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Adding empty column.", file=sys.stderr)
            df[col] = '' 
    
    # Fill any NaN values in the input DataFrame with empty strings before processing
    df = df.fillna('')

    # --- NEW: Filter out rows where subject starts with "Re:", "Fw:", etc. ---
    # This ensures these dockets are not processed at all for consolidation.
    initial_row_count = len(df)
    
    # Regex to match subjects starting with common reply/forward prefixes (case-insensitive)
    reply_forward_pattern = r"^(Re|Fw|REF|FWD):\s*"
    
    # Filter rows where the subject does NOT start with these patterns
    df = df[~df['subject'].astype(str).str.contains(reply_forward_pattern, case=False, na=False)].copy()
    
    if len(df) < initial_row_count:
        print(f"Filtered out {initial_row_count - len(df)} rows due to 'Re:'/'Fw:' subjects.")
    else:
        print("No rows found with 'Re:'/'Fw:' subjects to filter.")

    consolidated_rows_output = []
    processed_indices = set()

    # Get a list of subjects and their original indices for primary grouping
    # No Re:/Fw: removal for comparison, as per current request (already filtered out)
    subjects_with_indices = [(normalize_text_for_comparison(str(sub)), i) for i, sub in enumerate(df['subject'])]

    # Prepare TF-IDF Vectorizer for cosine similarity (for subjects)
    subject_vectorizer = TfidfVectorizer()
    # Fit and transform all cleaned subjects to get TF-IDF vectors
    subject_tfidf_matrix = subject_vectorizer.fit_transform([s for s, _ in subjects_with_indices])

    consolidated_group_counter = 0 # Initialize counter for consolidated groups

    for i, (current_subject_normalized, current_idx) in enumerate(subjects_with_indices):
        if current_idx in processed_indices:
            continue # Skip if this row has already been processed as part of a group

        # Step 1: Identify group based on similar subject (using fuzzy matching)
        current_group_indices = {current_idx}
        
        for j, (other_subject_normalized, other_idx) in enumerate(subjects_with_indices):
            if other_idx in processed_indices or other_idx == current_idx:
                continue

            # Fuzzy matching (token_sort_ratio)
            score = fuzz.token_sort_ratio(current_subject_normalized, other_subject_normalized)

            if score >= similarity_threshold:
                current_group_indices.add(other_idx)
        
        # Step 2: Refine grouping using Cosine Similarity for remaining unprocessed subjects
        # This helps catch semantic similarities not caught by fuzzy matching
        unprocessed_in_this_pass = [idx for idx in range(len(subjects_with_indices)) if idx not in processed_indices and idx != current_idx]
        
        if unprocessed_in_this_pass:
            current_vector = subject_tfidf_matrix[i:i+1] # Get vector for current subject
            
            for other_idx_raw in unprocessed_in_this_pass:
                other_vector = subject_tfidf_matrix[other_idx_raw:other_idx_raw+1]
                
                # Calculate cosine similarity
                cos_sim = cosine_similarity(current_vector, other_vector)[0][0]
                
                if cos_sim >= cosine_similarity_threshold:
                    current_group_indices.add(other_idx_raw)
        
        # Mark all rows in this subject-similar group as processed
        for idx in current_group_indices:
            processed_indices.add(idx)

        # Process the current subject-based group
        if current_group_indices:
            consolidated_group_counter += 1 # Increment for each new consolidated group
            current_consolidated_group_index = consolidated_group_counter

            group_df = df.loc[list(current_group_indices)].copy()

            consolidated_row_data = {}
            
            # --- Generate Initial Docket Numbering and Map for ALL dockets in this group ---
            # This map is used internally to get sequence numbers for all dockets in the group.
            all_dockets_in_group = []
            for docket_cell in group_df['docket_no'].astype(str):
                # Split and extend to get individual dockets
                all_dockets_in_group.extend([d.strip() for d in docket_cell.split(';') if d.strip()])
            unique_dockets_in_group = set(all_dockets_in_group) 
            
            # Create the 1., 2. numbering for ALL dockets in this raw group
            # Note: create_docket_sequence_map no longer takes group_category_index for its output string
            _, final_docket_to_seq_num_map = create_docket_sequence_map(unique_dockets_in_group)

            # --- Determine the master filter set: dockets that have meaningful DecodedBody content ---
            dockets_with_meaningful_body = set()
            
            for index, row in group_df.iterrows():
                row_dockets_str = str(row['docket_no'])
                row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]
                
                cleaned_body = clean_decoded_body(str(row['DecodedBody']))
                
                if cleaned_body: # Only consider if the cleaned body is not empty
                    for docket in row_dockets:
                        # Ensure it's a docket from this group and has meaningful body
                        if docket in final_docket_to_seq_num_map: 
                            dockets_with_meaningful_body.add(docket)
            
            # If no dockets in this group have a meaningful decoded body, skip this entire consolidated row
            if not dockets_with_meaningful_body:
                consolidated_group_counter -= 1 # Decrement if group is skipped
                continue

            # --- Consolidate Subject: Show only one subject for the group, numbered as category index ---
            # Use the cleaned subject (used for comparison) as the representative subject for the group's display
            consolidated_row_data['subject'] = f"{current_consolidated_group_index}. {current_subject_normalized.capitalize()}" 

            # --- Re-create the docket_no string using the final_docket_to_seq_num_map ---
            # This will now be 1. DocketA, 2. DocketB etc.
            docket_no_output_lines = []
            sorted_dockets_for_output = sorted(list(dockets_with_meaningful_body), key=lambda d: int(final_docket_to_seq_num_map[d]))
            for docket in sorted_dockets_for_output:
                seq_num = final_docket_to_seq_num_map[docket]
                docket_no_output_lines.append(f"{seq_num}. {docket}")
            consolidated_row_data['docket_no'] = "\n".join(docket_no_output_lines)

            # --- Generate the DecodedBody string (now content-centric numbering 1., 2., 3. ) ---
            all_bodies_for_group = []
            for index, row in group_df.iterrows():
                row_dockets_str = str(row['docket_no'])
                row_dockets = [d.strip() for d in row_dockets_str.split(';') if d.strip()]
                
                cleaned_body = clean_decoded_body(str(row['DecodedBody']))
                
                # Only add if the body is meaningful and associated with a meaningful docket
                if cleaned_body and any(docket in dockets_with_meaningful_body for docket in row_dockets):
                    all_bodies_for_group.append(cleaned_body)
            
            consolidated_row_data['DecodedBody'] = format_indexed_content_list(all_bodies_for_group)

            # --- Consolidate Problem Reported (now using full two-stage similarity grouping) ---
            all_problems_for_group_raw = []
            # Collect all problem reports from the current subject-consolidated group
            for index, row in group_df.iterrows():
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
            consolidated_row_data['priority_name'] = format_docket_attribute_grouping(group_df, 'priority_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)
            consolidated_row_data['disposition_name'] = format_docket_attribute_grouping(group_df, 'disposition_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)
            consolidated_row_data['sub_disposition_name'] = format_docket_attribute_grouping(group_df, 'sub_disposition_name', final_docket_to_seq_num_map, dockets_with_meaningful_body)

            # --- Handle remaining columns with simple 1. 2. numbering, filtering by dockets with content ---
            all_original_columns = df.columns.tolist()
            for col in all_original_columns:
                # Skip columns already processed above
                if col not in consolidated_row_data:
                    temp_col_values_map = defaultdict(list)
                    for idx_in_group, row_in_group in group_df.iterrows():
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
file_path = 'ticket_details_2024_01.csv'
output_file_path = '0consolidated_tickets_formatted.csv' 
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
        print(f"\nTotal original rows (before 'Re:'/'Fw:' filter): {initial_rows_before_mail_filter}")
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
            'sub_disposition_name'
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
if os.path.exists(temp_filtered_file_path):
    try:
        os.remove(temp_filtered_file_path)
        print(f"Temporary file '{temp_filtered_file_path}' removed.")
    except Exception as e:
        print(f"Error removing temporary file '{temp_filtered_file_path}': {e}", file=sys.stderr)
