
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from dateutil import rrule

def data_pipeline(file_pattern, end_date=None, data_type='both', submission_threshold=0.005, comment_threshold=0.001, word_count_threshold=20, verbose=True):
    """
    Full data cleaning pipeline that:
    1. Loads data from CSV files
    2. Removes submission-heavy subreddits
    3. Filters by word count
    4. Removes duplicates
    
    Prints statistics at each stage if verbose=True and cleans up intermediate data.
    
    Args:
        file_pattern (str): Pattern for the CSV files, e.g. 'folder/{}_identifier_{}.csv'
        end_date (str): End date in YYYY-MM format
        data_type (str): Type of data to load - 'submissions', 'comments', or 'both'
        submission_threshold (float): Threshold for removing submission-heavy subreddits
        comment_threshold (float): Comment threshold for removing submission-heavy subreddits  
        word_count_threshold (int): Minimum word count to keep entries
        verbose (bool): Whether to print statistics at each stage
        
    Returns:
        dict: Dictionary with cleaned DataFrames
    """
    # Load data
    data = load_data(file_pattern, end_date=end_date, data_type=data_type)
    
    if verbose:
        print('RAW DATA:')
        print_stats(data)
        
    # Remove submission-heavy subreddits
    data_removed = remove_submission_heavy_subreddits(
        data, 
        submission_threshold=submission_threshold,
        comment_threshold=comment_threshold
    )
    del data
    
    if verbose:
        print('\nREMOVED SUBMISSION-HEAVY SUBREDDITS:')
        print_stats(data_removed)
    
    # Filter by word count
    data_clean = filter_threshold(
        data_removed, 
        field='word_count', 
        threshold=word_count_threshold
    )
    del data_removed
    
    if verbose:
        print('\nCLEAN DATA:')
        print_stats(data_clean)
    
    # Remove duplicates
    data_dedup = remove_duplicates(data_clean)
    del data_clean
    
    if verbose:
        print('\nDEDUPED DATA:')
        print_stats(data_dedup)
    
    return data_dedup

def get_goemotions_thresholds():
    """
    Returns a dictionary mapping Go Emotions labels to their optimal classification thresholds. Taken from https://huggingface.co/SamLowe/roberta-base-go_emotions
    Returns:
        dict: Mapping of emotion labels to threshold values
    """
    return {
        'admiration': 0.25,
        'amusement': 0.45,
        'anger': 0.15,
        'annoyance': 0.10,
        'approval': 0.30,
        'caring': 0.40,
        'confusion': 0.55,
        'curiosity': 0.25,
        'desire': 0.25,
        'disappointment': 0.40,
        'disapproval': 0.30,
        'disgust': 0.20,
        'embarrassment': 0.10,
        'excitement': 0.35,
        'fear': 0.40,
        'gratitude': 0.45,
        'grief': 0.05,
        'joy': 0.40,
        'love': 0.25,
        'nervousness': 0.25,
        'optimism': 0.20,
        'pride': 0.10,
        'realization': 0.15,
        'relief': 0.05,
        'remorse': 0.10,
        'sadness': 0.40,
        'surprise': 0.15,
        'neutral': 0.25
    }


def load_data(file_pattern, start_date=None, end_date=None, data_type='both'):
    """
    Load and concatenate classified Reddit data between two dates.
    
    Args:
        file_pattern (str): Pattern for the CSV files, e.g. 'folder/{}_identifier_{}.csv'
                           First {} will be replaced with data_type (submissions/comments)
                           Second {} will be replaced with dates in YYYY-MM format
        start_date (str): Start date in YYYY-MM format. If None, defaults to 2005-06 for submissions
                         and 2005-12 for comments
        end_date (str): End date in YYYY-MM format
        data_type (str): Type of data to load - 'submissions', 'comments', or 'both'
        
    Returns:
        dict: Dictionary with keys 'submissions' and/or 'comments' containing the loaded DataFrames
    """
    # Determine which data types to load
    data_types = ['submissions', 'comments'] if data_type == 'both' else [data_type]
    
    # Set default start dates if None provided
    if start_date is None:
        start_dates = {
            'submissions': '2005-06',
            'comments': '2005-12'
        }
    else:
        start_dates = {dtype: start_date for dtype in data_types}
    
    # Load data
    results = {}
    for dtype in data_types:
        # Convert dates to datetime objects
        start = datetime.strptime(start_dates[dtype], '%Y-%m')
        end = datetime.strptime(end_date, '%Y-%m')
        
        # Generate list of dates between start and end
        dates = [dt.strftime('%Y-%m') for dt in 
                rrule.rrule(rrule.MONTHLY, dtstart=start, until=end)]
        
        data = []
        for date in tqdm(dates, desc=f'Loading {dtype}'):
            try:
                df_date = pd.read_csv(file_pattern.format(dtype, date))
                # Only append if dataframe is not empty
                if not df_date.empty:
                    data.append(df_date)
            except FileNotFoundError:
                print(f"Warning: File not found for {dtype} {date}")
                continue
                
        if data:
            # Get common columns across all dataframes
            common_cols = set.intersection(*[set(df.columns) for df in data])
            # Filter to only common columns before concatenating
            data = [df[list(common_cols)] for df in data]
            results[dtype] = pd.concat(data, ignore_index=True)
        else:
            results[dtype] = pd.DataFrame()
            
    return results



def print_stats(data_dict):
    """
    Print basic statistics for each DataFrame in the provided dictionary.
    
    Args:
        data_dict (dict): Dictionary containing DataFrames with Reddit data
    """
    for key, df in data_dict.items():
        if df.empty:
            print(f"\n{key} DataFrame is empty")
            continue
            
        print(f"\nStats for {key}:")
        print("-" * 50)
        
        # Basic counts
        print(f"Total entries: {len(df):,}")
        print(f"Total score sum: {df['score'].sum():,}")
        
        # Subreddit stats
        n_subreddits = df['subreddit'].nunique()
        print(f"Number of unique subreddits: {n_subreddits:,}")
        
        # Top subreddits
        print("\nTop 5 subreddits:")
        subreddit_counts = df['subreddit'].value_counts()
        for subreddit, count in subreddit_counts.head().items():
            percentage = (count / len(df)) * 100
            print(f"  {subreddit}: {count:,} entries ({percentage:.1f}%)")

def filter_threshold(data_dict, field='word_count', threshold=20):
    """
    Filter out entries in each DataFrame that have values below the threshold for the specified field.
    Prints volume counts before and after filtering.
    
    Args:
        data_dict (dict): Dictionary containing DataFrames with Reddit data
        field (str): Field to filter on. Defaults to 'word_count'
        threshold (int): Minimum value to keep. Defaults to 20
        
    Returns:
        dict: Dictionary with filtered DataFrames
    """
    filtered_dict = {}
    
    for key, df in data_dict.items():
        if df.empty:
            filtered_dict[key] = df
            continue
            
        filtered_dict[key] = df[df[field] >= threshold]
        print(f"\nFiltering {key} on {field} >= {threshold}: {len(df):,} -> {len(filtered_dict[key]):,} entries ({len(df) - len(filtered_dict[key]):,} removed)")
        
    return filtered_dict
def remove_duplicates(data_dict, text_col='text'):
    """
    Remove entries with duplicated text content from a DataFrame.
    Prints volume counts before and after filtering.
    
    Args:
        data_dict (dict): Dictionary containing DataFrame with Reddit data
        text_col (str): Column containing text to check for duplicates. Defaults to 'text'
        
    Returns:
        dict: Dictionary with deduplicated DataFrame
    """
    deduped_dict = {}
    
    for key, df in data_dict.items():
        if df.empty:
            deduped_dict[key] = df
            continue
            
        # Remove entries with duplicated text
        deduped_dict[key] = df.drop_duplicates(subset=[text_col])
            
        print(f"\nRemoving duplicates from {key}: {len(df):,} -> {len(deduped_dict[key]):,} entries ({len(df) - len(deduped_dict[key]):,} removed)")
        
    return deduped_dict

def remove_subreddits(data_dict, subreddits):
    """
    Remove entries from each DataFrame that belong to the specified subreddits.
    """
    removed_dict = {}
    for key, df in data_dict.items():
        if df.empty:
            removed_dict[key] = df
            continue

        removed_dict[key] = df[~df['subreddit'].isin(subreddits)]
        print(f"\nRemoving subreddits from {key}: {len(df):,} -> {len(removed_dict[key]):,} entries ({len(df) - len(removed_dict[key]):,} removed)")
        
    return removed_dict


def remove_submission_heavy_subreddits(data_dict, submission_threshold=0.05, comment_threshold=0.02):
    """
    Remove subreddits that have a disproportionately high number of submissions
    compared to their comment volume. Analyzes submissions but removes from both submissions and comments.
    
    Args:
        data_dict (dict): Dictionary containing 'submissions' and 'comments' DataFrames
        submission_threshold (float): Minimum percentage of total submissions (0-1)
        comment_threshold (float): Maximum percentage of total comments (0-1)
        
    Returns:
        dict: Dictionary with filtered DataFrames
    """
    filtered_dict = {}
    
    if 'submissions' not in data_dict or data_dict['submissions'].empty:
        return data_dict
        
    # Get submission counts by subreddit
    sub_counts = data_dict['submissions']['subreddit'].value_counts()
    total_subs = len(data_dict['submissions'])
    
    # Calculate submission percentages
    sub_pcts = sub_counts / total_subs
    
    # Get total comments by subreddit
    comment_counts = data_dict['submissions'].groupby('subreddit')['num_comments'].sum()
    total_comments = comment_counts.sum()
    
    # Calculate comment percentages 
    comment_pcts = comment_counts / total_comments
    
    # Create mask for subreddits meeting criteria
    subreddits_to_remove = [
        subreddit for subreddit in sub_pcts.index
        if (sub_pcts[subreddit] >= submission_threshold and 
            comment_pcts[subreddit] <= comment_threshold)
    ]
    
    # Print info about removed subreddits
    for subreddit in subreddits_to_remove:
        print(f"Removing {subreddit}: {sub_pcts[subreddit]:.2%} submissions, {comment_pcts[subreddit]:.2%} comments")
    print('\n')
    
    # Filter both submissions and comments
    for key, df in data_dict.items():
        if df.empty:
            filtered_dict[key] = df
            continue
        
        filtered_dict[key] = df[~df['subreddit'].isin(subreddits_to_remove)]
        print(f"Removing submission-heavy subreddits from {key}: {len(df):,} -> {len(filtered_dict[key]):,} entries ({len(df) - len(filtered_dict[key]):,} removed)")
    
    return filtered_dict