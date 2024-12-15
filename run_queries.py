import argparse
import sys
from dateutil import rrule
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import psycopg
import os

# Default connection parameters
argdefaults = {
    'hostaddr': '127.0.0.1',
    'port': 5432,
    'dbname': 'datasets',
    'user': 'postgres',
    'table': 'reddit.submissions',
    'search_terms': 'elon|musk'
}

def export_submissions(dataset, output_folder, hostaddr, table, port, dbname, user, search_terms):
    """Exports data for a specific monthly dataset to a CSV file."""
    first_term = search_terms.split('|')[0]
    filename = os.path.join(output_folder, f"submissions_{first_term}_{dataset}.csv")
    print(f"Starting export for dataset: {dataset}")
    try:
        with psycopg.connect(
            dbname=dbname,
            user=user,
            host=hostaddr,
            port=port
        ) as conn:
            with conn.cursor() as cursor:
                query = f"""
                    COPY (
                        SELECT dataset, created_utc, id, author, num_comments, score, subreddit, subreddit_subscribers, domain, title, selftext
                        FROM reddit.submissions
                        WHERE dataset = '{dataset}'
                        AND (
                            title ~* '\\m({search_terms})\\M' OR
                            selftext ~* '\\m({search_terms})\\M'
                        )
                    ) TO '{filename}' 
                    WITH CSV HEADER 
                    QUOTE '"' 
                    ESCAPE '"';
                """

                try:
                    cursor.execute(query)
                except Exception as e:
                    print(f"Error during query execution: {str(e)}")
                    raise e

        print(f"Completed export for {dataset}")
        return f"{dataset} exported successfully"
    except Exception as e:
        print(f"Failed export for {dataset}: {str(e)}")
        return f"Error exporting {dataset}: {str(e)}"

def export_comments(dataset, output_folder, hostaddr, table, port, dbname, user, search_terms):
    """Exports comment data for a specific monthly dataset to a CSV file."""
    first_term = search_terms.split('|')[0]
    filename = os.path.join(output_folder, f"comments_{first_term}_{dataset}.csv")
    print(f"Starting comment export for dataset: {dataset}")
    try:
        with psycopg.connect(
            dbname=dbname,
            user=user,
            host=hostaddr,
            port=port
        ) as conn:
            with conn.cursor() as cursor:
                query = f"""
                    COPY (
                        SELECT dataset, created_utc, id, author, score, author_created_utc, subreddit, body
                        FROM reddit.comments
                        WHERE dataset = '{dataset}'
                        AND body ~* '\\m({search_terms})\\M'
                    ) TO '{filename}' 
                    WITH CSV HEADER 
                    QUOTE '"' 
                    ESCAPE '"';
                """

                try:
                    cursor.execute(query)
                except Exception as e:
                    print(f"Error during query execution: {str(e)}")
                    raise e

        print(f"Completed comment export for {dataset}")
        return f"{dataset} comments exported successfully"
    except Exception as e:
        print(f"Failed comment export for {dataset}: {str(e)}")
        return f"Error exporting comments for {dataset}: {str(e)}"

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Export Reddit submissions or comments by dataset month.')
    parser.add_argument('--table', type=str, help='Table to query.', default=argdefaults['table'])
    parser.add_argument('--hostaddr', type=str, help='Host address.', default=argdefaults['hostaddr'])
    parser.add_argument('--port', type=int, help='Port.', default=argdefaults['port'])
    parser.add_argument('--dbname', type=str, help='Database name.', default=argdefaults['dbname'])
    parser.add_argument('--user', type=str, help='User.', default=argdefaults['user'])
    parser.add_argument('--output_folder', type=str, help='Folder to save results in.', required=True)
    parser.add_argument('--date_min', help='Minimum date (YYYY-MM) to query. Defaults to 2005-06 for submissions and 2005-12 for comments', default=None)
    parser.add_argument('--date_max', required=True, help='Maximum date (YYYY-MM) to query')
    parser.add_argument('--workers', type=int, help='Number of workers for parallel processing.', default=1)
    parser.add_argument('--type', choices=['submissions', 'comments', 'both'], 
                       default='both', help='Type of data to export: submissions, comments, or both')
    parser.add_argument('--search_terms', type=str, help='Search terms separated by |', default=argdefaults['search_terms'])
    args = parser.parse_args()

    # Generate list of monthly datasets based on date range
    datasets = []
    end_date = datetime.strptime(args.date_max, '%Y-%m')

    if args.type in ['submissions', 'both']:
        submissions_start = args.date_min if args.date_min else '2005-06'
        start_date = datetime.strptime(submissions_start, '%Y-%m')
        submission_datasets = [
            f"{dt.year}-{dt.month:02d}" 
            for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date)
        ]
        submission_datasets.sort()

    if args.type in ['comments', 'both']:
        comments_start = args.date_min if args.date_min else '2005-12'
        start_date = datetime.strptime(comments_start, '%Y-%m')
        comment_datasets = [
            f"{dt.year}-{dt.month:02d}" 
            for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date)
        ]
        comment_datasets.sort()

    # Set up multiprocessing
    num_cores = args.workers
    pool = mp.Pool(processes=num_cores)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Create partial functions with fixed arguments for multiprocessing
    process_submissions = partial(
        export_submissions,
        output_folder=args.output_folder,
        hostaddr=args.hostaddr,
        table=args.table,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        search_terms=args.search_terms
    )

    process_comments = partial(
        export_comments,
        output_folder=args.output_folder,
        hostaddr=args.hostaddr,
        table=args.table,
        port=args.port,
        dbname=args.dbname,
        user=args.user,
        search_terms=args.search_terms
    )

    # Run queries based on type selection
    if args.type in ['submissions', 'both']:
        with tqdm(total=len(submission_datasets), desc="Exporting submission datasets") as pbar:
            for _ in pool.imap(process_submissions, submission_datasets):
                pbar.update()

    if args.type in ['comments', 'both']:
        with tqdm(total=len(comment_datasets), desc="Exporting comment datasets") as pbar:
            for _ in pool.imap(process_comments, comment_datasets):
                pbar.update()

    # Clean up
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
