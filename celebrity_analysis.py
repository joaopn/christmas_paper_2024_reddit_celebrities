import reddit_classifiers as rc
import pandas as pd
import os
import argparse
import sys
import logging

def setup_logging(celebrity):
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging to file without timestamps
    logging.basicConfig(
        filename=f'logs/{celebrity}_parsing.log',
        level=logging.INFO,
        format='%(message)s'  # Only log the message without any additional formatting
    )
    
    # Also print to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Reddit content about a celebrity')
    
    # Required arguments
    parser.add_argument('--celebrity', type=str, help='Celebrity surname (e.g., musk)')
    parser.add_argument('--model', type=str, help='Model name for survey data (e.g., gpt-4o-mini)')
    
    # Optional data parsing arguments
    parser.add_argument('--end_date', type=str, default='2024-10', 
                      help='End date for data collection (YYYY-MM)')
    parser.add_argument('--start_date', type=str, default=None,
                      help='Start date for data collection (YYYY-MM)')
    parser.add_argument('--submission_threshold', type=float, default=0.005,
                      help='Threshold for submission filtering')
    parser.add_argument('--comment_threshold', type=float, default=0.001,
                      help='Threshold for comment filtering')
    parser.add_argument('--word_count_threshold', type=int, default=20,
                      help='Minimum word count threshold')
    parser.add_argument('--user_threshold', type=float, default=0.0001,
                      help='User activity threshold')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup directories
    os.makedirs('results', exist_ok=True)
    
    # Setup logging
    setup_logging(args.celebrity)
    
    # Redirect stdout to capture data_pipeline output
    logging.info(f"Starting analysis for {args.celebrity}")
    
    # Load data
    data = rc.util.data_pipeline(
        f'{args.celebrity}_classified/{{}}_{args.celebrity}_{{}}.csv',
        end_date=args.end_date,
        start_date=args.start_date,
        data_type='both',
        submission_threshold=args.submission_threshold,
        comment_threshold=args.comment_threshold,
        word_count_threshold=args.word_count_threshold,
        user_threshold=args.user_threshold,
        removed_users=['[deleted]', '[removed]', 'automoderator'],
        bot_filter=True,
        verbose=True
    )
    
    # Create and save each plot
    plots = [
        # Volume plot
        lambda: rc.plotting.plot_volume(data, title=f'({args.celebrity}) content volume'),
        
        # GoEmotions plots
        lambda: rc.plotting.plot_goemotions(
            data,
            emotions=['admiration', 'anger', 'annoyance', 'approval'],
            title=f'({args.celebrity}) content go_emotions',
            as_percentage=False
        ),
        lambda: rc.plotting.plot_goemotions(
            data,
            emotions=['admiration', 'anger', 'annoyance', 'approval'],
            title=f'({args.celebrity}) content go_emotions',
            as_percentage=True
        ),
        lambda: rc.plotting.plot_goemotions(
            data,
            title=f'({args.celebrity}) content go_emotions',
            as_percentage=False
        ),
        lambda: rc.plotting.plot_goemotions(
            data,
            title=f'({args.celebrity}) content go_emotions',
            as_percentage=True
        ),
        
        # Metrics plot
        lambda: rc.plotting.plot_metrics(
            data,
            metrics=['prosociality', 'toxicity', 'polarization'],
            title=f'({args.celebrity}) content metrics',
            quantiles=(0.25, 0.75),
            center='median'
        ),
        
        # Distributions plot
        lambda: rc.plotting.plot_distributions(
            data,
            metrics=['prosociality', 'toxicity', 'polarization'],
            periods=[(2005, 2020), (2021, 2022), (2023, 2024)]
        ),
    ]

    # Generate and save plots
    html_content = ""
    for i, plot_func in enumerate(plots):
        logging.info(f"Generating plot {i+1}/{len(plots)}")
        fig = plot_func()
        html_content += fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Survey dashboard
    try:
        survey_df = pd.read_csv(f'insilico-survey/{args.celebrity}_{args.model}.csv')
        survey_fig = rc.plotting.survey_dashboard(
            survey_df, 
            title=f'"{args.celebrity}" survey with {args.model}'
        )
        html_content += survey_fig.to_html(full_html=False, include_plotlyjs='cdn')
        logging.info("Survey dashboard generated successfully")
    except FileNotFoundError:
        logging.warning(f"Survey file not found: insilico-survey/{args.celebrity}_{args.model}.csv")
    except Exception as e:
        logging.error(f"Error generating survey dashboard: {str(e)}")

    # Save all plots into a single HTML file
    output_path = f'results/{args.celebrity}.html'
    with open(output_path, 'w') as f:
        f.write(f"<html><head><title>{args.celebrity.title()} Analysis</title></head><body>{html_content}</body></html>")

    logging.info(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main() 