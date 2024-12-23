import reddit_classifiers as rc
import pandas as pd
import os

# Load data
data = rc.util.data_pipeline(
    'musk_classified/{}_musk_{}.csv',
    end_date='2024-10',
    start_date=None,
    data_type='both',
    submission_threshold=0.005,
    comment_threshold=0.001,
    word_count_threshold=20,
    user_threshold=0.0001,
    removed_users=['[deleted]', '[removed]', 'automoderator'],
    bot_filter=True,
    verbose=True
)

# Create and save each plot
plots = [
    # Volume plot
    lambda: rc.plotting.plot_volume(data, title='(elon | musk) content volume'),
    
    # GoEmotions plots
    lambda: rc.plotting.plot_goemotions(
        data,
        emotions=['admiration', 'anger', 'annoyance', 'approval'],
        title='(elon | musk) content go_emotions',
        as_percentage=False
    ),
    lambda: rc.plotting.plot_goemotions(
        data,
        emotions=['admiration', 'anger', 'annoyance', 'approval'],
        title='(elon | musk) content go_emotions',
        as_percentage=True
    ),
    lambda: rc.plotting.plot_goemotions(
        data,
        title='(elon | musk) content go_emotions',
        as_percentage=False
    ),
    lambda: rc.plotting.plot_goemotions(
        data,
        title='(elon | musk) content go_emotions',
        as_percentage=True
    ),
    
    # Metrics plot
    lambda: rc.plotting.plot_metrics(
        data,
        metrics=['prosociality', 'toxicity', 'polarization'],
        title='(elon | musk) content metrics',
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
for plot_func in plots:
    fig = plot_func()
    html_content += fig.to_html(full_html=False, include_plotlyjs='cdn')

# Survey dashboard
survey_df = pd.read_csv('insilico-survey/musk_gpt-4o-mini.csv')
survey_fig = rc.plotting.survey_dashboard(survey_df, title='"elon musk" survey results')
html_content += survey_fig.to_html(full_html=False, include_plotlyjs='cdn')

# Save all plots into a single HTML file
with open('combined_plots.html', 'w') as f:
    f.write(f"<html><head><title>Combined Plots</title></head><body>{html_content}</body></html>")

print("All plots have been combined into 'combined_plots.html'")