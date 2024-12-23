import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

def simulate_survey(celebrity, n_samples, years, model, api_key):
    client = OpenAI(api_key=api_key)

    # Define survey questions
    questions = [
        f"How would you describe your overall opinion of {celebrity}? \n1) Very negative 2) Negative 3) Neutral 4) Positive 5) Very positive \nPlease answer only with 1, 2, 3, 4, or 5. This is very important.",
        f"How relatable do you find {celebrity}? \n1) Not at all 2) Slightly 3) Moderately 4) Quite a bit 5) Very much \nPlease answer only with 1, 2, 3, 4, or 5. This is very important.",
        f"How inspiring do you find {celebrity}? \n1) Not at all 2) Slightly 3) Moderately 4) Quite a bit 5) Very much \nPlease answer only with 1, 2, 3, 4, or 5. This is very important.",
        f"How likable do you find {celebrity}? \n1) Not at all 2) Slightly 3) Moderately 4) Quite a bit 5) Very much \nPlease answer only with 1, 2, 3, 4, or 5. This is very important.",
        f"How relevant do you find {celebrity}? \n1) Not at all 2) Slightly 3) Moderately 4) Quite a bit 5) Very much \nPlease answer only with 1, 2, 3, 4, or 5. This is very important."
    ]

    instructions = "You are taking part in a survey. Don't excuse yourself with being an AI."

    # Calculate total iterations for progress bar
    total_queries = len(questions) * n_samples * len(years) * 8  # 8 demographic groups
    
    # Simulate survey
    answers = []
    pbar = tqdm(total=total_queries, desc="Total Progress")
    
    for demo_template in tqdm([
        "You are a male person between 20 and 30 years old in January {}.",
        "You are a female person between 20 and 30 years old in January {}.",
        "You are a male person between 50 and 60 years old in January {}.",
        "You are a female person between 50 and 60 years old in January {}.",
        "You are a queer person in January {}.",
        "You are a straight person in January {}.",
        "You are a white person in January {}.",
        "You are a non-white person in January {}."
    ], desc="Demographics"):
        answers_temp = []
        for year in years:
            demo = demo_template.format(year)
            for _ in range(n_samples):
                n_temp = []
                for question in questions:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "user", "content": demo + " " + instructions + " " + question}
                        ]
                    )
                    n_temp.append(completion.choices[0].message.content)
                    pbar.update(1)
                answers_temp.append(n_temp + [year])
        answers.append(answers_temp)
    
    pbar.close()
    return answers

def process_columns(answers, demo_name, index):
    x = pd.DataFrame(answers[index], columns=['overall', 'relatable', 'inspiring', 'likable', 'relevant', 'year'])
    x['demo'] = demo_name
    return x

def main():
    parser = argparse.ArgumentParser(description='In-silico Celebrity Survey')
    parser.add_argument('--celebrity', type=str, required=True, help='Name of the celebrity')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples per demographic and time point')
    parser.add_argument('--years', type=int, nargs='+', default=[2018, 2021, 2024], help='Years to simulate the survey for')
    parser.add_argument('--output', type=str, default='insilico-survey/', help='Output directory for results')
    parser.add_argument('--model', type=str, default='gpt-4', help='OpenAI model to use')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key')

    args = parser.parse_args()

    # Simulate survey
    answers = simulate_survey(args.celebrity, args.samples, args.years, args.model, args.api_key)

    # Process answers into dataframes
    demo_names = ['male-young', 'female-young', 'male-old', 'female-old', 'queer', 'non-queer', 'white', 'non-white']
    dataframes = [process_columns(answers, demo_name, i) for i, demo_name in enumerate(demo_names)]

    # Concatenate separate dataframes to one large dataframe
    survey_results = pd.concat(dataframes)
    survey_results = survey_results.reset_index().drop(columns=['index'])

    # Data cleaning
    for c in ['overall', 'relatable', 'inspiring', 'likable', 'relevant']:
        survey_results[c] = [''.join(filter(str.isdigit, x)) if ''.join(filter(str.isdigit, x)) != '' else '99' for x in survey_results[c]]
        survey_results[c] = [x if 1 <= int(x) <= 5 else np.nan for x in survey_results[c]]

    # Store results to file
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Include model name in the output filename
    last_word = args.celebrity.split()[-1].lower()  # Get the last word of the celebrity's name
    output_filename = f"{last_word}_{args.model}.csv"
    survey_results.to_csv(os.path.join(args.output, output_filename), index=False)

if __name__ == '__main__':
    main()