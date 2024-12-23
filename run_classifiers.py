import argparse
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from optimum.onnxruntime import ORTModelForSequenceClassification
from datetime import datetime
from dateutil import rrule
import re

class RedditClassifier:
    def __init__(self, gpu_id=0, prosocial_dict_path=None, batch_size=8):
        self.device = torch.device(f'cuda:{gpu_id}')
        self.batch_size = batch_size
        
        # Initialize prosociality model and dictionary
        print("Loading prosociality/polarization model...")
        self.prosociality_model = SentenceTransformer('joaopn/glove-model-reduced-stopwords', device=f"cuda:{gpu_id}")
        self.load_prosocial_dict(prosocial_dict_path)
        self.load_polarization_dict()
        
        # Initialize toxicity model
        print("Loading toxicity model...")
        self.toxicity_model = ORTModelForSequenceClassification.from_pretrained(
            "joaopn/unbiased-toxic-roberta-onnx-fp16",
            file_name='model.onnx',
            provider="CUDAExecutionProvider",
            provider_options={'device_id': gpu_id}
        )
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained("joaopn/unbiased-toxic-roberta-onnx-fp16")
        self.toxicity_index = next(
            (idx for idx, label in self.toxicity_model.config.id2label.items() if label.lower() == 'toxicity'),
            None
        )
        if self.toxicity_index is None:
            raise ValueError("Toxicity label not found in model's id2label mapping.")
        
        # Initialize sentiment model (now using ONNX FP16)
        print("Loading sentiment model...")
        self.sentiment_model = ORTModelForSequenceClassification.from_pretrained(
            "joaopn/roberta-base-go_emotions-onnx-fp16",
            file_name='model.onnx',
            provider="CUDAExecutionProvider",
            provider_options={'device_id': gpu_id}
        )
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("joaopn/roberta-base-go_emotions-onnx-fp16")

    def load_prosocial_dict(self, dict_path=None):
        """Load and process prosocial dictionary."""
        dict_path = dict_path or os.path.join(os.path.dirname(__file__), 'data', 'prosocial_dictionary.csv')
        try:
            prosocial_dict = pd.read_csv(dict_path, header=None, names=['word'])
            prosocial_dict["word"] = prosocial_dict["word"].str.replace("*", "").str.replace("nt", "not")
            prosocial_terms = list(prosocial_dict["word"].values)
            
            dict_embeddings = self.prosociality_model.encode(
                prosocial_terms,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True
            )
            self.dict_embeddings_prosocial = torch.mean(dict_embeddings, dim=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find prosocial dictionary at {dict_path}")

    def load_polarization_dict(self, label_filter='issue'):
        """Load and process polarization dictionary."""
        dict_path = os.path.join(os.path.dirname(__file__), 'data', 'polarization_dictionary.csv')
        try:
            df = pd.read_csv(dict_path, header=0)
            if label_filter is not None:
                df = df[df['label'] == label_filter]
            unique_words = df['word'].unique()
            
            dict_embeddings = self.prosociality_model.encode(
                list(unique_words),
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True
            )
            self.dict_embeddings_polarization = torch.mean(dict_embeddings, dim=0)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find polarization dictionary at {dict_path}")

    def get_prosociality_scores(self, texts):
        """Calculate prosociality scores for a batch of texts."""
        text_embeddings = self.prosociality_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings_prosocial)
        return cos_sim.cpu().numpy().flatten()

    def get_toxicity_scores(self, texts):
        """Calculate toxicity scores for a batch of texts."""
        inputs = self.toxicity_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.toxicity_model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)
        return probabilities[:, self.toxicity_index].cpu().numpy()

    def get_sentiment_scores(self, texts):
        """Calculate sentiment scores for a batch of texts."""
        results = []
        
        for start_idx in range(0, len(texts), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_texts = texts[start_idx:end_idx]

            inputs = self.sentiment_tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.sentiment_model(
                    input_ids=inputs['input_ids'].to(self.device),
                    attention_mask=inputs['attention_mask'].to(self.device)
                )
                predictions = torch.nn.functional.sigmoid(outputs.logits)
                results.append(predictions.cpu())

        all_predictions = torch.cat(results, dim=0).numpy()
        return all_predictions

    def get_polarization_scores(self, texts):
        """Calculate polarization scores for a batch of texts."""
        text_embeddings = self.prosociality_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True
        )
        cos_sim = util.cos_sim(text_embeddings, self.dict_embeddings_polarization)
        return cos_sim.cpu().numpy().flatten()

    def clean_text(self, text):
        """Clean text by removing raw characters and processing markdown links."""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Return empty string for deleted/removed content
        if text.strip() in ['[deleted]', '[removed]']:
            return ""
        
        # Remove raw characters
        text = text.replace('\\n', ' ')  # Fix escaped newlines
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # Replace markdown links with (Link)
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1 (Link)', text)
        
        # Remove multiple spaces
        text = ' '.join(text.split())
        
        return text

    def count_words(self, text):
        """Count words in text."""
        if pd.isna(text):
            return 0
        return len(str(text).split())

    def process_file(self, filepath, output_folder, data_type):
        """Process a single file with all classifiers in batched mode."""
        try:
            # Load data
            df = pd.read_csv(filepath)
            
            # Clean and replace original text fields
            if data_type == 'submissions':
                df['title'] = df['title'].apply(self.clean_text)
                df['selftext'] = df['selftext'].apply(self.clean_text)
                df['text'] = df['title'] + ' ' + df['selftext']
                # Drop original text columns
                df = df.drop(['title', 'selftext'], axis=1)
            else:  # comments
                df['body'] = df['body'].apply(self.clean_text)
                df['text'] = df['body']
                # Drop original text column
                df = df.drop(['body'], axis=1)

            # Add word count
            df['word_count'] = df['text'].apply(self.count_words)

            # Process in batches
            prosocial_results = []
            toxicity_results = []
            sentiment_results = []
            polarization_results = []
            
            with tqdm(total=len(df), desc=os.path.basename(filepath)) as pbar:
                for start_idx in range(0, len(df), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(df))
                    batch_texts = df['text'].iloc[start_idx:end_idx].tolist()

                    prosocial_results.extend(self.get_prosociality_scores(batch_texts))
                    toxicity_results.extend(self.get_toxicity_scores(batch_texts))
                    sentiment_results.extend(self.get_sentiment_scores(batch_texts))
                    polarization_results.extend(self.get_polarization_scores(batch_texts))
                    pbar.update(end_idx - start_idx)

            # Combine results
            results_df = pd.DataFrame({
                'prosociality': prosocial_results,
                'toxicity': toxicity_results,
                'polarization': polarization_results
            })
            sentiment_df = pd.DataFrame(
                sentiment_results,
                columns=[self.sentiment_model.config.id2label[i] for i in range(len(self.sentiment_model.config.id2label))]
            )
            
            # Combine all results
            final_df = pd.concat([df, results_df, sentiment_df], axis=1)

            # Create new file, overwriting if it exists
            final_df.to_csv(os.path.join(output_folder, os.path.basename(filepath)), index=False, mode='w')
            
            return f"{filepath}"
        
        except Exception as e:
            return f"Error processing {filepath}: {str(e)}"

def process_date_range(args, data_type, batch_size):
    """Process files for a specific date range and data type."""
    # Set start date based on data type
    if data_type == 'submissions':
        start_date_str = args.date_min if args.date_min else '2005-06'
    else:  # comments
        start_date_str = args.date_min if args.date_min else '2005-12'
    
    start_date = datetime.strptime(start_date_str, '%Y-%m')
    end_date = datetime.strptime(args.date_max, '%Y-%m')
    
    # Generate list of dates
    dates = [
        f"{dt.year}-{dt.month:02d}" 
        for dt in rrule.rrule(rrule.MONTHLY, dtstart=start_date, until=end_date)
    ]
    
    # Generate file paths
    files = [
        os.path.join(args.input_folder, f"{data_type}_{args.celebrity}_{date}.csv")
        for date in dates
    ]
    
    # Initialize classifier
    classifier = RedditClassifier(
        gpu_id=args.gpu,
        batch_size=batch_size
    )
    
    # Process files without progress bar
    for filepath in files:
        if os.path.exists(filepath):
            classifier.process_file(filepath, args.output_folder, data_type)
        else:
            print(f"File not found: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Run classifiers on Reddit data.')
    parser.add_argument('--input_folder', type=str, required=True, 
                       help='Folder containing input CSV files')
    parser.add_argument('--output_folder', type=str, required=True, 
                       help='Folder to save results in')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for processing')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--type', choices=['submissions', 'comments', 'both'], 
                       default='both', help='Type of data to process')
    parser.add_argument('--date_min', help='Minimum date (YYYY-MM) to process. Defaults to 2005-06 for submissions and 2005-12 for comments', 
                       default=None)
    parser.add_argument('--date_max', required=True, 
                       help='Maximum date (YYYY-MM) to process')
    parser.add_argument('--celebrity', type=str, required=True,
                       help='Celebrity name to process (e.g., "musk")')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    # Process based on type
    if args.type in ['submissions', 'both']:
        process_date_range(args, 'submissions', args.batch_size)
    
    if args.type in ['comments', 'both']:
        process_date_range(args, 'comments', args.batch_size)

if __name__ == "__main__":
    main()
