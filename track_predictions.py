"""
Prediction History Tracker

Logs predictions and tracks accuracy over time.
- Logs new predictions with predicted result (H/D/A)
- Updates past predictions with actual results when games are played
- Calculates accuracy metrics
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

HISTORY_FILE = "data/prediction_history.csv"
MATCHES_FILE = "data/matches.csv"
PREDICTIONS_FILE = "predictions.json"
ACCURACY_FILE = "accuracy.json"

def load_history():
    """Load prediction history or create empty DataFrame."""
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        # Use format='mixed' to handle various date formats
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df['PredictionDate'] = pd.to_datetime(df['PredictionDate'], format='mixed', errors='coerce')
        return df
    else:
        return pd.DataFrame(columns=[
            'PredictionDate', 'Date', 'HomeTeam', 'AwayTeam', 
            'PredResult', 'ActualResult', 'Correct'
        ])

def get_predicted_result(home_exp, away_exp):
    """Convert expected goals to H/D/A result."""
    home_goals = round(home_exp)
    away_goals = round(away_exp)
    
    if home_goals > away_goals:
        return 'H'
    elif away_goals > home_goals:
        return 'A'
    else:
        return 'D'

def log_new_predictions(history_df, predictions_file=PREDICTIONS_FILE):
    """Add new predictions to history."""
    if not os.path.exists(predictions_file):
        print(f"No predictions file found: {predictions_file}")
        return history_df
    
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    # Handle both old (list) and new (dict with predictions key) formats
    if isinstance(data, dict) and 'predictions' in data:
        predictions = data['predictions']
    else:
        predictions = data
    
    today = datetime.now().date()
    new_rows = []
    
    for pred in predictions:
        home = pred['HomeTeam']
        away = pred['AwayTeam']
        
        # Check if this exact match (home/away combo) already exists in history
        # regardless of prediction date - prevents duplicates across runs
        exists = False
        if len(history_df) > 0:
            # Check if this match already exists and hasn't been evaluated yet
            exists = (
                (history_df['HomeTeam'] == home) & 
                (history_df['AwayTeam'] == away) &
                (history_df['ActualResult'].isna() | (history_df['ActualResult'] == ''))
            ).any()
        
        if not exists:
            pred_result = get_predicted_result(pred['HomeGoals_Exp'], pred['AwayGoals_Exp'])
            new_rows.append({
                'PredictionDate': today,
                'Date': pd.NaT,  # Will be filled when game is played
                'HomeTeam': home,
                'AwayTeam': away,
                'PredResult': pred_result,
                'ActualResult': '',
                'Correct': np.nan
            })
    
    if new_rows:
        history_df = pd.concat([history_df, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"Added {len(new_rows)} new predictions to history")
    else:
        print("No new predictions to add (all already in history)")
    
    return history_df

def update_with_actual_results(history_df, matches_file=MATCHES_FILE):
    """Update predictions with actual results from played games."""
    if not os.path.exists(matches_file):
        return history_df
    
    matches = pd.read_csv(matches_file)
    matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)
    
    updated = 0
    for idx, row in history_df.iterrows():
        if row['ActualResult'] == '' or pd.isna(row['ActualResult']):
            # Find matching game
            match = matches[
                (matches['HomeTeam'] == row['HomeTeam']) &
                (matches['AwayTeam'] == row['AwayTeam']) &
                (matches['Date'] >= row['PredictionDate'])
            ]
            
            if len(match) > 0:
                actual_match = match.iloc[0]
                actual_result = actual_match['FTR']
                history_df.at[idx, 'Date'] = actual_match['Date']
                history_df.at[idx, 'ActualResult'] = actual_result
                history_df.at[idx, 'Correct'] = 1 if row['PredResult'] == actual_result else 0
                updated += 1
    
    if updated > 0:
        print(f"Updated {updated} predictions with actual results")
    
    return history_df

def calculate_accuracy(history_df):
    """Calculate accuracy metrics."""
    # Only count predictions with actual results
    evaluated = history_df[history_df['ActualResult'] != ''].copy()
    evaluated = evaluated.dropna(subset=['Correct'])
    
    if len(evaluated) == 0:
        return {
            'total_predictions': len(history_df),
            'evaluated': 0,
            'correct': 0,
            'accuracy': 0.0,
            'pending': len(history_df)
        }
    
    correct = int(evaluated['Correct'].sum())
    total = len(evaluated)
    pending = len(history_df) - total
    
    return {
        'total_predictions': len(history_df),
        'evaluated': total,
        'correct': correct,
        'accuracy': round((correct / total) * 100, 1) if total > 0 else 0.0,
        'pending': pending
    }

def save_accuracy(accuracy_stats):
    """Save accuracy stats to JSON for website."""
    with open(ACCURACY_FILE, 'w') as f:
        json.dump(accuracy_stats, f, indent=2)
    print(f"Accuracy saved: {accuracy_stats['correct']}/{accuracy_stats['evaluated']} = {accuracy_stats['accuracy']}%")

def main():
    print("=" * 50)
    print("PREDICTION HISTORY TRACKER")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    
    # Load existing history
    history = load_history()
    print(f"Loaded {len(history)} predictions from history")
    
    # Update with actual results first
    history = update_with_actual_results(history)
    
    # Log new predictions
    history = log_new_predictions(history)
    
    # Calculate accuracy
    accuracy = calculate_accuracy(history)
    
    # Save files
    history.to_csv(HISTORY_FILE, index=False)
    print(f"History saved to {HISTORY_FILE}")
    
    save_accuracy(accuracy)
    
    print("\n" + "=" * 50)
    print(f"ACCURACY: {accuracy['correct']}/{accuracy['evaluated']} = {accuracy['accuracy']}%")
    print(f"Pending games: {accuracy['pending']}")
    print("=" * 50)

if __name__ == "__main__":
    main()
