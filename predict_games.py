"""
Prediction script for Bayesian Poisson EPL model.

Uses the trained bambi/statsmodels models to predict match outcomes.
Calculates all necessary features on-the-fly for prediction.
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import json
from datetime import datetime
from tabulate import tabulate

# Configuration
DATA_FILE = "data/matches.csv"
FEATURES_FILE = "data/features.csv"
MODEL_FILE = "models/models.pkl"
ROLLING_WINDOW = 5
CURRENT_SEASON = 2526

def load_models():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file {MODEL_FILE} not found. Run train_model.py first.")
    with open(MODEL_FILE, "rb") as f:
        models = pickle.load(f)
    return models

def load_data():
    """Load both raw matches and features."""
    matches = pd.read_csv(DATA_FILE)
    matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)
    
    if os.path.exists(FEATURES_FILE):
        features = pd.read_csv(FEATURES_FILE)
        features['Date'] = pd.to_datetime(features['Date'])
        return matches, features
    
    return matches, None

def get_team_latest_stats(df, team_name):
    """
    Calculate the latest rolling stats for a team from their most recent games.
    Returns all stats needed for the Bayesian model.
    """
    # Filter for games involving this team
    team_games = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    
    if len(team_games) == 0:
        return None
    
    team_games = team_games.sort_values('Date')
    
    # Calculate per-game stats from each team's perspective
    stats_list = []
    for _, row in team_games.iterrows():
        if row['HomeTeam'] == team_name:
            stats_list.append({
                'GoalsScored': row['FTHG'],
                'GoalsConceded': row['FTAG'],
                'Points': 3 if row['FTR'] == 'H' else (1 if row['FTR'] == 'D' else 0),
                'Shots': row.get('HS', np.nan),
                'ShotsConceded': row.get('AS', np.nan),
                'ShotsOnTarget': row.get('HST', np.nan),
                'ShotsOnTargetConceded': row.get('AST', np.nan)
            })
        else:
            stats_list.append({
                'GoalsScored': row['FTAG'],
                'GoalsConceded': row['FTHG'],
                'Points': 3 if row['FTR'] == 'A' else (1 if row['FTR'] == 'D' else 0),
                'Shots': row.get('AS', np.nan),
                'ShotsConceded': row.get('HS', np.nan),
                'ShotsOnTarget': row.get('AST', np.nan),
                'ShotsOnTargetConceded': row.get('HST', np.nan)
            })
    
    stats_df = pd.DataFrame(stats_list).tail(ROLLING_WINDOW)
    
    return {
        'AvgGoalsScored': stats_df['GoalsScored'].mean(),
        'AvgGoalsConceded': stats_df['GoalsConceded'].mean(),
        'AvgPoints': stats_df['Points'].mean(),
        'AvgShots': stats_df['Shots'].mean() if 'Shots' in stats_df else np.nan,
        'AvgShotsConceded': stats_df['ShotsConceded'].mean() if 'ShotsConceded' in stats_df else np.nan,
        'AvgShotsOnTarget': stats_df['ShotsOnTarget'].mean() if 'ShotsOnTarget' in stats_df else np.nan,
        'AvgShotsOnTargetConceded': stats_df['ShotsOnTargetConceded'].mean() if 'ShotsOnTargetConceded' in stats_df else np.nan
    }

def get_team_position(df, team_name):
    """Calculate current league position for a team."""
    # Filter to current season
    if 'Season' in df.columns:
        season_df = df[df['Season'] == CURRENT_SEASON]
    else:
        season_df = df[df['Date'] >= '2025-08-15']
    
    # Calculate points for all teams
    standings = {}
    for _, row in season_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        if h not in standings: standings[h] = {'pts': 0, 'gd': 0}
        if a not in standings: standings[a] = {'pts': 0, 'gd': 0}
        
        if row['FTR'] == 'H':
            standings[h]['pts'] += 3
        elif row['FTR'] == 'A':
            standings[a]['pts'] += 3
        else:
            standings[h]['pts'] += 1
            standings[a]['pts'] += 1
        
        standings[h]['gd'] += row['FTHG'] - row['FTAG']
        standings[a]['gd'] += row['FTAG'] - row['FTHG']
    
    # Sort by points then goal diff
    sorted_teams = sorted(standings.items(), key=lambda x: (-x[1]['pts'], -x[1]['gd']))
    
    for i, (team, _) in enumerate(sorted_teams, 1):
        if team == team_name:
            return i
    
    return 10  # Default middle position if not found

def predict_match(home_team, away_team, models, df):
    """
    Predict a single match using the trained Bayesian models.
    """
    home_stats = get_team_latest_stats(df, home_team)
    away_stats = get_team_latest_stats(df, away_team)
    
    if not home_stats:
        return {'error': f"No data for {home_team}"}
    if not away_stats:
        return {'error': f"No data for {away_team}"}
    
    home_position = get_team_position(df, home_team)
    away_position = get_team_position(df, away_team)
    
    # Build feature DataFrame matching the training format
    features = pd.DataFrame([{
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Home_Position': home_position,
        'Away_Position': away_position,
        'Home_AvgGoalsScored': home_stats['AvgGoalsScored'],
        'Home_AvgGoalsConceded': home_stats['AvgGoalsConceded'],
        'Away_AvgGoalsScored': away_stats['AvgGoalsScored'],
        'Away_AvgGoalsConceded': away_stats['AvgGoalsConceded'],
        'Home_AvgShots': home_stats.get('AvgShots', 12),
        'Home_AvgShotsOnTarget': home_stats.get('AvgShotsOnTarget', 4),
        'Away_AvgShots': away_stats.get('AvgShots', 10),
        'Away_AvgShotsOnTarget': away_stats.get('AvgShotsOnTarget', 3.5)
    }])
    
    # Replace NaN with reasonable defaults
    features = features.fillna({
        'Home_AvgShots': 12, 'Away_AvgShots': 10,
        'Home_AvgShotsOnTarget': 4, 'Away_AvgShotsOnTarget': 3.5
    })
    
    model_type = models.get('model_type', 'unknown')
    
    if model_type == 'bayesian_poisson':
        # Use bambi posterior predictive
        try:
            model_home = models['model_home']
            model_away = models['model_away']
            results_home = models['results_home']
            results_away = models['results_away']
            
            # Posterior predictive sampling
            preds_home = model_home.predict(results_home, data=features, kind='pps')
            preds_away = model_away.predict(results_away, data=features, kind='pps')
            
            # Mean of posterior predictions
            home_goals_exp = float(preds_home['FTHG'].values.mean())
            away_goals_exp = float(preds_away['FTAG'].values.mean())
        except Exception as e:
            print(f"Bayesian prediction error: {e}")
            # Fallback to simple average
            home_goals_exp = home_stats['AvgGoalsScored']
            away_goals_exp = away_stats['AvgGoalsScored']
    
    elif model_type == 'statsmodels_glm':
        # Use statsmodels predict
        model_home = models['model_home']
        model_away = models['model_away']
        
        home_goals_exp = float(model_home.predict(features).values[0])
        away_goals_exp = float(model_away.predict(features).values[0])
    
    else:
        # Fallback to simple rolling average
        home_goals_exp = home_stats['AvgGoalsScored']
        away_goals_exp = away_stats['AvgGoalsScored']
    
    return {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HomeGoals_Exp': round(home_goals_exp, 2),
        'AwayGoals_Exp': round(away_goals_exp, 2),
        'PredictedScore': f"{int(round(home_goals_exp))} - {int(round(away_goals_exp))}"
    }

def get_next_fixtures():
    """Define upcoming fixtures."""
    return [
        ("Everton", "Leeds"),
        ("Leeds", "Arsenal"),
        ("Wolves", "Bournemouth"),
        ("Brighton", "Everton"),
        ("Chelsea", "West Ham"),
        ("Liverpool", "Newcastle"),
        ("Aston Villa", "Brentford"),
        ("Man United", "Fulham"),
        ("Nott'm Forest", "Crystal Palace"),
        ("Tottenham", "Man City"),
        ("Sunderland", "Burnley")
    ]

def get_past_results():
    """Recent past results for evaluation."""
    return [
        {"Home": "West Ham", "Away": "Sunderland", "Actual": "3 - 1"},
        {"Home": "Fulham", "Away": "Brighton", "Actual": "2 - 1"},
        {"Home": "Burnley", "Away": "Tottenham", "Actual": "2 - 2"},
        {"Home": "Man City", "Away": "Wolves", "Actual": "2 - 0"},
        {"Home": "Bournemouth", "Away": "Liverpool", "Actual": "3 - 2"},
        {"Home": "Crystal Palace", "Away": "Chelsea", "Actual": "1 - 3"},
        {"Home": "Newcastle", "Away": "Aston Villa", "Actual": "0 - 2"},
        {"Home": "Brentford", "Away": "Nott'm Forest", "Actual": "0 - 2"},
        {"Home": "Arsenal", "Away": "Man United", "Actual": "2 - 3"}
    ]

def main():
    parser = argparse.ArgumentParser(description="Predict EPL Match Scores")
    parser.add_argument("--home", type=str, help="Home Team Name")
    parser.add_argument("--away", type=str, help="Away Team Name")
    parser.add_argument("--json", type=str, help="Output JSON file for predictions")
    parser.add_argument("--past-results-json", type=str, help="Output JSON for past results")
    args = parser.parse_args()

    print("Loading models and data...")
    try:
        models = load_models()
        df, features_df = load_data()
        print(f"Model type: {models.get('model_type', 'unknown')}")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    teams = sorted(df['HomeTeam'].unique())
    
    if args.home and args.away:
        # Single match prediction
        result = predict_match(args.home, args.away, models, df)
        if 'error' in result:
            print(result['error'])
        else:
            print(f"\n{result['HomeTeam']} vs {result['AwayTeam']}")
            print(f"Expected Goals: {result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}")
            print(f"PREDICTION: {result['PredictedScore']}")
    
    elif args.json:
        # Batch predictions
        predictions = []
        for h, a in get_next_fixtures():
            if h in teams and a in teams:
                result = predict_match(h, a, models, df)
                if 'error' not in result:
                    predictions.append(result)
        
        with open(args.json, 'w') as f:
            json.dump(predictions, f, indent=4)
        print(f"Predictions saved to {args.json}")
        
        if args.past_results_json:
            past_results = []
            for match in get_past_results():
                h, a = match["Home"], match["Away"]
                if h in teams and a in teams:
                    result = predict_match(h, a, models, df)
                    if 'error' not in result:
                        past_results.append({
                            'HomeTeam': h,
                            'AwayTeam': a,
                            'PredictedScore': result['PredictedScore'],
                            'ActualScore': match["Actual"],
                            'ExpectedGoals': f"{result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}"
                        })
            
            with open(args.past_results_json, 'w') as f:
                json.dump(past_results, f, indent=4)
            print(f"Past results saved to {args.past_results_json}")
    
    else:
        # Interactive mode
        print("\n--- Interactive Prediction ---")
        while True:
            h = input("Home Team (or 'q'): ").strip()
            if h.lower() == 'q': break
            a = input("Away Team: ").strip()
            
            if h not in teams or a not in teams:
                print("Team not found.")
                continue
            
            result = predict_match(h, a, models, df)
            if 'error' in result:
                print(result['error'])
            else:
                print(tabulate([
                    ["Match", f"{result['HomeTeam']} vs {result['AwayTeam']}"],
                    ["Expected Goals", f"{result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}"],
                    ["PREDICTION", result['PredictedScore']]
                ], tablefmt="grid"))

if __name__ == "__main__":
    main()
