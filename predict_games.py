"""
Prediction script for Bayesian Poisson EPL model.

Uses the trained bambi/statsmodels models to predict match outcomes.
Calculates all necessary features on-the-fly for prediction.
Includes xG, xGA, and Elo features when available.
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse
import json
from datetime import datetime
from tabulate import tabulate
from scipy.stats import poisson

# Try to import advanced data loader for Elo ratings
try:
    from advanced_data_loader import fetch_elo_ratings, fetch_understat_xg
    HAS_ADVANCED_DATA = True
except ImportError:
    HAS_ADVANCED_DATA = False

# Import fixture scraper for dynamic fixture fetching
try:
    from fixture_scraper import (
        get_current_matchday, 
        scrape_fixtures, 
        fetch_matches,
        normalize_team_name
    )
    HAS_FIXTURE_SCRAPER = True
except ImportError:
    HAS_FIXTURE_SCRAPER = False
    print("Warning: fixture_scraper not available, using fallback")

# Configuration
DATA_FILE = "data/matches.csv"
FEATURES_FILE = "data/features.csv"
MODEL_FILE = "models/models.pkl"
ROLLING_WINDOW = 5
CURRENT_SEASON = 2526

# Cache for Elo ratings
_elo_cache = None

def calculate_poisson_probs(home_exp, away_exp, max_goals=5):
    """
    Calculate probability matrix for all scorelines from 0-0 to max_goals-max_goals.
    Returns a dict with probabilities for display in heatmap.
    """
    probs = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
            probs[f"{h}-{a}"] = round(prob * 100, 1)  # Store as percentage
    return probs

def calculate_outcome_probs(home_exp, away_exp, max_goals=8):
    """
    Calculate probabilities of Home Win, Draw, Away Win from Poisson model.
    Uses a higher max_goals for more accurate estimation.
    """
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp)
            if h > a:
                home_win += prob
            elif h == a:
                draw += prob
            else:
                away_win += prob
    
    return {
        'H': round(home_win * 100, 1),
        'D': round(draw * 100, 1),
        'A': round(away_win * 100, 1)
    }

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

def get_team_latest_stats(df, team_name, features_df=None):
    """
    Calculate the latest rolling stats for a team from their most recent games.
    Returns all stats needed for the Bayesian model.
    If features_df is provided, also extracts xG/xGA stats from it.
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
    
    result = {
        'AvgGoalsScored': stats_df['GoalsScored'].mean(),
        'AvgGoalsConceded': stats_df['GoalsConceded'].mean(),
        'AvgPoints': stats_df['Points'].mean(),
        'AvgShots': stats_df['Shots'].mean() if 'Shots' in stats_df else np.nan,
        'AvgShotsConceded': stats_df['ShotsConceded'].mean() if 'ShotsConceded' in stats_df else np.nan,
        'AvgShotsOnTarget': stats_df['ShotsOnTarget'].mean() if 'ShotsOnTarget' in stats_df else np.nan,
        'AvgShotsOnTargetConceded': stats_df['ShotsOnTargetConceded'].mean() if 'ShotsOnTargetConceded' in stats_df else np.nan
    }
    
    # Try to get xG stats from features dataframe
    if features_df is not None:
        # Get latest xG averages from features file
        team_features = features_df[
            (features_df['HomeTeam'] == team_name) | (features_df['AwayTeam'] == team_name)
        ].sort_values('Date')
        
        if len(team_features) > 0:
            latest = team_features.iloc[-1]
            if team_features.iloc[-1]['HomeTeam'] == team_name:
                result['AvgxG'] = latest.get('Home_AvgxG', np.nan)
                result['AvgxGA'] = latest.get('Home_AvgxGA', np.nan)
            else:
                result['AvgxG'] = latest.get('Away_AvgxG', np.nan)
                result['AvgxGA'] = latest.get('Away_AvgxGA', np.nan)
    
    return result


def get_team_elo(team_name):
    """
    Get current Elo rating for a team.
    Uses cached data if available.
    """
    global _elo_cache
    
    if not HAS_ADVANCED_DATA:
        return None
    
    # Load Elo ratings if not cached
    if _elo_cache is None:
        _elo_cache = fetch_elo_ratings(use_cache=True)
    
    if _elo_cache is None:
        return None
    
    # Find team in cache
    team_row = _elo_cache[_elo_cache['Team'] == team_name]
    if len(team_row) > 0:
        return float(team_row.iloc[0]['Elo'])
    
    # Try alternative names
    return None

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

def predict_match(home_team, away_team, models, df, features_df=None):
    """
    Predict a single match using the trained Bayesian models.
    Includes xG/xGA and Elo features when available.
    """
    home_stats = get_team_latest_stats(df, home_team, features_df)
    away_stats = get_team_latest_stats(df, away_team, features_df)
    
    if not home_stats:
        return {'error': f"No data for {home_team}"}
    if not away_stats:
        return {'error': f"No data for {away_team}"}
    
    home_position = get_team_position(df, home_team)
    away_position = get_team_position(df, away_team)
    
    # Get Elo ratings
    home_elo = get_team_elo(home_team)
    away_elo = get_team_elo(away_team)
    
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
        'Away_AvgShotsOnTarget': away_stats.get('AvgShotsOnTarget', 3.5),
        # Estimate odds probabilities based on position (rough approximation)
        'Odds_HomeProb': max(0.25, 0.55 - (home_position - away_position) * 0.02),
        'Odds_AwayProb': max(0.15, 0.25 + (home_position - away_position) * 0.02),
        # xG features
        'Home_AvgxG': home_stats.get('AvgxG', home_stats['AvgGoalsScored']),
        'Home_AvgxGA': home_stats.get('AvgxGA', home_stats['AvgGoalsConceded']),
        'Away_AvgxG': away_stats.get('AvgxG', away_stats['AvgGoalsScored']),
        'Away_AvgxGA': away_stats.get('AvgxGA', away_stats['AvgGoalsConceded']),
        'Home_xGDiff': home_stats.get('AvgxG', home_stats['AvgGoalsScored']) - home_stats.get('AvgxGA', home_stats['AvgGoalsConceded']),
        'Away_xGDiff': away_stats.get('AvgxG', away_stats['AvgGoalsScored']) - away_stats.get('AvgxGA', away_stats['AvgGoalsConceded']),
        # Elo features
        'Home_Elo': home_elo if home_elo else 1500,
        'Away_Elo': away_elo if away_elo else 1500,
        'Elo_Diff': (home_elo - away_elo) if (home_elo and away_elo) else 0
    }])
    
    # Replace NaN with reasonable defaults
    features = features.fillna({
        'Home_AvgShots': 12, 'Away_AvgShots': 10,
        'Home_AvgShotsOnTarget': 4, 'Away_AvgShotsOnTarget': 3.5,
        'Odds_HomeProb': 0.45, 'Odds_AwayProb': 0.28,
        'Home_AvgxG': 1.3, 'Away_AvgxG': 1.1,
        'Home_AvgxGA': 1.1, 'Away_AvgxGA': 1.3,
        'Home_xGDiff': 0.2, 'Away_xGDiff': -0.2,
        'Home_Elo': 1500, 'Away_Elo': 1500, 'Elo_Diff': 0
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
    
    
    # Calculate probability matrix for heatmap
    prob_matrix = calculate_poisson_probs(home_goals_exp, away_goals_exp)
    
    # Calculate H/D/A probabilities
    outcome_probs = calculate_outcome_probs(home_goals_exp, away_goals_exp)
    
    return {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'HomeGoals_Exp': round(home_goals_exp, 2),
        'AwayGoals_Exp': round(away_goals_exp, 2),
        'PredictedScore': f"{int(round(home_goals_exp))} - {int(round(away_goals_exp))}",
        'WinProbs': outcome_probs,
        'ProbMatrix': prob_matrix
    }

def get_next_fixtures():
    """
    Fetch upcoming fixtures dynamically from the API.
    Format: (HomeTeam, AwayTeam, MatchDate)
    Returns fixtures for the current matchday and next matchday.
    """
    if not HAS_FIXTURE_SCRAPER:
        print("Fixture scraper not available, returning empty fixtures")
        return [], [], 24
    
    try:
        # Get current matchday from API
        current_md = get_current_matchday()
        print(f"Current matchday from API: {current_md}")
        
        # Fetch current matchday fixtures
        current_fixtures = scrape_fixtures(current_md)
        current_list = [
            (fix['HomeTeam'], fix['AwayTeam'], fix['MatchDate'])
            for fix in current_fixtures
            if fix['HomeTeam'] and fix['AwayTeam']
        ]
        
        # Fetch next matchday fixtures
        next_fixtures = scrape_fixtures(current_md + 1)
        next_list = [
            (fix['HomeTeam'], fix['AwayTeam'], fix['MatchDate'])
            for fix in next_fixtures
            if fix['HomeTeam'] and fix['AwayTeam']
        ]
        
        print(f"Fetched {len(current_list)} current fixtures, {len(next_list)} next fixtures")
        return current_list, next_list, current_md
        
    except Exception as e:
        print(f"Error fetching fixtures from API: {e}")
        return [], [], 24

def get_past_results(current_matchday=None):
    """
    Fetch past results dynamically from the API.
    Gets the previous matchday's completed games with actual scores.
    """
    if not HAS_FIXTURE_SCRAPER:
        print("Fixture scraper not available, returning empty results")
        return [], 23
    
    try:
        # Determine the previous matchday
        if current_matchday is None:
            current_matchday = get_current_matchday()
        
        past_md = current_matchday - 1
        if past_md < 1:
            return [], 0
        
        print(f"Fetching past results for matchday {past_md}")
        
        # Fetch finished matches from the previous matchday
        matches = fetch_matches(matchday=past_md, status="FINISHED")
        
        results = []
        for m in matches:
            home = m.get("homeTeam", {}).get("shortName", m.get("homeTeam", {}).get("name", ""))
            away = m.get("awayTeam", {}).get("shortName", m.get("awayTeam", {}).get("name", ""))
            
            # Normalize team names
            home = normalize_team_name(home)
            away = normalize_team_name(away)
            
            # Get actual score
            ft = m.get("score", {}).get("fullTime", {})
            home_goals = ft.get("home", 0)
            away_goals = ft.get("away", 0)
            actual = f"{home_goals} - {away_goals}"
            
            # Get match date
            utc_date = m.get("utcDate", "")
            match_date = utc_date[:10] if utc_date else ""
            
            results.append({
                "Home": home,
                "Away": away,
                "Actual": actual,
                "Date": match_date,
                "PredictedScore": "-",  # Will be filled from saved predictions
                "WinProbs": {"H": 0, "D": 0, "A": 0}
            })
        
        print(f"Fetched {len(results)} past results for matchday {past_md}")
        return results, past_md
        
    except Exception as e:
        print(f"Error fetching past results: {e}")
        return [], 23

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
        result = predict_match(args.home, args.away, models, df, features_df)
        if 'error' in result:
            print(result['error'])
        else:
            print(f"\n{result['HomeTeam']} vs {result['AwayTeam']}")
            print(f"Expected Goals: {result['HomeGoals_Exp']:.2f} - {result['AwayGoals_Exp']:.2f}")
            print(f"PREDICTION: {result['PredictedScore']}")
    
    elif args.json:
        # Batch predictions for current matchday
        matchday_current, matchday_next, current_md_num = get_next_fixtures()
        
        # Current matchday predictions
        current_predictions = []
        for h, a, date_str in matchday_current:
            if h in teams and a in teams:
                result = predict_match(h, a, models, df, features_df)
                if 'error' not in result:
                    result['MatchDate'] = date_str
                    result['Actual'] = '-'  # Will be updated when game is played
                    current_predictions.append(result)
        
        # Save with matchday info
        output = {
            'matchday': current_md_num,
            'predictions': current_predictions
        }
        
        with open(args.json, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Predictions saved to {args.json}")
        
        if args.past_results_json:
            past_results_list, past_md_num = get_past_results(current_md_num)
            past_output = {
                'matchday': past_md_num,
                'results': past_results_list
            }
            
            with open(args.past_results_json, 'w') as f:
                json.dump(past_output, f, indent=4)
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
            
            result = predict_match(h, a, models, df, features_df)
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
