"""
Bayesian Poisson Model for EPL Goal Prediction

Uses bambi (built on PyMC) to train Bayesian Poisson regression models that learn:
- Team attack strength (from HomeTeam/AwayTeam effects)
- Adversary defensive weakness 
- League position effects
- Rolling shot statistics

Trains on ALL observed games in the current season.
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuration
FEATURES_FILE = "data/features.csv"
MODEL_DIR = "models"
CURRENT_SEASON = 2526  # Season 25/26

def load_features():
    df = pd.read_csv(FEATURES_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def filter_current_season(df):
    """
    Filter to only include games from the current season (25/26).
    Season started August 15, 2025.
    """
    if 'Season' in df.columns:
        return df[df['Season'] == CURRENT_SEASON].copy()
    else:
        # Fallback: filter by date if Season column not available
        season_start = pd.to_datetime('2025-08-15')
        return df[df['Date'] >= season_start].copy()

def train_bayesian_poisson(train_df):
    """
    Train Bayesian Poisson models for home and away goals.
    
    Model specification:
    - FTHG ~ HomeTeam + AwayTeam + Home_Position + Away_Position + rolling_stats
    - FTAG ~ AwayTeam + HomeTeam + Away_Position + Home_Position + rolling_stats
    
    The model learns:
    - HomeTeam effect in FTHG model = attack strength
    - AwayTeam effect in FTHG model = defensive weakness when facing that team
    """
    # Use statsmodels initially for faster training
    # Bambi requires g++ compiler for good performance
    print(f"\nTraining Poisson GLM models on {len(train_df)} games...")
    print(f"Teams in dataset: {train_df['HomeTeam'].nunique()}")
    
    return train_statsmodels_fallback(train_df)
    
    print(f"\nTraining Bayesian Poisson models on {len(train_df)} games...")
    print(f"Teams in dataset: {train_df['HomeTeam'].nunique()}")
    
    # Prepare formula with available columns
    base_features = []
    
    # Always include team effects (categorical)
    base_features.append("HomeTeam")
    base_features.append("AwayTeam")
    
    # Add position if available
    if 'Home_Position' in train_df.columns:
        base_features.append("Home_Position")
        base_features.append("Away_Position")
    
    # Add rolling stats if available
    rolling_cols = [
        'Home_AvgGoalsScored', 'Home_AvgGoalsConceded',
        'Away_AvgGoalsScored', 'Away_AvgGoalsConceded',
        'Home_AvgShots', 'Home_AvgShotsOnTarget',
        'Away_AvgShots', 'Away_AvgShotsOnTarget'
    ]
    
    for col in rolling_cols:
        if col in train_df.columns:
            base_features.append(col)
    
    formula_home = "FTHG ~ " + " + ".join(base_features)
    formula_away = "FTAG ~ " + " + ".join(base_features)
    
    print(f"\nHome Goals Formula: {formula_home[:80]}...")
    print(f"Away Goals Formula: {formula_away[:80]}...")
    
    # Train Home Goals Model
    print("\nFitting Home Goals Model (this may take a few minutes)...")
    model_home = bmb.Model(
        formula_home,
        data=train_df,
        family="poisson"
    )
    
    # Use fewer draws for faster training (can increase for production)
    results_home = model_home.fit(
        draws=2000,
        chains=2,
        tune=1000,
        progressbar=True
    )
    
    # Train Away Goals Model
    print("\nFitting Away Goals Model...")
    model_away = bmb.Model(
        formula_away,
        data=train_df,
        family="poisson"
    )
    
    results_away = model_away.fit(
        draws=2000,
        chains=2,
        tune=1000,
        progressbar=True
    )
    
    print("\nModel training complete!")
    
    # Print summary of team effects
    print("\n--- Team Attack Strength (top 5 from Home model) ---")
    try:
        summary = az.summary(results_home)
        team_effects = summary[summary.index.str.contains('HomeTeam')]
        if len(team_effects) > 0:
            print(team_effects.head())
    except Exception as e:
        print(f"Could not print summary: {e}")
    
    return {
        'model_home': model_home,
        'results_home': results_home,
        'model_away': model_away,
        'results_away': results_away,
        'formula_home': formula_home,
        'formula_away': formula_away,
        'model_type': 'bayesian_poisson'
    }

def train_statsmodels_fallback(train_df):
    """
    Fallback to statsmodels GLM if bambi is not available.
    Uses the same formula structure.
    """
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    
    print("\nUsing statsmodels GLM fallback...")
    
    # Build formula
    features = []
    
    # Categorical team effects using C() notation
    features.append("C(HomeTeam)")
    features.append("C(AwayTeam)")
    
    # Add numeric features if available
    numeric_cols = [
        'Home_Position', 'Away_Position',
        'Home_AvgGoalsScored', 'Home_AvgGoalsConceded',
        'Away_AvgGoalsScored', 'Away_AvgGoalsConceded',
        'Home_AvgShots', 'Away_AvgShots'
    ]
    
    for col in numeric_cols:
        if col in train_df.columns:
            features.append(col)
    
    formula_home = "FTHG ~ " + " + ".join(features)
    formula_away = "FTAG ~ " + " + ".join(features)
    
    print(f"Formula: {formula_home[:60]}...")
    
    # Train models
    model_home = smf.glm(formula=formula_home, data=train_df, family=sm.families.Poisson()).fit()
    model_away = smf.glm(formula=formula_away, data=train_df, family=sm.families.Poisson()).fit()
    
    print(f"\nHome Model - AIC: {model_home.aic:.2f}")
    print(f"Away Model - AIC: {model_away.aic:.2f}")
    
    return {
        'model_home': model_home,
        'model_away': model_away,
        'formula_home': formula_home,
        'formula_away': formula_away,
        'model_type': 'statsmodels_glm'
    }

def save_models(models):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    model_path = os.path.join(MODEL_DIR, "models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to {model_path}")

def main():
    print("=" * 60)
    print("BAYESIAN POISSON MODEL TRAINING")
    print(f"Season: 25/26 | Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("=" * 60)
    
    # Load features
    df = load_features()
    print(f"Loaded {len(df)} total games")
    
    # Filter to current season only
    train_df = filter_current_season(df)
    print(f"Current season games: {len(train_df)}")
    
    if len(train_df) < 50:
        print("WARNING: Less than 50 games available. Predictions may be unreliable.")
    
    # Train models
    models = train_bayesian_poisson(train_df)
    
    # Save models
    save_models(models)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
