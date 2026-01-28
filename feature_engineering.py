import pandas as pd
import numpy as np
import os

# Configuration
DATA_FILE = "data/matches.csv"
OUTPUT_FILE = "data/features.csv"
ROLLING_WINDOW = 5

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please run data_loader.py first.")
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} rows from {DATA_FILE}")
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    
    # Sort by Date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def calculate_league_position(df):
    """
    Calculate current league position for each team at each match date.
    Uses cumulative points up to (but not including) the current game.
    """
    # Create a team-centric view to calculate standings
    games = []
    
    for _, row in df.iterrows():
        date = row['Date']
        season = row.get('Season', None)
        
        # Home team result
        if row['FTR'] == 'H':
            home_pts, away_pts = 3, 0
        elif row['FTR'] == 'A':
            home_pts, away_pts = 0, 3
        else:
            home_pts, away_pts = 1, 1
            
        games.append({
            'Date': date,
            'Season': season,
            'Team': row['HomeTeam'],
            'Points': home_pts,
            'GF': row['FTHG'],
            'GA': row['FTAG']
        })
        games.append({
            'Date': date,
            'Season': season,
            'Team': row['AwayTeam'],
            'Points': away_pts,
            'GF': row['FTAG'],
            'GA': row['FTHG']
        })
    
    games_df = pd.DataFrame(games).sort_values(['Season', 'Date', 'Team'])
    
    # Calculate cumulative stats per team per season (shifted to not include current game)
    games_df['CumPoints'] = games_df.groupby(['Season', 'Team'])['Points'].transform(
        lambda x: x.shift(1).cumsum().fillna(0)
    )
    games_df['CumGD'] = games_df.groupby(['Season', 'Team']).apply(
        lambda g: (g['GF'] - g['GA']).shift(1).cumsum().fillna(0)
    ).reset_index(level=[0,1], drop=True)
    
    # Calculate position within each date/season
    def calc_position(group):
        # Sort by points (desc), then goal diff (desc)
        sorted_group = group.sort_values(['CumPoints', 'CumGD'], ascending=[False, False])
        sorted_group['Position'] = range(1, len(sorted_group) + 1)
        return sorted_group
    
    games_df = games_df.groupby(['Season', 'Date'], group_keys=False).apply(calc_position)
    
    return games_df[['Date', 'Team', 'Position']]

def calculate_team_stats(df):
    """
    Calculates rolling statistics for each team including:
    - Goals scored/conceded
    - Points
    - Shots and shots on target
    """
    # Check which columns are available
    has_shots = 'HS' in df.columns and 'AS' in df.columns
    has_shots_target = 'HST' in df.columns and 'AST' in df.columns
    
    # Create a team-centric dataframe for HOME games
    home_cols = ['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR']
    if has_shots:
        home_cols.extend(['HS', 'AS'])
    if has_shots_target:
        home_cols.extend(['HST', 'AST'])
        
    home_df = df[home_cols].copy()
    home_df.rename(columns={
        'HomeTeam': 'Team', 
        'FTHG': 'GoalsScored', 
        'FTAG': 'GoalsConceded'
    }, inplace=True)
    if has_shots:
        home_df.rename(columns={'HS': 'Shots', 'AS': 'ShotsConceded'}, inplace=True)
    if has_shots_target:
        home_df.rename(columns={'HST': 'ShotsOnTarget', 'AST': 'ShotsOnTargetConceded'}, inplace=True)
    home_df['IsHome'] = 1
    home_df['Points'] = home_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    
    # Create a team-centric dataframe for AWAY games
    away_cols = ['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR']
    if has_shots:
        away_cols.extend(['AS', 'HS'])
    if has_shots_target:
        away_cols.extend(['AST', 'HST'])
        
    away_df = df[away_cols].copy()
    away_df.rename(columns={
        'AwayTeam': 'Team', 
        'FTAG': 'GoalsScored', 
        'FTHG': 'GoalsConceded'
    }, inplace=True)
    if has_shots:
        away_df.rename(columns={'AS': 'Shots', 'HS': 'ShotsConceded'}, inplace=True)
    if has_shots_target:
        away_df.rename(columns={'AST': 'ShotsOnTarget', 'HST': 'ShotsOnTargetConceded'}, inplace=True)
    away_df['IsHome'] = 0
    away_df['Points'] = away_df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    team_df = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # Calculate rolling stats (shifted by 1 to use only PAST data)
    def rolling_mean(series):
        return series.shift(1).rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    team_df['AvgGoalsScored'] = team_df.groupby('Team')['GoalsScored'].transform(rolling_mean)
    team_df['AvgGoalsConceded'] = team_df.groupby('Team')['GoalsConceded'].transform(rolling_mean)
    team_df['AvgPoints'] = team_df.groupby('Team')['Points'].transform(rolling_mean)
    
    if has_shots:
        team_df['AvgShots'] = team_df.groupby('Team')['Shots'].transform(rolling_mean)
        team_df['AvgShotsConceded'] = team_df.groupby('Team')['ShotsConceded'].transform(rolling_mean)
    
    if has_shots_target:
        team_df['AvgShotsOnTarget'] = team_df.groupby('Team')['ShotsOnTarget'].transform(rolling_mean)
        team_df['AvgShotsOnTargetConceded'] = team_df.groupby('Team')['ShotsOnTargetConceded'].transform(rolling_mean)
    
    return team_df

def merge_features(original_df, team_stats_df, position_df=None):
    """
    Merges calculated team stats back into the original match dataframe.
    """
    # Determine which stats columns are available
    stat_cols = ['AvgGoalsScored', 'AvgGoalsConceded', 'AvgPoints']
    if 'AvgShots' in team_stats_df.columns:
        stat_cols.extend(['AvgShots', 'AvgShotsConceded'])
    if 'AvgShotsOnTarget' in team_stats_df.columns:
        stat_cols.extend(['AvgShotsOnTarget', 'AvgShotsOnTargetConceded'])
    
    # Merge Home Team Stats
    home_stats = team_stats_df[team_stats_df['IsHome'] == 1][['Date', 'Team'] + stat_cols]
    merged_df = pd.merge(original_df, home_stats, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    
    # Rename home stats
    rename_dict = {col: f'Home_{col}' for col in stat_cols}
    merged_df.rename(columns=rename_dict, inplace=True)
    merged_df.drop(columns=['Team'], inplace=True)
    
    # Merge Away Team Stats
    away_stats = team_stats_df[team_stats_df['IsHome'] == 0][['Date', 'Team'] + stat_cols]
    merged_df = pd.merge(merged_df, away_stats, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    
    # Rename away stats
    rename_dict = {col: f'Away_{col}' for col in stat_cols}
    merged_df.rename(columns=rename_dict, inplace=True)
    merged_df.drop(columns=['Team'], inplace=True)
    
    # Merge league positions if available
    if position_df is not None:
        # Home position
        merged_df = pd.merge(
            merged_df, 
            position_df.rename(columns={'Team': 'HomeTeam', 'Position': 'Home_Position'}),
            on=['Date', 'HomeTeam'], 
            how='left'
        )
        # Away position
        merged_df = pd.merge(
            merged_df,
            position_df.rename(columns={'Team': 'AwayTeam', 'Position': 'Away_Position'}),
            on=['Date', 'AwayTeam'],
            how='left'
        )
    
    return merged_df

def main():
    print("Starting feature engineering...")
    try:
        df = load_data()
        
        # Keep relevant columns for the Bayesian model
        needed_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        optional_cols = ['HS', 'AS', 'HST', 'AST', 'Season']
        
        # Add available optional columns
        for col in optional_cols:
            if col in df.columns:
                needed_cols.append(col)
        
        df_subset = df[needed_cols].copy()
        
        # Calculate features
        print("Calculating team rolling stats...")
        team_stats = calculate_team_stats(df_subset)
        
        print("Calculating league positions...")
        try:
            position_df = calculate_league_position(df_subset)
        except Exception as e:
            print(f"Warning: Could not calculate positions: {e}")
            position_df = None
        
        print("Merging features...")
        features_df = merge_features(df_subset, team_stats, position_df)
        
        # Drop rows with NaN (first few games where rolling stats are not available)
        na_count = features_df.isna().any(axis=1).sum()
        print(f"Dropping {na_count} rows with missing data (start of seasons).")
        features_df.dropna(inplace=True)
        
        print(f"Saving {len(features_df)} rows to {OUTPUT_FILE}...")
        features_df.to_csv(OUTPUT_FILE, index=False)
        print("Feature engineering complete.")
        print(f"Columns: {list(features_df.columns)}")
        
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
