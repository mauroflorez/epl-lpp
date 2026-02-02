"""
Advanced Data Loader for EPL Goal Prediction

Fetches additional data sources using the soccerdata library:
- xG (Expected Goals) and xGA (Expected Goals Against) from Understat
- Elo ratings from ClubElo

This data enhances the prediction model with better offensive/defensive metrics.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Configuration
DATA_DIR = "data"
XG_FILE = os.path.join(DATA_DIR, "xg_data.csv")
ELO_FILE = os.path.join(DATA_DIR, "elo_data.csv")

# Season mapping for Understat (uses format like "2024/2025")
UNDERSTAT_SEASONS = [
    "2018/2019", "2019/2020", "2020/2021", 
    "2021/2022", "2022/2023", "2023/2024", "2024/2025", "2025/2026"
]

# Team name mappings between different sources
TEAM_NAME_MAPPING = {
    # Understat -> Football-Data naming
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Brighton and Hove Albion": "Brighton",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Sheffield United": "Sheffield United",
    "Leicester City": "Leicester",
    "Leeds United": "Leeds",
    "Aston Villa": "Aston Villa",
    "West Bromwich Albion": "West Brom",
    "AFC Bournemouth": "Bournemouth",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich",
}

# ClubElo team name mapping
ELO_TEAM_MAPPING = {
    "ManUnited": "Man United",
    "ManCity": "Man City",
    "NottmForest": "Nott'm Forest",
    "WestHam": "West Ham",
    "AstonVilla": "Aston Villa",
    "CrystalPalace": "Crystal Palace",
    "SheffieldUnited": "Sheffield United",
    "LeicesterCity": "Leicester",
    "LeedsUnited": "Leeds",
    "WestBrom": "West Brom",
    "IpswichTown": "Ipswich",
    "LutonTown": "Luton",
    "AFCBournemouth": "Bournemouth",
}


def standardize_team_name(name, source="understat"):
    """Standardize team names to match Football-Data.co.uk format."""
    if source == "understat":
        return TEAM_NAME_MAPPING.get(name, name)
    elif source == "elo":
        return ELO_TEAM_MAPPING.get(name, name)
    return name


def fetch_understat_xg(seasons=None, use_cache=True):
    """
    Fetch xG and xGA data for each match from Understat.
    
    Returns a DataFrame with:
    - Date, HomeTeam, AwayTeam
    - home_xG, away_xG (expected goals for each team)
    """
    if use_cache and os.path.exists(XG_FILE):
        print(f"Loading cached xG data from {XG_FILE}...")
        df = pd.read_csv(XG_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    
    try:
        import soccerdata as sd
    except ImportError:
        print("ERROR: soccerdata not installed. Run: pip install soccerdata")
        return None
    
    if seasons is None:
        seasons = UNDERSTAT_SEASONS
    
    print(f"Fetching xG data from Understat for seasons: {seasons}")
    
    all_data = []
    
    for season in seasons:
        try:
            print(f"  Fetching season {season}...")
            understat = sd.Understat(leagues="ENG-Premier League", seasons=season)
            
            # Get team match stats which include xG
            team_stats = understat.read_team_match_stats()
            
            if team_stats is not None and len(team_stats) > 0:
                # Reset index to get columns
                team_stats = team_stats.reset_index()
                
                # The data comes per-team per-match, we need to aggregate to match level
                # Group by game_id to get both teams' stats
                all_data.append(team_stats)
                print(f"    Loaded {len(team_stats)} team-match records")
        except Exception as e:
            print(f"  Warning: Could not fetch {season}: {e}")
            continue
    
    if not all_data:
        print("No xG data could be fetched.")
        return None
    
    # Combine all seasons
    combined = pd.concat(all_data, ignore_index=True)
    
    # Process to get match-level xG data
    xg_df = process_understat_data(combined)
    
    # Save cache
    if xg_df is not None and len(xg_df) > 0:
        xg_df.to_csv(XG_FILE, index=False)
        print(f"Saved {len(xg_df)} matches with xG data to {XG_FILE}")
    
    return xg_df


def process_understat_data(df):
    """
    Process raw Understat data into match-level xG data.
    
    The soccerdata library returns data with columns like:
    - home_team, away_team, date, home_xg, away_xg (game-level format)
    """
    if df is None or len(df) == 0:
        return None
    
    print(f"Processing Understat data with columns: {df.columns.tolist()}")
    
    matches = []
    
    # Check if data is already in game-level format (one row per game)
    if 'home_xg' in df.columns and 'away_xg' in df.columns:
        print("Data is in game-level format with home_xg/away_xg columns")
        
        for _, row in df.iterrows():
            # Get team names
            home_team = row.get('home_team', '')
            away_team = row.get('away_team', '')
            
            # Get date
            date = row.get('date', None)
            
            # Get xG values
            home_xg = float(row.get('home_xg', 0))
            away_xg = float(row.get('away_xg', 0))
            
            if home_team and away_team and date:
                matches.append({
                    'Date': pd.to_datetime(date),
                    'HomeTeam': standardize_team_name(home_team, 'understat'),
                    'AwayTeam': standardize_team_name(away_team, 'understat'),
                    'home_xG': home_xg,
                    'away_xG': away_xg,
                    'home_xGA': away_xg,  # xGA is opponent's xG
                    'away_xGA': home_xg,
                })
        
        print(f"Processed {len(matches)} matches from game-level format")
    
    # Check for older team-level format with 'xG' column
    elif 'xG' in df.columns or 'xg' in df.columns:
        xg_col = 'xG' if 'xG' in df.columns else 'xg'
        print(f"Data is in team-level format with {xg_col} column")
        
        # Group by game to pair home and away teams
        if 'game' in df.columns:
            for game_id, group in df.groupby('game'):
                home_row = group[group['venue'] == 'home'] if 'venue' in group.columns else None
                away_row = group[group['venue'] == 'away'] if 'venue' in group.columns else None
                
                if home_row is not None and len(home_row) > 0 and away_row is not None and len(away_row) > 0:
                    home_row = home_row.iloc[0]
                    away_row = away_row.iloc[0]
                    
                    matches.append({
                        'Date': pd.to_datetime(home_row.get('date', home_row.get('Date'))),
                        'HomeTeam': standardize_team_name(home_row.get('team', home_row.get('Team')), 'understat'),
                        'AwayTeam': standardize_team_name(away_row.get('team', away_row.get('Team')), 'understat'),
                        'home_xG': float(home_row[xg_col]),
                        'away_xG': float(away_row[xg_col]),
                        'home_xGA': float(away_row[xg_col]),
                        'away_xGA': float(home_row[xg_col]),
                    })
    else:
        print(f"Warning: Could not find xG columns. Available columns: {df.columns.tolist()}")
        return None
    
    if not matches:
        print("Could not process Understat data into match format.")
        return None
    
    result = pd.DataFrame(matches)
    result['Date'] = pd.to_datetime(result['Date'])
    
    # Remove duplicates (same match from both team perspectives)
    result = result.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'])
    
    print(f"Final processed xG data: {len(result)} unique matches")
    return result


def fetch_elo_ratings(use_cache=True, cache_hours=24):
    """
    Fetch current Elo ratings for all EPL teams from ClubElo.
    
    Returns a DataFrame with:
    - Team, Elo
    """
    # Check cache freshness
    if use_cache and os.path.exists(ELO_FILE):
        file_time = datetime.fromtimestamp(os.path.getmtime(ELO_FILE))
        if datetime.now() - file_time < timedelta(hours=cache_hours):
            print(f"Loading cached Elo data from {ELO_FILE}...")
            return pd.read_csv(ELO_FILE)
    
    try:
        import soccerdata as sd
    except ImportError:
        print("ERROR: soccerdata not installed. Run: pip install soccerdata")
        return None
    
    print("Fetching current Elo ratings from ClubElo...")
    
    try:
        elo = sd.ClubElo()
        current_elo = elo.read_by_date()
        
        if current_elo is None or len(current_elo) == 0:
            print("No Elo data returned.")
            return None
        
        # Reset index and filter to EPL teams
        current_elo = current_elo.reset_index()
        
        # ClubElo uses 'team' and 'elo' columns
        team_col = 'team' if 'team' in current_elo.columns else 'Team'
        elo_col = 'elo' if 'elo' in current_elo.columns else 'Elo'
        
        # Filter to English teams (by league if available, or by known team names)
        if 'league' in current_elo.columns:
            epl_elo = current_elo[current_elo['league'] == 'ENG-Premier League'].copy()
        else:
            # Filter by known EPL team names
            epl_teams = [
                'Arsenal', 'AstonVilla', 'Bournemouth', 'Brentford', 'Brighton',
                'Burnley', 'Chelsea', 'CrystalPalace', 'Everton', 'Fulham',
                'Liverpool', 'LutonTown', 'ManCity', 'ManUnited', 'Newcastle',
                'NottmForest', 'SheffieldUnited', 'Tottenham', 'WestHam', 'Wolves',
                'Leicester', 'Leeds', 'Ipswich', 'Southampton', 'Sunderland'
            ]
            epl_elo = current_elo[current_elo[team_col].isin(epl_teams)].copy()
        
        # Standardize team names
        epl_elo['Team'] = epl_elo[team_col].apply(lambda x: standardize_team_name(x, 'elo'))
        epl_elo['Elo'] = epl_elo[elo_col]
        
        result = epl_elo[['Team', 'Elo']].copy()
        
        # Save cache
        result.to_csv(ELO_FILE, index=False)
        print(f"Saved Elo data for {len(result)} teams to {ELO_FILE}")
        
        return result
        
    except Exception as e:
        print(f"Error fetching Elo ratings: {e}")
        return None


def get_historical_elo(team, date):
    """
    Get Elo rating for a team at a specific date.
    Uses cached data or fetches if needed.
    """
    try:
        import soccerdata as sd
        elo = sd.ClubElo()
        
        # Standardize team name for ClubElo lookup
        elo_name = None
        for elo_key, fd_name in ELO_TEAM_MAPPING.items():
            if fd_name == team:
                elo_name = elo_key
                break
        
        if elo_name is None:
            elo_name = team
        
        team_history = elo.read_team_history(elo_name)
        
        if team_history is not None and len(team_history) > 0:
            team_history = team_history.reset_index()
            team_history['date'] = pd.to_datetime(team_history['from'])
            
            # Find closest date before the match
            before_match = team_history[team_history['date'] <= date]
            if len(before_match) > 0:
                return float(before_match.iloc[-1]['elo'])
        
        return None
    except:
        return None


def merge_xg_with_matches(matches_df, xg_df):
    """
    Merge xG data with existing match data.
    
    Matches on Date, HomeTeam, AwayTeam.
    """
    if xg_df is None or len(xg_df) == 0:
        print("No xG data to merge.")
        return matches_df
    
    # Ensure Date columns are datetime
    matches_df['Date'] = pd.to_datetime(matches_df['Date'], dayfirst=True)
    xg_df['Date'] = pd.to_datetime(xg_df['Date'])
    
    # Normalize dates to date only (no time)
    matches_df['DateOnly'] = matches_df['Date'].dt.date
    xg_df['DateOnly'] = xg_df['Date'].dt.date
    
    # Merge
    merged = pd.merge(
        matches_df,
        xg_df[['DateOnly', 'HomeTeam', 'AwayTeam', 'home_xG', 'away_xG', 'home_xGA', 'away_xGA']],
        on=['DateOnly', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    merged = merged.drop(columns=['DateOnly'])
    
    matched = merged['home_xG'].notna().sum()
    print(f"Matched xG data for {matched}/{len(merged)} matches ({100*matched/len(merged):.1f}%)")
    
    return merged


def merge_elo_with_matches(matches_df, elo_df):
    """
    Merge current Elo ratings with match data.
    
    For historical matches, uses current Elo as approximation.
    For more accuracy, would need to fetch historical Elo per date.
    """
    if elo_df is None or len(elo_df) == 0:
        print("No Elo data to merge.")
        return matches_df
    
    # Merge home team Elo
    merged = pd.merge(
        matches_df,
        elo_df.rename(columns={'Team': 'HomeTeam', 'Elo': 'Home_Elo'}),
        on='HomeTeam',
        how='left'
    )
    
    # Merge away team Elo
    merged = pd.merge(
        merged,
        elo_df.rename(columns={'Team': 'AwayTeam', 'Elo': 'Away_Elo'}),
        on='AwayTeam',
        how='left'
    )
    
    # Calculate Elo difference
    merged['Elo_Diff'] = merged['Home_Elo'] - merged['Away_Elo']
    
    matched = merged['Home_Elo'].notna().sum()
    print(f"Matched Elo data for {matched}/{len(merged)} matches")
    
    return merged


def load_and_merge_all_data(matches_df):
    """
    Main function to load all advanced data and merge with matches.
    
    Args:
        matches_df: DataFrame with match data from Football-Data.co.uk
        
    Returns:
        DataFrame with additional xG, xGA, and Elo columns
    """
    print("\n" + "="*60)
    print("LOADING ADVANCED DATA (xG, xGA, Elo)")
    print("="*60)
    
    # Fetch xG data
    xg_df = fetch_understat_xg()
    
    # Fetch Elo ratings
    elo_df = fetch_elo_ratings()
    
    # Merge with matches
    if xg_df is not None:
        matches_df = merge_xg_with_matches(matches_df, xg_df)
    
    if elo_df is not None:
        matches_df = merge_elo_with_matches(matches_df, elo_df)
    
    print("="*60)
    print("ADVANCED DATA LOADING COMPLETE")
    print("="*60 + "\n")
    
    return matches_df


def main():
    """Test the data loading."""
    print("Testing Advanced Data Loader...")
    
    # Test xG fetching
    print("\n--- Testing Understat xG ---")
    xg_df = fetch_understat_xg(seasons=["2024/2025"])
    if xg_df is not None:
        print(f"Fetched {len(xg_df)} matches with xG data")
        print(xg_df.head())
    
    # Test Elo fetching
    print("\n--- Testing ClubElo ---")
    elo_df = fetch_elo_ratings()
    if elo_df is not None:
        print(f"Fetched Elo for {len(elo_df)} teams")
        print(elo_df.head(10))
    
    print("\nAdvanced data loader test complete!")


if __name__ == "__main__":
    main()
