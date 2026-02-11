"""
Fixture scraper for Premier League matches.
Uses football-data.org API to get fixtures and results.
"""

import requests
import json
from datetime import datetime, timedelta
import os

# football-data.org API
API_BASE = "https://api.football-data.org/v4"
# Free tier API key (you can register at football-data.org for your own)
API_KEY = os.environ.get("FOOTBALL_DATA_API_KEY", "")

# Timezone offset for PT (Pacific Time is UTC-8, or UTC-7 during DST)
PT_OFFSET = timedelta(hours=-8)

# Team name mapping: API name -> Local name
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Burnley FC": "Burnley",
    # Short names from API
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton Hove": "Brighton",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich",
    "Ipswich": "Ipswich",
    "Leeds United": "Leeds",
    "Leeds": "Leeds",
    "Leicester City": "Leicester",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Man City",
    "Manchester City": "Man City",
    "Man United": "Man United",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Newcastle": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham": "Nott'm Forest",
    "Nott'm Forest": "Nott'm Forest",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Spurs": "Tottenham",
    "Tottenham": "Tottenham",
    "West Ham United": "West Ham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Wolves": "Wolves",
    "Burnley": "Burnley",
}

def normalize_team_name(name):
    """Normalize team name to match our local data."""
    return TEAM_NAME_MAP.get(name, name)

def get_headers():
    """Get API headers."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-Auth-Token"] = API_KEY
    return headers

def fetch_matches(matchday=None, status=None):
    """
    Fetch Premier League matches.
    
    Args:
        matchday: Optional matchday number (1-38)
        status: Optional status filter (SCHEDULED, LIVE, FINISHED, etc.)
    
    Returns:
        List of matches
    """
    url = f"{API_BASE}/competitions/PL/matches"
    params = {}
    
    if matchday:
        params["matchday"] = matchday
    if status:
        params["status"] = status
    
    try:
        response = requests.get(url, headers=get_headers(), params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("matches", [])
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        return []

def convert_to_pt(utc_time_str):
    """Convert UTC time to Pacific Time."""
    try:
        utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
        pt_time = utc_time + PT_OFFSET
        return pt_time
    except:
        return None

def format_match_data(match):
    """Format match data for our application."""
    home_team = match.get("homeTeam", {}).get("shortName", match.get("homeTeam", {}).get("name", "Unknown"))
    away_team = match.get("awayTeam", {}).get("shortName", match.get("awayTeam", {}).get("name", "Unknown"))
    
    # Normalize team names to match our local data
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)
    
    # Get match time
    utc_date = match.get("utcDate", "")
    pt_time = convert_to_pt(utc_date)
    
    match_date = pt_time.strftime("%Y-%m-%d") if pt_time else ""
    match_time_pt = pt_time.strftime("%H:%M") if pt_time else ""
    date_display = pt_time.strftime("%d/%m") if pt_time else ""
    
    # Get score if finished
    status = match.get("status", "")
    score = match.get("score", {})
    full_time = score.get("fullTime", {})
    
    if status == "FINISHED":
        home_goals = full_time.get("home", 0)
        away_goals = full_time.get("away", 0)
        actual = f"{home_goals} - {away_goals}"
    else:
        actual = "-"
    
    return {
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "MatchDate": match_date,
        "MatchTime_PT": match_time_pt,
        "DateDisplay": date_display,
        "Status": status,
        "Actual": actual
    }

def get_current_matchday():
    """Get the current matchday from the API."""
    url = f"{API_BASE}/competitions/PL"
    try:
        response = requests.get(url, headers=get_headers(), timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("currentSeason", {}).get("currentMatchday", 24)
    except:
        return 24  # Default fallback

def scrape_fixtures(matchday):
    """
    Scrape fixtures for a specific matchday.
    
    Returns:
        List of formatted match data
    """
    matches = fetch_matches(matchday=matchday)
    
    if not matches:
        print(f"No matches found for matchday {matchday}")
        return []
    
    formatted = [format_match_data(m) for m in matches]
    return formatted

def update_predictions_with_actual(predictions_file, matchday):
    """
    Update predictions.json with actual results from API.
    """
    matches = fetch_matches(matchday=matchday)
    
    if not matches:
        print("No matches to update")
        return
    
    # Create lookup by team names (normalized)
    results = {}
    for m in matches:
        home = m.get("homeTeam", {}).get("shortName", m.get("homeTeam", {}).get("name", ""))
        away = m.get("awayTeam", {}).get("shortName", m.get("awayTeam", {}).get("name", ""))
        
        # Normalize names to match our local data
        home = normalize_team_name(home)
        away = normalize_team_name(away)
        
        if m.get("status") == "FINISHED":
            ft = m.get("score", {}).get("fullTime", {})
            results[f"{home} vs {away}"] = f"{ft.get('home', 0)} - {ft.get('away', 0)}"
    
    # Update predictions file
    try:
        with open(predictions_file, 'r') as f:
            data = json.load(f)
        
        predictions = data.get("predictions", data) if isinstance(data, dict) else data
        
        for pred in predictions:
            key = f"{pred['HomeTeam']} vs {pred['AwayTeam']}"
            if key in results:
                pred["Actual"] = results[key]
        
        with open(predictions_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Updated {len(results)} results in {predictions_file}")
    except Exception as e:
        print(f"Error updating predictions: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape EPL Fixtures")
    parser.add_argument("--matchday", type=int, help="Matchday number (1-38)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--update-actuals", type=str, help="Update actuals in predictions file")
    args = parser.parse_args()
    
    # Get current matchday if not specified
    matchday = args.matchday or get_current_matchday()
    print(f"Fetching Matchday {matchday} fixtures...")
    
    fixtures = scrape_fixtures(matchday)
    
    if args.output and fixtures:
        with open(args.output, 'w') as f:
            json.dump(fixtures, f, indent=4)
        print(f"Saved {len(fixtures)} fixtures to {args.output}")
    else:
        for fix in fixtures:
            status = "✓" if fix["Actual"] != "-" else "⏳"
            print(f"{status} {fix['HomeTeam']} vs {fix['AwayTeam']} ({fix['DateDisplay']} {fix['MatchTime_PT']} PT) - {fix['Actual']}")
    
    if args.update_actuals:
        update_predictions_with_actual(args.update_actuals, matchday)
