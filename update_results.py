import json

# Load predictions
with open('predictions.json', 'r') as f:
    data = json.load(f)

# Actual results from January 31st, 2026
results = {
    'Leeds vs Arsenal': '0 - 4',
    'Wolves vs Bournemouth': '0 - 2',
    'Brighton vs Everton': '1 - 1',
    'Chelsea vs West Ham': '3 - 2',
    'Liverpool vs Newcastle': '4 - 1',
}

# Update predictions
for pred in data['predictions']:
    key = f"{pred['HomeTeam']} vs {pred['AwayTeam']}"
    if key in results:
        pred['Actual'] = results[key]
        print(f"Updated: {key} = {results[key]}")

# Save updated predictions
with open('predictions.json', 'w') as f:
    json.dump(data, f, indent=4)

print("\nPredictions updated!")
