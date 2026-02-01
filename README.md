# epl-lpp: EPL Match Prediction

> ⚠️ **Disclaimer**: This is an experimental project I had in mind for a while but didn't have time to implement. I'm now building it with the help of AI (Antigravity/Claude). **This is purely for testing and learning purposes - predictions should NOT be trusted for any real decisions!**

## Overview

A Poisson regression model for predicting English Premier League match scores. The model learns team attack strength and defensive weakness from historical match data.

## How It Works

The model uses a Poisson GLM with the following structure:

```
Home Goals ~ HomeTeam + AwayTeam + Positions + Rolling Stats
Away Goals ~ AwayTeam + HomeTeam + Positions + Rolling Stats
```

- **HomeTeam effect** → Attack strength of the home team
- **AwayTeam effect** → Defensive weakness when facing that opponent
- **Positions** → Current league standings
- **Rolling Stats** → Goals, shots, shots on target (last 5 games)

## Features

- Rolling averages for goals scored/conceded
- Rolling averages for shots and shots on target
- Dynamic league position calculation
- Trained on all current season games

## Usage

```bash
# Generate features
python feature_engineering.py

# Train model
python train_model.py

# Predict a single match
python predict_games.py --home "Liverpool" --away "Arsenal"

# Generate all predictions (JSON output)
python predict_games.py --json data/predictions.json
```

## Requirements

```
pip install -r requirements.txt
```

## Project Structure

```
├── data/
│   ├── matches.csv       # Raw match data
│   ├── features.csv      # Processed features
│   └── predictions.json  # Model predictions
├── models/
│   └── models.pkl        # Trained model
├── feature_engineering.py
├── train_model.py
├── predict_games.py
└── requirements.txt
```

---

*Built with the assistance of AI for educational and experimental purposes.*
