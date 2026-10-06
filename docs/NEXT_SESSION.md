# Next Session Plan

## Project status
- End-to-end pipeline shipped (collect → clean → features → train → backtest → dashboard)
- Only profitable config: H/D, odds 3-4, edge ≥5% → +6.7% ROI on 42 bets (small sample)
- All other strategies negative after Kenyan tax

## Priority 1: Add xG features (Understat)
- pip install understat
- Scrape EPL xG per match 2020-2025
- Join to pl_clean.csv
- Add rolling xG features to feature_engineering.py
- Retrain, re-run odds_3_4 backtest
- Success = ROI > +10% on more bets

## Priority 2: Extend to more seasons (1993-2025 available)
- Update SEASON_MAP in collect_fd_uk.py
- Rerun full pipeline

## Priority 3: Niche markets (paid API-Football)
- Corners, cards, player props
- Need API key, ~$15/month

## Run commands
- Collect: python collect_fd_uk.py
- Clean: python clean_data.py
- Features: python src/data_processing/feature_engineering.py
- Train: python src/models/train.py
- Backtest: python -m src.models.run_backtest
- O/U backtest: python -m src.models.run_ou_backtest
- Dashboard: streamlit run src/dashboard/app.py
