# UTTT: Ultimate Tic-Tac-Toe Self-Play (Tier 0)

- `uttt/env`: Gym-like environment (rules, state encoding)
- `uttt/mcts`: UCT + transposition table
- `uttt/agents`: random/heuristic baselines, AZ-lite later
- `uttt/eval`: Elo + tournaments
- `uttt/scripts`: CLI runners (self-play, train, eval)
- `uttt/tests`: unit tests

## Quickstart
```bash
conda activate uttt
pre-commit install
pytest
python -m uttt.scripts.play_cli
