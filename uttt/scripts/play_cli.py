from uttt.env.state import UTTTEnv
from uttt.agents.random import RandomAgent


def main():
    env = UTTTEnv(seed=42)
    agent = RandomAgent()
    moves = 0
    while True:
        a = agent.select_action(env)
        step = env.step(a)
        moves += 1
        if step.terminated:
            winner = step.info["winner"]
            outcome = "draw" if winner == 0 else ("X wins" if winner == 1 else "O wins")
            print(f"Game over in {moves} moves: {outcome}")
            break


if __name__ == "__main__":
    main()
