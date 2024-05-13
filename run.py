from datetime import datetime
import argparse
import logging
import numpy as np
import os
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.werewolf_env import WerewolfEnvironment
from src.agent import VanillaLanguageAgent, OmniscientLanguageAgent
from src.strategic_agent import DeductiveLanguageAgent, DiverseDeductiveAgent, StrategicLanguageAgent


def main(args):
    # parse args
    parser = argparse.ArgumentParser(description="werewolf run", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", type=str, default="vanilla", 
                        choices=["vanilla", "omniscient", "deductive", "diverse", "strategic"],
                        help="name of the log directory, default=vanilla")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0613", 
                        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-4"], 
                        help="which model to use, default=gpt-3.5-turbo-0613")
    parser.add_argument("--temperature", type=float, default=0.7, help="model's sampling temperature, default=0.7")
    args = parser.parse_known_args(args)[0]

    # make log dir if not exits
    now = datetime.now()
    model_dir = f"{ROOT_DIR}/log/{args.name}/{args.model}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    all_files = os.listdir(model_dir)
    n_logs = len(all_files) if "result.json" not in all_files else len(all_files) - 1
    log_dir = f"{model_dir}/game{n_logs}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = f"{log_dir}/env.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger = logging.getLogger("env")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    print(f"The log files can be found at {log_file}.")

    # make env and agents
    env = WerewolfEnvironment()
    n_player = env.n_player

    if args.name == "vanilla":
        agents = [VanillaLanguageAgent(args.model, args.temperature, f"{log_dir}/player_{i}.log") for i in range(n_player)]
    elif args.name == "omniscient":
        agents = [OmniscientLanguageAgent(args.model, args.temperature, f"{log_dir}/player_{i}.log") for i in range(n_player)]
    elif args.name == "deductive":
        agents = [DeductiveLanguageAgent(args.model, args.temperature, f"{log_dir}/player_{i}.log") for i in range(n_player)]
    elif args.name == "diverse":
        agents = [DiverseDeductiveAgent(args.model, args.temperature, f"{log_dir}/player_{i}.log") for i in range(n_player)]
    elif args.name == "strategic":
        agents = [StrategicLanguageAgent(args.model, args.temperature, f"{log_dir}/player_{i}.log") for i in range(n_player)]
    else:
        raise NotImplementedError

    # run games
    done = False
    obs = env.reset()
    for idx, role in enumerate(env.info["roles"]):
        logger.info(f"player_{idx}: {role}")

    while not done:
        actions = [None for _ in range(n_player)]
        for idx, agent_obs in enumerate(obs):
            if agent_obs is None:
                continue
            print('--------GETTING ALL THE EMBEDDINGS in run.py----------')
            # actions[idx] = agents[idx].act(agent_obs)
            print(agents[idx].get_embeddings(agent_obs))
            print('--------GETTING ALL THE EMBEDDINGS in run.py----------')
        obs, rewards, dones, info = env.step(actions)


        done = np.all(dones)

        # log info
        if info["status"] == "night" and info["n_round"] > 1:
            logger.info(f"day {info['n_round'] - 1} voting: {info['history'][-2]['voting']}")
            logger.info(f"remaining players: {', '.join(f'player_{i} ({env.roles[i]})' for i, alive in enumerate(env.alive) if alive)}.")
        elif info["status"] == "discussion":
            if len(info['history'][-1]['discussion']) == 0:
                logger.info(f"night {info['n_round']}:")
                for item in info['history'][-1]['night']:
                    logger.info(item)
                logger.info(f"day {info['n_round']} announcement: {info['history'][-1]['announcement']}")
                logger.info(f"remaining players: {', '.join(f'player_{i} ({env.roles[i]})' for i, alive in enumerate(env.alive) if alive)}.")
                logger.info(f"day {info['n_round']} discussion:")
            else:
                idx = list(info["history"][-1]["discussion"])[-1]
                logger.info(f"player_{idx} ({info['roles'][idx]}) said: \"{info['history'][-1]['discussion'][idx]}\"")
        elif info["status"] == "voting":
            idx = list(info["history"][-1]["discussion"])[-1]
            logger.info(f"player_{idx} ({info['roles'][idx]}) said: \"{info['history'][-1]['discussion'][idx]}\"")

    # log winner
    logger.info(f"{info['winners']} win the game.")
    print(f"{info['winners']} win the game.")


if __name__ == "__main__":
    main(sys.argv[1:])
