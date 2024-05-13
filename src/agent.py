import json
import logging
import numpy as np
import openai
import time


MAX_ERROR = 5


class LanguageAgent(object):
    def __init__(self):
        self.system_prompt = (
            "You are an expert in playing the social deduction game named Werewolf. "
            "The game has seven roles including two Werewolves, one Seer, one Doctor, and three Villagers. "
            "There are seven players including player_0, player_1, player_2, player_3, player_4, player_5, and player_6.\n\n"
            "At the beginning of the game, each player is assigned a hidden role which divides them into the Werewolves and the Villagers (Seer, Doctor, Villagers). "
            "Then the game alternates between the night round and the day round until one side wins the game.\n\n"
            "In the night round: the Werewolves choose one player to kill; "
            "the Seer chooses one player to see if they are a Werewolf; "
            "the Doctor chooses one player including themselves to save without knowing who is chosen by the Werewolves; "
            "the Villagers do nothing.\n\n"
            "In the day round: three phases including an announcement phase, a discussion phase, and a voting phase are performed in order.\n"
            "In the announcement phase, an announcement of last night's result is made to all players. "
            "If player_i was killed and not saved last night, the announcement will be \"player_i was killed.\"; "
            "if a player was killed and saved last night, the announcement will be \"no player was killed.\"\n"
            "In the discussion phase, each remaining player speaks only once in order from player_0 to player_6 to discuss who might be the Werewolves.\n"
            "In the voting phase, each player votes for one player or choose not to vote. "
            "The player with the most votes is eliminated and the game continues to the next night round.\n\n"
            "The Werewolves win the game if the number of remaining Werewolves is equal to the number of remaining Seer, Doctor, and Villagers. "
            "The Seer, Doctor, and Villagers win the game if both Werewolves are eliminated."
        )

    def act(self, obs):
        raise NotImplementedError


class VanillaLanguageAgent(LanguageAgent):
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0.7, log_file=None):
        super().__init__()
        self.n_player = 7
        self.model = model
        self.temperature = temperature
        self.language_observation=''
        self.current_action_candidates=[]

        if log_file != None:
            self.handler = logging.FileHandler(log_file)
            self.handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger = logging.getLogger(log_file[-12:-4])
            self.logger.setLevel(logging.INFO)
            self.logger.addHandler(self.handler)
    
    def act(self, obs):
        self.id = obs["id"]
        self.role = obs["role"]
        self.teammate = obs["teammate"]
        self.n_round = obs["n_round"]
        self.status = obs["status"]
        self.alive = obs["alive"]
        self.seen_players = obs["seen_players"]

        prompt = self.organize_information(obs["history"]) + "\n\n" + self.action_format()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        action = self.get_action(messages)
        return action

    def organize_information(self, history):
        facts = []
        discussions = []
        # id and role
        if self.role == "Werewolf":
            facts.append(f"you are player_{self.id}, your role is {self.role}, your teammate is player_{self.teammate}.")
        else:
            facts.append(f"you are player_{self.id}, your role is {self.role}.")
        # round and phase
        if self.status == "night":
            facts.append(f"current round and phase: night {self.n_round}.")
        elif self.status == "discussion":
            facts.append(f"current round and phase: day {self.n_round} discussion.")
        elif self.status == "voting":
            facts.append(f"current round and phase: day {self.n_round} voting.")
        # remaining players
        players = ", ".join([f"player_{i}" for i, alive in enumerate(self.alive) if alive])
        facts.append(f"remaining players: {players}.")
        # night, announcement, and voting
        for n, h in enumerate(history):
            if "night" in h.keys():
                facts.append(f"night {n + 1}: {h['night']}")
            if "announcement" in h.keys():
                facts.append(f"day {n + 1} announcement: {h['announcement']}")
            if "voting" in h.keys():
                facts.append(f"day {n + 1} voting: {h['voting']}")
            if "discussion" in h.keys():
                for k, v in h["discussion"].items():
                    if k == self.id:
                        discussions.append(f"in day {n + 1} discussion, your said: {v}")
                    else:
                        discussions.append(f"in day {n + 1} discussion, player_{k} said: {v}")

        n_fact = len(facts)
        prompt = "Facts:\n"
        prompt += "\n".join([f"[{i + 1}] {item}" for i, item in enumerate(facts)])

        if len(discussions) > 0:
            prompt += "\n\nDiscussions:\n"
            prompt += "\n".join([f"[{n_fact + i + 1}] {item}" for i, item in enumerate(discussions)])

        if (self.role == "Seer") and (len(self.seen_players) > 0):
            prompt += "\n\nConfirmed roles:"
            for k, v in self.seen_players.items():
                if self.alive[k]:
                    prompt += f"\nplayer_{k}: {v} (alive)."
                else:
                    prompt += f"\nplayer_{k}: {v} (dead)."

        return prompt

    def get_action(self, messages):
        self.logger.info(f"PROMPT\n{messages[-1]['content']}\n")

        result = None
        available_actions = self.get_available_actions()
        n_error = 0
        while result is None and n_error <= MAX_ERROR:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                result = json.loads(response.choices[0].message.content)
                if self.status in ["night", "voting"]:
                    assert result["action"] in available_actions
            except openai.error.RateLimitError:
                print("Reach OpenAI API rate limit, sleep for 1s.")
                time.sleep(1)
                print("Try to get response again...\n")
                result = None
            except openai.error.ServiceUnavailableError:
                print("OpenAI server is overloaded or not ready yet, sleep for 1s.")
                time.sleep(1)
                print("Try to get response again...\n")
                result = None
            except openai.error.InvalidRequestError:
                print("Exceed maximum context length, change model to gpt-3.5-turbo-16k-0613")
                self.model = "gpt-3.5-turbo-16k-0613"
                print("Try to get response again...\n")
                result = None
            except AssertionError:
                print(f"Illegal action: {result['action']}.")
                print(f"Available actions are: {', '.join(available_actions)}.")
                print("Try to get response again...\n")
                result = None
                n_error += 1
                if n_error > MAX_ERROR:
                    print("Reach max error, randomly choose from available actions\n")
                    result = {"action": np.random.choice(available_actions)}
                    self.logger.warning(f"Reach max error, randomly choose from available actions: {result['action']}")
            except Exception as error:
                print(f"{type(error).__name__}: {error}.")
                print("Try to get response again...\n")
                result = None
                n_error += 1
        if result is not None:
            self.logger.info(f"RESPONSE\n{response.choices[0].message.content}\n")
        action = self.parse_action(result)
        return action

    def parse_action(self, result):
        if self.status == "night":
            p_idx = result["action"].find("player")
            action = int(result["action"][p_idx + 7])
        elif self.status == "discussion":
            action = result["discussion"]
        elif self.status == "voting":
            if result["action"] == "do not vote":
                action = -1
            else:
                p_idx = result["action"].find("player")
                action = int(result["action"][p_idx + 7])
        return action

    def action_format(self):
        available_actions = self.get_available_actions()

        if self.status == "night":
            if self.role == "Werewolf":
                if self.alive[self.teammate]:
                    prompt = (
                        f"Now it is night {self.n_round} round, "
                        "you and your teammate should choose one player together to kill. "
                        f"As player_{self.id} and a {self.role}, "
                        "you should first reason about the situation, "
                        f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                        "You should only respond in JSON format as described below.\n"
                        "Response Format:\n"
                        "{\n"
                        "    \"reasoning\": \"reason about the current situation\",\n"
                        "    \"action\": \"kill player_i\"\n"
                        "}\n"
                        "Ensure the response can be parsed by Python json.loads"
                    )
                else:
                    prompt = (
                        f"Now it is night {self.n_round} round, "
                        "you should choose one player to kill. "
                        f"As player_{self.id} and a {self.role}, "
                        "you should first reason about the situation, "
                        f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                        "You should only respond in JSON format as described below.\n"
                        "Response Format:\n"
                        "{\n"
                        "    \"reasoning\": \"reason about the current situation\",\n"
                        "    \"action\": \"kill player_i\"\n"
                        "}\n"
                        "Ensure the response can be parsed by Python json.loads"
                    )
            elif self.role == "Seer":
                prompt = (
                    f"Now it is night {self.n_round} round, "
                    "you should choose one player to see if they are a Werewolf. "
                    f"As player_{self.id} and a {self.role}, "
                    "you should first reason about the situation, "
                    f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"reasoning\": \"reason about the current situation\",\n"
                    "    \"action\": \"see player_i\"\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
            elif self.role == "Doctor":
                prompt = (
                    f"Now it is night {self.n_round} round, "
                    "you should save one player from being killed. "
                    f"As player_{self.id} and a {self.role}, "
                    "you should first reason about the situation, "
                    f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"reasoning\": \"reason about the current situation\",\n"
                    "    \"action\": \"save player_i\"\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
        elif self.status == "discussion":
            prompt = (
                f"Now it is day {self.n_round} dicussion phase and it is your turn to speak. "
                f"As player_{self.id} and a {self.role}, "
                "you should first reason about the current situation only to yourself, "
                "then speak to all other players.\n\n"
                "You should only respond in JSON format as described below.\n"
                "Response Format:\n"
                "{\n"
                "    \"reasoning\": \"reason about the current situation only to yourself\",\n"
                "    \"discussion\": \"speak to all other players\"\n"
                "}\n"
                "Ensure the response can be parsed by Python json.loads"
            )
        elif self.status == "voting":
            if self.role == "Werewolf":
                prompt = (
                    f"Now it is day {self.n_round} voting phase, "
                    "you should vote for one player or do not vote to maximize the Werewolves' benefit. "
                    f"As player_{self.id} and a {self.role}, "
                    f"you should first reason about the current situation, "
                    f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"reasoning\": \"reason about the current situation\",\n"
                    "    \"action\": \"vote for player_i\"\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
            else:
                prompt = (
                    f"Now it is day {self.n_round} voting phase, "
                    "you should vote for one player that is most likely to be a Werewolf or do not vote. "
                    f"As player_{self.id} and a {self.role}, "
                    f"you should first reason about the current situation, "
                    f"then choose from the following actions: {', '.join(available_actions)}.\n\n"
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"reasoning\": \"reason about the current situation\",\n"
                    "    \"action\": \"vote for player_i\"\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
        else:
            raise NotImplementedError
        return prompt

    def get_available_actions(self):
        if self.status == "night":
            if self.role == "Werewolf":
                available_actions = [f"kill player_{i}" for i, alive in enumerate(self.alive) if alive and (i != self.id) and (i != self.teammate)]
            if self.role == "Seer":
                available_actions = [f"see player_{i}" for i, alive in enumerate(self.alive) if alive and (i != self.id)]
            if self.role == "Doctor":
                available_actions = [f"save player_{i}" for i, alive in enumerate(self.alive) if alive]
        elif self.status == "discussion":
            available_actions = None
        elif self.status == "voting":
            available_actions = ["do not vote"] + [f"vote for player_{i}" for i, alive in enumerate(self.alive) if alive and (i != self.id)]
        return available_actions


class OmniscientLanguageAgent(VanillaLanguageAgent):
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0.7, log_file=None, confidence=9):
        super().__init__(model, temperature, log_file)
        self.confidence = confidence

    def act(self, obs):
        self.id = obs["id"]
        self.role = obs["role"]
        self.roles = obs["roles"]
        self.teammate = obs["teammate"]
        self.n_round = obs["n_round"]
        self.status = obs["status"]
        self.alive = obs["alive"]
        self.seen_players = obs["seen_players"]

        prompt = self.organize_information(obs["history"]) + "\n\n" + self.action_format()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        action = self.get_action(messages)
        return action

    def organize_information(self, history):
        print('-------------ORGANIZE INFORMATION CALLLED---------------')
        facts = []
        truths = []
        deceptions = []
        # id and role
        if self.role == "Werewolf":
            facts.append(f"you are player_{self.id}, your role is {self.role}, your teammate is player_{self.teammate}.")
        else:
            facts.append(f"you are player_{self.id}, your role is {self.role}.")
        # round and phase
        if self.status == "night":
            facts.append(f"current round and phase: night {self.n_round}.")
        elif self.status == "discussion":
            facts.append(f"current round and phase: day {self.n_round} discussion.")
        elif self.status == "voting":
            facts.append(f"current round and phase: day {self.n_round} voting.")
        # remaining players
        players = ", ".join([f"player_{i}" for i, alive in enumerate(self.alive) if alive])
        facts.append(f"remaining players: {players}.")
        # night, announcement, and voting
        for n, h in enumerate(history):
            if "night" in h.keys():
                facts.append(f"night {n + 1}: {h['night']}")
            if "announcement" in h.keys():
                facts.append(f"day {n + 1} announcement: {h['announcement']}")
            if "voting" in h.keys():
                facts.append(f"day {n + 1} voting: {h['voting']}")
            if "discussion" in h.keys():
                for k, v in h["discussion"].items():
                    if k == self.id:
                        facts.append(f"in day {n + 1} discussion, your said: {v}")
                    elif self.roles[k] != "Werewolf":
                        truths.append(f"in day {n + 1} discussion, player_{k} said: {v}")
                    else:
                        deceptions.append(f"in day {n + 1} discussion, player_{k} said: {v}")

        # all information
        n_fact = len(facts)
        n_truth = len(truths)
        n_deception = len(deceptions)
        prompt = "Facts:\n"
        prompt += "\n".join([f"[{i + 1}] {item}" for i, item in enumerate(facts)])
        if n_truth > 0:
            prompt += "\n\nPotential truths:\n"
            prompt += "\n".join([f"[{n_fact + i + 1}] {item}" for i, item in enumerate(truths)])
        if n_deception > 0:
            prompt += "\n\nPotential deceptions:\n"
            prompt += "\n".join([f"[{n_fact + n_truth + i + 1}] {item}" for i, item in enumerate(deceptions)])

        # ground truth deduction
        prompt += "\n\nYour deduction:"
        for i, role in enumerate(self.roles):
            if not self.alive[i]:
                continue
            if i == self.id:
                prompt += f"\nplayer_{i}: {self.role}."
            elif self.role == "Werewolf" and i == self.teammate:
                prompt += f"\nplayer_{i}: Werewolf teammate."
            elif self.role == "Seer" and i in self.seen_players.keys():
                prompt += f"\nplayer_{i}: confirmed to be a {self.seen_players[i]}."
            elif self.confidence <= 5:
                prompt += f"\nplayer_{i}: Uncertain."
            elif self.confidence <= 7:
                prompt += f"\nplayer_{i}: may be a {role}."
            elif self.confidence <= 9:
                prompt += f"\nplayer_{i}: likely to be a {role}."
            else:
                prompt += f"\nplayer_{i}: must be a {role}."
        self.language_observation=prompt
        print(prompt)
        print('-------------ORGANIZE INFORMATION CALLLED---------------')
        return prompt
