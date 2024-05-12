import json
import numpy as np
import openai
import time
from openai import OpenAI
from src.agent import VanillaLanguageAgent
client = OpenAI()

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

MAX_ERROR = 5


class DeductiveLanguageAgent(VanillaLanguageAgent):
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0.7, log_file=None):
        super().__init__(model, temperature, log_file)
        self.facts = [None, None, None]
        self.potential_truths = []
        self.potential_deceptions = []
        self.deduction = {i: {"role": "Uncertain", "confidence": 5, "reliability": 6} for i in range(self.n_player)}
        self.evidence = []

    def act(self, obs):
        self.id = obs["id"]
        self.role = obs["role"]
        self.teammate = obs["teammate"]
        self.n_round = obs["n_round"]
        self.status = obs["status"]
        self.alive = obs["alive"]
        self.seen_players = obs["seen_players"]
        
        self.add_history(obs["new_history"])
        if self.n_round == 1 and self.status == "night":
            pass
        else:
            deduction_prompt = self.organize_information() + "\n\n" + self.deduction_result("previous") + "\n\n" + self.deduction_format()
            deduction_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": deduction_prompt},
            ]
            self.get_deduction(deduction_messages)
            self.update_history()
        
        prompt = self.organize_information() + "\n\n" + self.deduction_result() + "\n\n" + self.action_format()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        action = self.get_action(messages)
        return action

    def add_history(self, new_history):
        # update facts
        if self.role == "Werewolf":
            self.facts[0] = f"you are player_{self.id}, your role is {self.role}, your teammate is player_{self.teammate}."
        else:
            self.facts[0] = f"you are player_{self.id}, your role is {self.role}."
        if self.status == "night":
            self.facts[1] = f"current round and phase: night {self.n_round}."
        elif self.status == "discussion":
            self.facts[1] = f"current round and phase: day {self.n_round} discussion."
        elif self.status == "voting":
            self.facts[1] = f"current round and phase: day {self.n_round} voting."
        players = ", ".join([f"player_{i}" for i, alive in enumerate(self.alive) if alive])
        self.facts[2] = f"remaining players: {players}."
        # add new history
        for n, h in enumerate(new_history):
            n_round = self.n_round - len(new_history) + n + 1
            if "night" in h.keys():
                self.facts.append(f"night {n_round}: {h['night']}")
            if "announcement" in h.keys():
                self.facts.append(f"day {n_round} announcement: {h['announcement']}")
            if "voting" in h.keys():
                self.facts.append(f"day {n_round} voting: {h['voting']}")
            if "discussion" in h.keys():
                for k, v in h["discussion"].items():
                    if k == self.id:  # self discussion
                        self.facts.append(f"in day {n_round} discussion, your said: {v}")
                    elif self.deduction[k]["reliability"] > 6:
                        self.potential_truths.append({"id": k, "statement": f"in day {n_round} discussion, player_{k} said: {v}"})
                    else:
                        self.potential_deceptions.append({"id": k, "statement": f"in day {n_round} discussion, player_{k} said: {v}"})
    
    def organize_information(self):
        n_fact = len(self.facts)
        n_truth = len(self.potential_truths)
        n_deception = len(self.potential_deceptions)
        prompt = "Facts:\n"
        prompt += "\n".join([f"[{i + 1}] {item}" for i, item in enumerate(self.facts)])
        if n_truth > 0:
            prompt += "\n\nPotential truths:\n"
            prompt += "\n".join([f"[{n_fact + i + 1}] {item['statement']}" for i, item in enumerate(self.potential_truths)])
        if n_deception > 0:
            prompt += "\n\nPotential deceptions:\n"
            prompt += "\n".join([f"[{n_fact + n_truth + i + 1}] {item['statement']}" for i, item in enumerate(self.potential_deceptions)])
        return prompt
    
    def deduction_format(self):
        if self.role == "Werewolf":
            other_players = [f"player_{i}" for i, alive in enumerate(self.alive) if alive and (i != self.id) and (i != self.teammate)]
            prompt = (
                f"As player_{self.id} and a {self.role}, "
                "you should reflect on your previous deduction "
                f"and reconsider the hidden roles of {', '.join(other_players)}. "
                "You should provide your reasoning, rate your confidence, "
                "and cite all key information as evidence to support your deduction.\n\n"
                "You should only respond in JSON format as described below.\n"
                "Response Format:\n"
                "{\n"
                "    \"player_i\": {\n"
                "        \"role\": select the most likely hidden role of this player from [\"Werewolf\", \"Seer\", \"Doctor\", \"Villager\", \"Uncertain\"],\n"
                "        \"reasoning\": your reflection and reasoning,\n"
                "        \"confidence\": rate the confidence of your deduction from 5 (pure guess) to 10 (absolutely sure),\n"
                "        \"evidence\": list of integers that cite the key information\n"
                "    }\n"
                "}\n"
                "Ensure the response can be parsed by Python json.loads"
            )
        else:
            other_players = [f"player_{i}" for i, alive in enumerate(self.alive) if alive and (i != self.id)]
            prompt = (
                f"As player_{self.id} and a {self.role}, "
                "you should reflect on your previous deduction "
                f"and reconsider the hidden roles of {', '.join(other_players)}. "
                "Remember there is at least one Werewolf among them. "
                "You should provide your reasoning, rate your confidence, "
                "and cite all key information as evidence to support your deduction.\n\n"
                "You should only respond in JSON format as described below.\n"
                "Response Format:\n"
                "{\n"
                "    \"player_i\": {\n"
                "        \"role\": select the most likely hidden role of this player from [\"Werewolf\", \"Seer\", \"Doctor\", \"Villager\", \"Non-Werewolf\", \"Uncertain\"],\n"
                "        \"reasoning\": your reflection and reasoning,\n"
                "        \"confidence\": rate the confidence of your deduction from 5 (pure guess) to 10 (absolutely sure),\n"
                "        \"evidence\": list of integers that cite the key information\n"
                "    }\n"
                "}\n"
                "Ensure the response can be parsed by Python json.loads"
            )
        return prompt

    def get_deduction(self, messages):
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

                self.evidence = []
                for k, v in result.items():
                    p_idx = k.find("player")
                    idx = int(k[p_idx + 7])
                    self.deduction[idx]["role"] = v["role"]
                    if v["role"] == "Uncertain":
                        self.deduction[idx]["confidence"] = 5
                        self.deduction[idx]["reliability"] = 6
                    elif v["role"] == "Werewolf":
                        self.deduction[idx]["confidence"] = v["confidence"]
                        self.deduction[idx]["reliability"] = 11 - v["confidence"]
                    else:
                        self.deduction[idx]["confidence"] = v["confidence"]
                        self.deduction[idx]["reliability"] = v["confidence"]
                    self.evidence += v["evidence"]

            except openai.error.RateLimitError:
                print("Reach OpenAI API rate limit, sleep for 5s.")
                time.sleep(5)
                print("Try to get response again...\n")
                result = None
            except openai.error.ServiceUnavailableError:
                print("OpenAI server is overloaded or not ready yet, sleep for 5s.")
                time.sleep(5)
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
            except Exception as error:
                print(f"{type(error).__name__}: {error}.")
                print("Try to get response again...\n")
                result = None
                n_error += 1
        if result is not None:
            self.logger.info(f"RESPONSE\n{response.choices[0].message.content}\n")

        return result

    def update_history(self):
        n_fact = len(self.facts)
        evidence = set(self.evidence)
        statements = self.potential_truths + self.potential_deceptions
        cited_statements = [h for i, h in enumerate(statements) if (n_fact + i + 1) in evidence]
        self.potential_truths = []
        self.potential_deceptions = []
        for h in cited_statements:
            if self.deduction[h["id"]]["reliability"] > 6:
                self.potential_truths.append(h)
            else:
                self.potential_deceptions.append(h)

    def deduction_result(self, mode=None):
        if mode == "previous":
            prompt = "Your previous deduction:"
        else:
            prompt = "Your deduction:"
        for k, v in self.deduction.items():
            if not self.alive[k]:
                continue
            elif k == self.id:
                prompt += f"\nplayer_{k}: {self.role}."
            elif self.role == "Werewolf" and k == self.teammate:
                prompt += f"\nplayer_{k}: Werewolf teammate."
            elif self.role == "Werewolf" and v["role"] == "Werewolf":
                prompt += f"\nplayer_{k}: Non-Werewolf."
            elif self.role == "Seer" and k in self.seen_players.keys():
                prompt += f"\nplayer_{k}: confirmed to be a {self.seen_players[k]}."
            elif v["role"] == "Uncertain":
                if self.role == "Werewolf":
                    prompt += f"\nplayer_{k}: Non-Werewolf."
                else:
                    prompt += f"\nplayer_{k}: Uncertain."
            elif v["confidence"] <= 5:
                prompt += f"\nplayer_{k}: Uncertain."
            elif v["confidence"] <= 7:
                prompt += f"\nplayer_{k}: may be a {v['role']}."
            elif v["confidence"] <= 9:
                prompt += f"\nplayer_{k}: likely to be a {v['role']}."
            else:
                prompt += f"\nplayer_{k}: must be a {v['role']}."
        return prompt


class DiverseDeductiveAgent(DeductiveLanguageAgent):
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0.7, log_file=None, n_candidate=3):
        super().__init__(model, temperature, log_file)

        self.n_candidate = n_candidate

    def act(self, obs):
        self.id = obs["id"]
        self.role = obs["role"]
        self.teammate = obs["teammate"]
        self.n_round = obs["n_round"]
        self.status = obs["status"]
        self.alive = obs["alive"]
        self.seen_players = obs["seen_players"]
        
        # deductive reasoning
        self.add_history(obs["new_history"])
        if self.n_round == 1 and self.status == "night":
            pass
        else:
            deduction_prompt = self.organize_information() + "\n\n" + self.deduction_result("previous") + "\n\n" + self.deduction_format()
            deduction_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": deduction_prompt},
            ]
            self.get_deduction(deduction_messages)
            self.update_history()
        
        # generate action candidates
        candidate_prompt = self.organize_information() + "\n\n" + self.deduction_result() + "\n\n" + self.candidate_format()
        candidate_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": candidate_prompt},
        ]
        action_candidates = self.get_candidates(candidate_messages)

        # get action
        prompt = self.organize_information() + "\n\n" + self.deduction_result() + "\n\n" + self.choice_format(action_candidates)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        choice = self.get_choice(messages)
        action = self.parse_action(action_candidates[choice])

        return action

    def candidate_format(self):
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
                        f"You should propose {self.n_candidate} different strategies "
                        "and provide corresponding reasoning and action. "
                        "You should only respond in JSON format as described below.\n"
                        "Response Format:\n"
                        "{\n"
                        "    \"strategy_i\": {\n"
                        "        \"reasoning\": \"reason about the current situation\",\n"
                        "        \"action\": \"kill player_i\"\n"
                        "    }\n"
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
                        f"You should propose {self.n_candidate} different strategies "
                        "and provide corresponding reasoning and action. "
                        "You should only respond in JSON format as described below.\n"
                        "Response Format:\n"
                        "{\n"
                        "    \"strategy_i\": {\n"
                        "        \"reasoning\": \"reason about the current situation\",\n"
                        "        \"action\": \"kill player_i\"\n"
                        "    }\n"
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
                    f"You should propose {self.n_candidate} different strategies "
                    "and provide corresponding reasoning and action. "
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"strategy_i\": {\n"
                    "        \"reasoning\": \"reason about the current situation\",\n"
                    f"        \"action\": \"see player_i\"\n"
                    "    }\n"
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
                    f"You should propose {self.n_candidate} different strategies "
                    "and provide corresponding reasoning and action. "
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"strategy_i\": {\n"
                    "        \"reasoning\": \"reason about the current situation\",\n"
                    f"       \"action\": \"save player_i\"\n"
                    "    }\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
        elif self.status == "discussion":
            prompt = (
                f"Now it is day {self.n_round} dicussion phase and it is your turn to speak. "
                f"As player_{self.id} and a {self.role}, "
                "you should first reason about the current situation only to yourself, "
                "then speak to all other players.\n\n"
                f"You should propose {self.n_candidate} different strategies "
                "and provide corresponding reasoning and discussion. "
                "You should only respond in JSON format as described below.\n"
                "Response Format:\n"
                "{\n"
                "    \"strategy_i\": {\n"
                "        \"reasoning\": \"reason about the current situation only to yourself\",\n"
                "        \"discussion\": \"speak to all other players\"\n"
                "    }\n"
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
                    f"You should propose {self.n_candidate} different strategies "
                    "and provide corresponding reasoning and action. "
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"strategy_i\": {\n"
                    "        \"reasoning\": \"reason about the current situation\",\n"
                    f"        \"action\": \"vote for player_i\"\n"
                    "    }\n"
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
                    f"You should propose {self.n_candidate} different strategies "
                    "and provide corresponding reasoning and action. "
                    "You should only respond in JSON format as described below.\n"
                    "Response Format:\n"
                    "{\n"
                    "    \"strategy_i\": {\n"
                    "        \"reasoning\": \"reason about the current situation\",\n"
                    f"        \"action\": \"vote for player_i\"\n"
                    "    }\n"
                    "}\n"
                    "Ensure the response can be parsed by Python json.loads"
                )
        else:
            raise NotImplementedError
        return prompt

    def get_candidates(self, messages):
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
                assert len(result) == self.n_candidate
                if self.status in ["night", "voting"]:
                    for k in result.keys():
                        assert result[k]["action"] in available_actions
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
                if len(result) != self.n_candidate:
                    print(f"number of action candidates {len(result)} != {self.n_candidate}")
                    self.logger.info(f"RESPONSE\n{response.choices[0].message.content}\n")
                else:
                    for k in result.keys():
                        if result[k]["action"] not in available_actions:
                            print(f"Illegal action: {result[k]['action']}.")
                            print(f"Available actions are: {', '.join(available_actions)}.")
                print("Try to get response again...\n")
                result = None
                n_error += 1
            except Exception as error:
                print(f"{type(error).__name__}: {error}.")
                self.logger.error(f"RESPONSE\n{response.choices[0].message.content}\n")
                print("Try to get response again...\n")
                result = None
                n_error += 1
        if result is not None:
            self.logger.info(f"RESPONSE\n{response.choices[0].message.content}\n")
        self.current_action_candidates=result
        
        return result

    def choice_format(self, action_candidates):
        if self.status == "night":
            prompt = f"Now it is night {self.n_round} round. "
        elif self.status == "discussion":
            prompt = f"Now it is day {self.n_round} dicussion phase. "
        elif self.status == "voting":
            prompt = f"Now it is day {self.n_round} voting phase. "
        prompt += f"As player_{self.id} and a {self.role}, you should choose from the following strategies:\n"
        for k, v in action_candidates.items():
            prompt += f"{k}: {json.dumps(v, indent=4)}\n"
        prompt += (
            "\nYou should reason about which is the best strategy and choose the best one. "
            "You should only respond in JSON format as described below.\n"
            "Response Format:\n"
            "{\n"
            "    \"reasoning\": \"which is the best strategy and why\",\n"
            "    \"choice\": \"strategy_i\"\n"
            "}\n"
            "Ensure the response can be parsed by Python json.loads"
        )
        return prompt

    def get_choice(self, messages):
        self.logger.info(f"PROMPT\n{messages[-1]['content']}\n")

        result = None
        n_error = 0
        while result is None and n_error <= MAX_ERROR:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                result = json.loads(response.choices[0].message.content)
                choice = result["choice"]
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
            except Exception as error:
                print(f"{type(error).__name__}: {error}.")
                print("Try to get response again...\n")
                result = None
                n_error += 1
        if result is not None:
            self.logger.info(f"RESPONSE\n{response.choices[0].message.content}\n")
        return choice


class StrategicLanguageAgent(DiverseDeductiveAgent):
    def __init__(self, model="gpt-3.5-turbo-0613", temperature=0.7, log_file=None, n_candidate=3):
        super().__init__(model, temperature, log_file, n_candidate)
        self.all_roles = np.array(["Werewolf", "Seer", "Doctor", "Villager", "Unknown"])
        self.all_phase = np.array(["night", "discussion", "voting"])

        self.round_info = [  # update in game.
            {
                "secret_action": np.zeros(7),
                "announcement": np.zeros(7),
                "voting": np.zeros(49),
            } for _ in range(3)
        ]
        self.deduction = None
        """Example deduction:
        self.deduction = {
            "player_0": {"role": "Seer", "confidence": 10},
            "player_1": {"role": "Unknown", "confidence": 6},
            "player_2": {"role": "Unknown", "confidence": 6},
            "player_3": {"role": "Unknown", "confidence": 6},
            "player_4": {"role": "Unknown", "confidence": 6},
            "player_5": {"role": "Werewolf", "confidence": 9},
            "player_6": {"role": "Unknown", "confidence": 6},
        }
        """

    def player_embedding(self):
        player_id = np.eye(7)[self.id]
        player_role = np.eye(4)[np.where(self.all_roles == self.role)[0][0]]
        n_round = np.array([self.n_round])
        phase = np.eye(3)[np.where(self.all_phase == self.status)[0][0]]
        alive = np.array(self.alive)

        round_embedding = []
        for info_dict in self.round_info:
            for v in info_dict.values():
                round_embedding.append(v)
        round_embedding = np.concatenate(round_embedding)

        deduction_embedding = []
        for i in range(7):
            if i == self.id:
                role = np.eye(5)[np.where(self.all_roles == self.role)[0][0]]
                confidence = np.array([10])
            else:
                role = np.eye(5)[np.where(self.all_roles == self.deduction[f"player_{i}"]["role"])[0][0]]
                confidence = np.array([self.deduction[f"player_{i}"]["confidence"]])
            deduction_embedding.extend([role, confidence])
        deduction_embedding = np.concatenate(deduction_embedding)

        player_embedding = np.concatenate([
            player_id,
            player_role,
            n_round,
            phase,
            alive,
            round_embedding,
            deduction_embedding,
        ])
        return player_embedding
    def get_language_observation_embedding(self):
        return get_embedding(self.language_observation)
    def get_language_action_embedding(self):
        action_embeddings=[]
        for action in self.current_action_candidates:
            action_s= action['reasoning']+ ' ' + action['action']
            e_action=get_embedding(action_s)
            action_embeddings.append(e_action)
        return action_embeddings



