from copy import deepcopy
import numpy as np


class WerewolfEnvironment(object):

    def __init__(self):
        self.n_player = 7
        self.rewards=np.zeros(self.n_player)
        self.actually_killed = None
        self.n_werewolf = 2
        self.n_seer = 1
        self.n_doctor = 1
        self.n_villager = 3
        self.voting_encoding = np.zeros((self.n_player, self.n_player), dtype=int)
        self.roles = np.array(
            ["Werewolf" for _ in range(self.n_werewolf)] +
            ["Seer" for _ in range(self.n_seer)] +
            ["Doctor" for _ in range(self.n_doctor)] +
            ["Villager" for _ in range(self.n_villager)]
        )

    def reset(self):
        """
        Returns:
            obs: List[Union[None, List[Dict]]]
                list of observations, length is n_player.
                obs[i] is None if player[i] does not act in the next step.
                otherwise obs[i] is a list of dict.

                e.g., obs[0] = {
                    "id": 0,
                    "role": "Seer",
                    "teammate": None,
                    "n_round": 2,
                    "status": "night" / "discussion" / "voting",
                    "alive": [True, False, False, True, True, True, True],
                    "history": [
                        {
                            "night": "you saw player_1 is a Werewolf.",
                            "announcement": "player_2 was killed last night.",
                            "dicussion": {
                                0: "I'm a Villager ...",
                                ...,
                                6: "I agree with ...",
                            },
                            "voting": "player_1 had the most votes and was eliminated...",
                        },
                        ...
                    ],
                }
        """
        # assign roles
        self.roles = np.random.permutation(self.roles)
        self.alive = np.ones(self.n_player, dtype=bool)

        # reset game
        self.round = "night"
        self.phase = "all"
        self.n_round = 1
        self.kill_target = None
        self.see_target = None
        self.save_target = None
        self.seen_players = {}

        # init history and info
        self.histories = [[{}] for _ in range(self.n_player)]
        self.new_histories = [[{}] for _ in range(self.n_player)]
        self.info = {
            "roles": list(self.roles),
            "history": [{"night": [], "announcement": None, "discussion": {}, "voting": None}],
        }
        return self.generate_obs()

    def step(self, actions):
        """
        Args:
            actions: List[Union[None, int, str]]
                list of actions, length is n_player.
                actions[i] is None if player[i] does not act in the this step.
                actions[i] is an int if it is a night action or a voting action.
                actions[i] is a string if it is a statement action in the discussion.
        Returns:
            obs: List[Union[None, List[Dict]]]
                list of observations, length is n_player.
                obs[i] is None if player[i] does not act in the next step.
                otherwise obs[i] is a list of dict.
            rewards: np.array
            dones: np.array
            info: Dict
        """
        if self.round == "night":
            # get actions
            for role, action in zip(self.roles, actions):
                if action is None:
                    continue
                if role == "Werewolf":
                    self.kill_target = action
                elif role == "Seer":
                    self.see_target = action
                elif role == "Doctor":
                    self.save_target = action

            if self.phase == "all" and np.sum(self.roles[self.alive] == "Werewolf") > 1:
                # continue to get the next werewolf's action if more than one werewolf left
                self.phase = "next"
            else:
                # add actions to history and info
                for idx in range(self.n_player):
                    if (not self.alive[idx]) or (self.roles[idx] == "Villager"):
                        continue
                    if self.roles[idx] == "Werewolf":
                        if np.sum(self.roles[self.alive] == "Werewolf") > 1:
                            secret_action = f"you and your teamamte chose to kill player_{self.kill_target}."
                        else:
                            secret_action = f"you chose to kill player_{self.kill_target}."
                        night_info = f"player_{idx} chose to kill player_{self.kill_target}."
                    elif self.roles[idx] == "Seer":
                        if self.roles[self.see_target] == "Werewolf":
                            self.rewards[self.roles == "Seer"] += 2
                            self.seen_players[self.see_target] = "Werewolf"
                            secret_action = f"you saw player_{self.see_target} is a Werewolf."
                            night_info = f"player_{idx} saw player_{self.see_target} is a Werewolf."
                        else:
                            self.seen_players[self.see_target] = "Non-Werewolf"
                            self.rewards[self.roles == "Seer"] -= 2
                            secret_action = f"you saw player_{self.see_target} is not a Werewolf."
                            night_info = f"player_{idx} saw player_{self.see_target} is not a Werewolf."
                    elif self.roles[idx] == "Doctor":
                        secret_action = f"you chose to save player_{self.save_target}."
                        night_info = f"player_{idx} chose to save player_{self.save_target}."
                    self.histories[idx][-1]["night"] = secret_action
                    self.new_histories[idx][-1]["night"] = secret_action
                    self.info["history"][-1]["night"].append(night_info)

                # apply actions and add announcement to histories
                if self.kill_target != self.save_target:
                    #reward for successfully killing
                    self.rewards[self.roles == "Werewolf"] += 5

                    self.alive[self.kill_target] = False
                    self.actually_killed = self.kill_target
                    announcement = f"player_{self.kill_target} was killed last night."
                    # reward for not successfully saving
                    self.rewards[self.roles == "Doctor"] -= 5
                else:
                    # reward for not successfully killing
                    self.rewards[self.roles == "Werewolf"] -= 5
                    # reward for successfully saving
                    self.rewards[self.roles == "Doctor"] +=5
                    announcement = f"no player was killed last night."
                for idx in range(self.n_player):
                    if self.alive[idx] or (idx == self.kill_target):
                        self.histories[idx][-1]["announcement"] = announcement
                        self.new_histories[idx][-1]["announcement"] = announcement
                self.info["history"][-1]["announcement"] = announcement

                # continue to the day round discussison phase
                self.round = "day"
                self.phase = "discussion"
                self.speaker = np.argmax(self.alive)
                for idx in range(self.n_player):
                    if self.alive[idx]:
                        self.histories[idx][-1]["discussion"] = {}
                        self.new_histories[idx][-1]["discussion"] = {}

        elif self.round == "day" and self.phase == "discussion":
            discussion = actions[self.speaker]
            # info_discussion = f"player_{self.speaker} ({self.roles[self.speaker]}) said: \"{actions[self.speaker]}\""
            for idx in range(self.n_player):
                if self.alive[idx]:
                    self.histories[idx][-1]["discussion"][self.speaker] = discussion
                    self.new_histories[idx][-1]["discussion"][self.speaker] = discussion
            self.info["history"][-1]["discussion"][int(self.speaker)] = discussion

            # check if discussion end
            n_left = np.sum(self.alive[self.speaker + 1:])
            if n_left > 0:
                self.speaker += np.argmax(self.alive[self.speaker + 1:]) + 1
            else:
                self.phase = "voting"

        elif self.round == "day" and self.phase == "voting":
            # count votes: i means vote for player_i, -1 means do not vote
            cnt_vote = np.zeros(self.n_player + 1)
            voter_list = [[] for _ in range(self.n_player + 1)]
            for idx, action in enumerate(actions):
                if action is None:
                    continue
                #Individual voting rewards
                if self.roles[action] == "Werewolf" and self.roles[idx] != "Werewolf":
                    self.rewards[idx] += 1
                    self.rewards[action] -= 1
                if self.roles[action] != "Werewolf" and self.roles[idx] != "Werewolf":
                    self.rewards[idx] -= 1
                    self.rewards[self.roles == "Werewolf"] += 1


                if self.alive[idx]:
                    self.voting_encoding[idx, action] = 1
                cnt_vote[action] += 1
                voter_list[action].append(f"player_{idx}")

            # get voting result
            if np.sum(cnt_vote[:-1]) == 0:
                # all players choose not to vote
                voting_result = f"no player was eliminated."

            else:
                voting_target = np.random.choice(np.where(cnt_vote[:-1] == np.max(cnt_vote[:-1]))[0])
                self.alive[voting_target] = False
                voting_result = f"player_{voting_target} had the most votes and was eliminated."
                if self.roles[self.see_target] == "Werewolf":
                    self.rewards[self.roles == "Werewolf"] -= 5
                    self.rewards[self.roles != "Werewolf"] += 5
                elif  self.roles[self.see_target] != "Werewolf":
                    self.rewards[self.roles == "Werewolf"] += 5
                    self.rewards[self.roles != "Werewolf"] -= 5


            # add voting to obs and info
            for idx in range(self.n_player):
                if cnt_vote[idx] > 0:
                    voters = ", ".join(voter_list[idx])
                    voting_result += f" {voters} voted for player_{idx}."
            if cnt_vote[-1] > 0:
                voters = ", ".join(voter_list[-1])
                voting_result += f" {voters} chose not to vote."
            for idx in range(self.n_player):
                if self.alive[idx]:
                    self.histories[idx][-1]["voting"] = voting_result
                    self.new_histories[idx][-1]["voting"] = voting_result
            self.info["history"][-1]["voting"] = voting_result
            # Exporting Player vectors
            self.generate_player_vector()

            # continue to the next night round

            self.round = "night"
            self.phase = "all"
            self.n_round += 1
            self.kill_target = None
            self.see_target = None
            self.save_target = None

            for idx in range(self.n_player):
                if self.alive[idx]:
                    self.histories[idx].append({})
                    self.new_histories[idx].append({})
            self.info["history"].append({"night": [], "announcement": None, "discussion": {}, "voting": None})

        # check if game is over
        alive_werewolf = np.sum(self.roles[self.alive] == "Werewolf")
        alive_villager = np.sum(self.roles[self.alive] != "Werewolf")
        if alive_werewolf > alive_villager:
            # the Werewolves win the game
            self.rewards[self.roles == "Werewolf"] += 100
            self.rewards[self.roles != "Werewolf"] -= 100
            dones = np.ones(self.n_player, dtype=bool)
            self.info["winners"] = "the Werewolves"
        elif alive_werewolf == 0:
            # the Villagers win the game
            self.rewards[self.roles != "Werewolf"] += 100
            self.rewards[self.roles == "Werewolf"] -= 100
            dones = np.ones(self.n_player, dtype=bool)
            self.info["winners"] = "the Villagers"
        elif self.n_round >= 10:
            # randomly choose the winner
            self.info["winners"] = np.random.choice(["the Werewolves", "the Villagers"])
            if self.info["winners"] == "the Werewolves":
                self.rewards[self.roles == "Werewolf"] += 100
                self.rewards[self.roles != "Werewolf"] -= 100
                dones = np.ones(self.n_player, dtype=bool)
            else:
                self.rewards[self.roles != "Werewolf"] += 100
                self.rewards[self.roles == "Werewolf"] -= 100
                dones = np.ones(self.n_player, dtype=bool)
        else:

            dones = (self.alive == False)

        return self.generate_obs(), self.rewards, dones, self.generate_info()

    def generate_obs(self):
        obs = [None for _ in range(self.n_player)]
        all_werewolves = np.where(self.roles == "Werewolf")[0]
        if self.round == "night" and self.phase == "all":
            # first alive Werewolf, Seer, and Doctor
            werewolf_idx = np.argmax(self.alive * (self.roles == "Werewolf"))
            for idx in range(self.n_player):
                if (not self.alive[idx]) or (self.roles[idx] == "Villager"):
                    continue
                if (self.roles[idx] == "Werewolf") and (idx != werewolf_idx):
                    continue
                teammate = all_werewolves[all_werewolves != werewolf_idx][0] if self.roles[idx] == "Werewolf" else None
                seen_players = deepcopy(self.seen_players) if self.roles[idx] == "Seer" else None
                obs[idx] = {
                    "id": idx,
                    "role": self.roles[idx],
                    "roles": deepcopy(self.roles),
                    "teammate": teammate,
                    "n_round": self.n_round,
                    "status": "night",
                    "alive": deepcopy(self.alive),
                    "history": deepcopy(self.histories[idx]),
                    "new_history": deepcopy(self.new_histories[idx]),
                    "seen_players": seen_players,
                }
                self.new_histories[idx] = [{}]
        elif self.round == "night" and self.phase == "next":
            # next Werewolf
            idx = np.where(self.roles == "Werewolf")[0][1]
            teammate = all_werewolves[all_werewolves != idx][0]
            history = deepcopy(self.histories[idx])
            history[-1]["night"] = f"your teammate proposed to kill player_{self.kill_target}."
            new_history = deepcopy(self.new_histories[idx])
            new_history[-1]["night"] = f"your teammate proposed to kill player_{self.kill_target}."
            obs[idx] = {
                "id": idx,
                "role": self.roles[idx],
                "roles": deepcopy(self.roles),
                "teammate": teammate,
                "n_round": self.n_round,
                "status": "night",
                "alive": deepcopy(self.alive),
                "history": history,
                "new_history": new_history,
                "seen_players": None,
            }
            self.new_histories[idx] = [{}]
        elif self.round == "day" and self.phase == "discussion":
            # current speaker
            idx = self.speaker
            teammate = all_werewolves[all_werewolves != idx][0] if self.roles[idx] == "Werewolf" else None
            seen_players = deepcopy(self.seen_players) if self.roles[idx] == "Seer" else None
            obs[idx] = {
                "id": idx,
                "role": self.roles[idx],
                "roles": deepcopy(self.roles),
                "teammate": teammate,
                "n_round": self.n_round,
                "status": "discussion",
                "alive": deepcopy(self.alive),
                "history": deepcopy(self.histories[idx]),
                "new_history": deepcopy(self.new_histories[idx]),
                "seen_players": seen_players,
            }
            self.new_histories[idx] = [{"discussion": {}}]
        elif self.round == "day" and self.phase == "voting":
            # alive players
            for idx in range(self.n_player):
                if not self.alive[idx]:
                    continue
                teammate = all_werewolves[all_werewolves != idx][0] if self.roles[idx] == "Werewolf" else None
                seen_players = deepcopy(self.seen_players) if self.roles[idx] == "Seer" else None
                obs[idx] = {
                    "id": idx,
                    "role": self.roles[idx],
                    "roles": deepcopy(self.roles),
                    "teammate": teammate,
                    "n_round": self.n_round,
                    "status": "voting",
                    "alive": deepcopy(self.alive),
                    "history": deepcopy(self.histories[idx]),
                    "new_history": deepcopy(self.new_histories[idx]),
                    "seen_players": seen_players,
                }
                self.new_histories[idx] = [{}]
        else:
            raise NotImplementedError

        return obs

    def generate_info(self):
        info = deepcopy(self.info)
        info["n_round"] = self.n_round
        if self.round == "night":
            info["status"] = "night"
        elif self.round == "day" and self.phase == "discussion":
            info["status"] = "discussion"
        elif self.round == "day" and self.phase == "voting":
            info["status"] = "voting"
        return info

    def generate_player_vector(self):
        """
        Generate the player vector according to the specifications in the table.

        Returns:
            player_vectors: List[Dict]
                List containing a vector for each player, as described in the table.
        """
        player_vectors = []

        # One-hot encode roles and phase
        role_dict = {"Werewolf": 0, "Seer": 1, "Doctor": 2, "Villager": 3}
        phase_dict = {"night": 0, "discussion": 1, "voting": 2}

        # Loop through each player and generate their corresponding vector
        for player_id in range(self.n_player):
            secret_action = np.zeros(7)
            if self.roles[player_id] == 'Werewolf':
                secret_action = np.eye(self.n_player)[self.kill_target].tolist()
            elif self.roles[player_id] == 'Seer':
                secret_action = np.eye(self.n_player)[self.see_target].tolist()
            elif self.roles[player_id] == 'Doctor':
                secret_action = np.eye(self.n_player)[self.save_target].tolist()
            if self.actually_killed is not None:
                announ_vector = np.eye(self.n_player)[self.actually_killed].tolist()
            else:
                # Handle the None case, e.g., return a zero vector or a specific 'null' vector
                announ_vector = [0] * self.n_player  # A zero vector of the same length

            player_vector = {
                "ID": np.eye(self.n_player)[player_id].tolist(),
                "Role": np.eye(4)[role_dict[self.roles[player_id]]].tolist(),
                "Round": [self.n_round],
                "Phase": np.eye(3)[phase_dict[self.phase]].tolist(),
                "Alive Players": self.alive.tolist(),
                "Secret Action": secret_action,
                "announcement": announ_vector,
                "voting_action": self.voting_encoding.flatten(),

            }
            # Add the generated vector to the list of player vectors
            # print("Players took this action: ")
            # print(player_vector['voting_action'])
            # print('----------------------------------------------------------------------------------------------------------------------------------------------')
            player_vectors.append(player_vector)
        # print(player_vectors)
        return player_vectors
    def get_round_info(self, player_id):
        # One-hot encode roles and phase
        role_dict = {"Werewolf": 0, "Seer": 1, "Doctor": 2, "Villager": 3}
        phase_dict = {"night": 0, "discussion": 1, "voting": 2}
        secret_action = np.zeros(7)
        if self.roles[player_id] == 'Werewolf':
            secret_action = np.eye(self.n_player)[self.kill_target].tolist()
        elif self.roles[player_id] == 'Seer':
            secret_action = np.eye(self.n_player)[self.see_target].tolist()
        elif self.roles[player_id] == 'Doctor':
            secret_action = np.eye(self.n_player)[self.save_target].tolist()
        if self.actually_killed is not None:
            announ_vector = np.eye(self.n_player)[self.actually_killed].tolist()
        else:
            # Handle the None case, e.g., return a zero vector or a specific 'null' vector
            announ_vector = [0] * self.n_player  # A zero vector of the same length

        round_info = {
            "secret_action": secret_action,
            "announcement": announ_vector,
            "voting": self.voting_encoding.flatten(),
        }
        return round_info


