# Author: Ethan Lee
# 2024/5/2 18:22
import os
import pickle
import numpy as np


def loadPolicy(speed: int = 8, success_rate: float = 1.0, verbose: bool = False) -> dict:
    game_dictionary = {}
    profile = None
    for filename in os.listdir('./rl_models'):
        if verbose:
            print("Reading {}... (Extracted as speed {}, success rate {})".format(filename, speed, success_rate))
        fr = open('./rl_models/{}'.format(filename), 'rb')
        profile = 'S{}_R{:.1f}'.format(int(speed), success_rate)
        game_dictionary[profile] = pickle.load(fr)
        fr.close()

    return game_dictionary[profile]


def getAvailableActions(silo: list) -> list:
    positions = list()
    for i in range(5):
        if silo[i][2] is None:
            positions.append(i)
    return positions + [5]


def decide_action(observation: list, policy: dict, ava_actions: list) -> int:
    # 整理球筐队伍 'm'=我方(me), 'e'=敌方(enemy)
    for i in range(5):
        for j in range(3):
            if observation[i][j] is None:
                observation[i][j] = 0
            elif observation[i][j] == 'm':
                observation[i][j] = 1
            elif observation[i][j] == 'e':
                observation[i][j] = -1
    state = str(observation)
    try:
        action = np.argmax(policy[state][:])
    except KeyError:
        print("Random Choice!!!")
        action = np.random.choice(ava_actions)  # 如果没有训练到则随机选择，但不能选择不可选的情况

    if action == 5:
        action = 9

    return action


def isEndGame(silo: list) -> str or None:

    is_full = True
    need_detail_check_team = None

    # 整理球筐队伍 'm'=我方(me), 'e'=敌方(enemy)
    for i in range(5):
        for j in range(3):
            if silo[i][j] == 0:
                silo[i][j] = None
            elif silo[i][j] == 1:
                silo[i][j] = 'm'
            elif silo[i][j] == -1:
                silo[i][j] = 'e'

    # Check for the top rice
    top_count = {}
    for i in range(5):
        if silo[i][2] not in top_count:
            top_count[silo[i][2]] = 0
        top_count[silo[i][2]] += 1

        # Top own by same team for 3 rice
        if top_count[silo[i][2]] >= 3:
            need_detail_check_team = silo[i][2]

        if silo[i][2] is None:
            is_full = False

    # There is no further
    if need_detail_check_team is None:
        return 'f' if is_full is True else None

    # Check if underneath is remained by the same color
    success_col_count = 0
    for i in range(5):
        if silo[i][2] != need_detail_check_team:
            continue
        # Success for one column and break
        for j in range(2):
            if silo[i][j] == need_detail_check_team:
                success_col_count += 1
                break

        if success_col_count >= 3:
            return need_detail_check_team
    return 'f' if is_full is True else None


if __name__ == '__main__':
    model = loadPolicy()
    obs = [[-1, 1, 1], [-1, -1, -1], [1, 1, 1], [-1, -1, 1], [1, -1, 1]]  # E.g.
    isEnd = isEndGame(obs)
    if isEnd is None:
        ava_acts = getAvailableActions(obs)
        decision = decide_action(obs, model, ava_acts)
        print(decision)
    elif isEnd == 'f':
        print('谷仓已满，未大胜')
    elif isEnd == 'm':
        print('大胜！')
        print('播放结算动画！')
    elif isEnd == 'e':
        print('DAMN!!!敌方大胜')
