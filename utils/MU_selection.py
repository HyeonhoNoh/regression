import itertools
import torch

def Q_table_to_action(Q_table, user_group, game_parameters):
    if len(Q_table[1][0]) == game_parameters['n_actions'][0]:
        max_idx = 0
        while True:
            Q_idx = Q_table[1][0][max_idx]
            if Q_idx in user_group:
                return [Q_idx.item()]
            else:
                max_idx += 1
    elif len(Q_table[1][0]) == game_parameters['n_actions'][1]:
        user_group_ = [str(i) for i in range(10)]
        user_list = []
        for i in range(4):
            user_list.append(list(map(''.join, itertools.combinations(user_group_, i+1))))
        user_list = sum(user_list,[])
        max_idx = 0
        while True:
            Q_idx = Q_table[1][0][max_idx]
            selected_user = [int(i) for i in user_list[Q_idx]]
            if set(selected_user).issubset(set(user_group)):
                return selected_user
            else:
                max_idx += 1

def action_to_user(action):
    user_group_ = [str(i) for i in range(10)]
    user_list = []
    for i in range(4):
        user_list.append(list(map(''.join, itertools.combinations(user_group_, i + 1))))
    user_list = sum(user_list, [])
    selected_user = [int(i) for i in user_list[action]]
    return selected_user

def user_to_action(user):
    user_group_ = [str(i) for i in range(10)]
    user_list = []
    for i in range(4):
        user_list.append(list(map(''.join, itertools.combinations(user_group_, i + 1))))
    user_list = sum(user_list, [])
    user_str_list = [str(i.item()) for i in user]
    user_str = "".join(user_str_list)

    user_idx = user_list.index(user_str)
    return torch.tensor([[user_idx]], dtype=torch.int)

