from collections import deque
import copy
def map_list_combination(params_list):
    res = deque([{}])
    for key in params_list:
        value_list = params_list[key]
        l = len(res)
        for i in range(l):
            cur_dict = res.popleft()
            for value in value_list:
                new_cur_dict = copy.deepcopy(cur_dict)
                new_cur_dict[key] = value
                res.insert(-1, (dict)(new_cur_dict))
    return res




