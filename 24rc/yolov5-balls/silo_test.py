# Author: Ethan Lee
# 2024/6/11 下午5:33
import numpy as np

if __name__ == '__main__':
    # 假设 silo_cnt, silo_my 是你的输入列表，它们都是长度为5的一维列表
    silo_cnt = np.array([1, 3, 3, 2, 2])
    silo_my = np.array([0, 1, 2, 1, 2])

    # 创建一个5x3的全零数组 silo_state
    silo_state = np.zeros((5, 3), dtype=int)

    # 遍历 silo_my，将 silo_state 中的对应列的前n行赋值为1
    for i in range(len(silo_my)):
        silo_state[i, :silo_my[i]] = 1  # 使用与silo_my[i]长度相同的切片进行赋值

    # 遍历 silo_cnt，如果 silo_cnt 中的值大于 silo_my 中的对应值，
    # 那么在 silo_state 中的对应列的下一行赋值为-1
    for i in range(len(silo_cnt)):
        if silo_cnt[i] > silo_my[i]:
            silo_state[i, silo_my[i]:silo_cnt[i]] = -1

    print(silo_state)
