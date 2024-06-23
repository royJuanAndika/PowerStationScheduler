import random
from tes2 import *

def generateRandomChromosome():
    unit = [20,15,35,40,15,15,10]
    interval = [2,2,1,1,1,2,1]

    capacity = [150, 150, 150, 150, 150, 150, 150]
    arr = [[0 for j in range(6)] for i in range(2)]

    for i in range(2):
        for j in range(6):
            num = random.randint(1, 7)
            print(j, capacity[j], unit[num-1])
            if num != 0 and capacity[j] >= unit[num-1]:
                interval[num-1] -= 1
                capacity[j] -= unit[num-1]
                arr[i][j] = num
            else:
                arr[i][j] = 0

    # Transpose the list of lists
    transposed_arr = [[arr[j][i] for j in range(2)] for i in range(6)]
    return transposed_arr


arr = generateRandomChromosome()
print(arr)

jancok()