from ast import List
import matplotlib.pyplot as plt
import copy
import random
from collections import Counter
from itertools import chain
import pandas as pd
from pyscript import document
# import pyarrow
# -------------------------------------------------Needed for Web----------------------------------------------------------------
def accessWebTable():
    webTable = []
    webTable.append([0,0])
    table = document.querySelector("#myTable")
    for i in range(0, table.rows.length):
        # skip the header
        if i == 0:
            continue
        
        row = []
        for j in range(1, table.rows[i].cells.length):
            input_element = table.rows[i].cells[j].querySelector("input")
            value = int(input_element.value)
            
            # value = input.value
            row.append(value)
            # table.rows[i].cells[j].innerHTML = "Hello"
        webTable.append(row)
    return webTable
def greet(event):
    # accessWebTable()
    run(accessWebTable())  
# --------------------------------------------------------------------------------------------------------------------------------


class Listrik:
    def __init__(self, tabelPembangkitListrik, periode, minListrik):
        self.TabelPembangkitListrik = tabelPembangkitListrik
        self.periode = periode
        self.minListrik = minListrik
        self.initListrik = self.findInitListrik()
        self.maxValue, self.maxIndex = self.findMaxAndIndex()
        
        
    def findInitListrik(self):
        print(self.TabelPembangkitListrik[0])
        return sum([row[0] for row in self.TabelPembangkitListrik])
    
    def findMaxAndIndex(self):
        max_val = max(self.TabelPembangkitListrik, key=lambda x: x[1])
        max_index = self.TabelPembangkitListrik.index(max_val)
        return max_val[1], max_index
    

    def getMaxValue(self):
        return self.maxValue

    def getMaxIndex(self):
        return self.maxIndex

    def getTabelPembangkitListrik(self):
        return self.TabelPembangkitListrik

    def setTabelPembangkitListrik(self, tabelPembangkitListrik):
        self.TabelPembangkitListrik = tabelPembangkitListrik

    def getPeriode(self):
        return self.periode

    def setPeriode(self, periode):
        self.periode = periode

    def getMinListrik(self):
        return self.minListrik

    def setMinListrik(self, minListrik):
        self.minListrik = minListrik

    def getInitListrik(self):
        return self.initListrik

    def setInitListrik(self, initListrik):
        self.initListrik = initListrik





# Valid Chromosome
def generateRandomChromosome(listrik_instance):
    unit = [row[0] for row in listrik_instance.getTabelPembangkitListrik()]
    interval = [row[1] for row in listrik_instance.getTabelPembangkitListrik()]
    capacity = [listrik_instance.getInitListrik()] * listrik_instance.getPeriode()
    arr = [[0 for j in range(6)] for i in range(2)]

    for i in range(2):
        for j in range(listrik_instance.getPeriode()):
            num = random.randint(1, listrik_instance.getPeriode()+1)
            if num != 0 and capacity[j] >= unit[num - 1]:
                interval[num - 1] -= 1
                capacity[j] -= unit[num - 1]
                arr[i][j] = num
            else:
                arr[i][j] = 0

    # Transpose the list of lists
    transposed_arr = [[arr[j][i] for j in range(2)] for i in range(6)]
    return transposed_arr


# array = [[0 for j in range(6)] for i in range(2)]
# array = generateRandomChromosome(test1)


# Random Chromosome
def check(interval):
    # Return True if all intervals are not zero
    for i in range(len(interval)):
        if interval[i] == 0:
            return False
    return True


# FOCUS
def generateChromosome(listrik_instance): #INCOMPLETE. DON'T USE
    unit = [row[0] for row in listrik_instance.getTabelPembangkitListrik()]
    interval = [row[1] for row in listrik_instance.getTabelPembangkitListrik()]
    capacity = [listrik_instance.getInitListrik()] * listrik_instance.getPeriode()
    arr = [[0 for j in range(6)] for i in range(2)]
    maxval = False

    for i in range(2):
        for j in range(listrik_instance.getPeriode()):
            num = random.randint(1, listrik_instance.getPeriode()+1)
            num = random.randint(1, 7)
            if num == 4: #Ubah jadi .getMaxIndex()(?) tp hasil returnnya ga cocok
                maxval = True 
            if maxval == True and i != 0 and num == 4:
                num = random.randint(1, 7)
            while interval[num - 1] == 0:  # Check if the selected interval is zero
                if not check(interval):  # If all intervals are zero, break the loop
                    num = 0
                    break
            if i > 0 and arr[i - 1][j] == 4:
                num = 0
            if num != 0 and capacity[j] >= unit[num - 1]:
                interval[num - 1] -= 1
                capacity[j] -= unit[num - 1]  # Move this line up
                arr[i][j] = num
            else:
                arr[i][j] = 0
            if (
                j > 0 and arr[i - 1][j] == 4
            ):  # If the previous index's corresponding subindex = 4, current subindex can't be used for other number
                arr[i][j] = 0

    # Transpose the list of lists
    transposed_arr = [[arr[j][i] for j in range(2)] for i in range(6)]
    return transposed_arr

    return arr


# Elite Chromosome
def fillElite(population, currentPopulation, fitnessScorePopulation):
    largestIdx, secondLargestIdx = findTwoLargestIndex(fitnessScorePopulation)

    currentPopulation.append(population[largestIdx])
    currentPopulation.append(population[secondLargestIdx])

    # DEBUG
    #   print("\n\n")
    #   print(fitnessScorePopulation[largestIdx])
    #   print(fitnessScorePopulation[secondLargestIdx])
    #   print("\n\n")
    return currentPopulation


# Elite Chromosome determinor
def takeBestChromosome(currentPopulation, listOfFitness):
    if not listOfFitness or all(v is None for v in listOfFitness):
        return None, None

    maxIndexOfFitness = max(listOfFitness)
    # return valuenya, bukan index.
    # DEBUG
    # print("max index of fitness",maxIndexOfFitness)
    for i in range(0, len(listOfFitness) - 1):
        if listOfFitness[i] == maxIndexOfFitness:
            return currentPopulation[i], listOfFitness[i]


# Terminator
def terminate(bestChromosomeFitness, iterationCount):
    if iterationCount == 500:
        return True
    elif bestChromosomeFitness == 100:
        return True
    else:
        return False


# Fitness Calculator
def calculateFitnessScore(population, tablePembangkitListrik, listrik_instance):
    fitnessScore = []
    for i in range(len(population)):
        fitnessScore.append(100)

    # for each chromosome
    # print("POPULATION SIZE=",len(population))
    for i in range(len(population)):
        tempTablePembangkitListrik = copy.deepcopy(
            listrik_instance.getTabelPembangkitListrik()
        )
        # print("\ntable original = ", tablePembangkitListrik, "\n")
        # print("\n tabel awal iterasi ke-",i, " = ", tempTablePembangkitListrik, "\n")
        # print("\n CHROMOSOME", i, "\n")
        indexChromosome = i
        adaDuplicate = False
        adaKekuranganMw = False

        # for each bulan
        # print("fitnessAwal =", fitnessScore)
        for j in range(len(population[i])):
            sisaMegaWatt = listrik_instance.getInitListrik()
            ada4 = False
            if (
                hasDuplicates(population[i][j])
                and population[i][j][0] != 0
                and population[i][j][1] != 0
            ):
                # boolean ini dipake untuk pengecekkan selanjutnya. kalo ada duplicate, score langsung 1.
                # adaDuplicate = True
                fitnessScore[indexChromosome] -= 6

            # print("fitness iterasi j ke-", j , " duplicates = ", fitnessScore)

            # for each pembangkit listrik
            for k in range(len(population[i][j])):
                # jika di period itu ada 4, ada4 = true

                # print("tempTablePembangkitListrik[population[i][j][k]][0] =", population[i][j][k])
                # cek interval
                tempTablePembangkitListrik[population[i][j][k]][1] -= 1
                # hitung sisa megawatt
                # print("\n", tempTablePembangkitListrik[population[i][j][k]][0], "\n")
                sisaMegaWatt -= tempTablePembangkitListrik[population[i][j][k]][0]

            # ---------------------HITUNG SISA MEGAWATT-----------------
            # print("\n sisa megaWatt \n",sisaMegaWatt)
            # print(ada4)
            if sisaMegaWatt < listrik_instance.getMinListrik() and ada4 == True:
                fitnessScore[indexChromosome] -= 24
                # adaKekuranganMw = True
                # print("fitness iterasi j ke-", j, " after sisaMW = ", fitnessScore)
                # print("fitness iterasi j ke-", j, " after sisaMW = ", fitnessScore)
            # else :
            #   fitnessScore[indexChromosome] += 0
            # print("fitness iterasi j ke-", j, " after sisaMW = ", fitnessScore)

            # print("\n")
            # ----------------------------------------------------------

        # ---------------------HITUNG INTERVAL----------------------
        # print(tempTablePembangkitListrik)
        for i in range(1, len(tempTablePembangkitListrik)):
            if tempTablePembangkitListrik[i][1] != 0:
                fitnessScore[indexChromosome] -= 2.5
            # elif(tempTablePembangkitListrik[i][1] == 0):
            #   fitnessScore[indexChromosome] += 14
            # print("\n", tempTablePembangkitListrik, "\n")
            # print("fitness iterasi i ke-", i, " after interval =  ", fitnessScore)
        # -------------------------------------------------------

        # if(adaDuplicate):
        #   fitnessScore[indexChromosome] = 1

        # if(adaKekuranganMw):
        #   fitnessScore[indexChromosome] = 1

    return fitnessScore


# Two Largest Index
def findTwoLargestIndex(arr):
    largest_idx, second_largest_idx = None, None
    largest, second_largest = float("-inf"), float("-inf")

    for i, num in enumerate(arr):
        if num > largest:
            second_largest = largest
            second_largest_idx = largest_idx
            largest = num
            largest_idx = i
        elif num > second_largest:
            second_largest = num
            second_largest_idx = i

    # print("largest: ", largest_idx)

    # print("second largest: ", second_largest_idx)

    return largest_idx, second_largest_idx


# Mutation
def mutate(chromosome, mutation_rate):
    if random.random() < mutation_rate:
        for _ in range(1):
            tempChromosome = [list(period) for period in chromosome]
            list_1d = list(chain.from_iterable(tempChromosome))
            element_counts = Counter({i: 0 for i in range(0, 8)})
            # print(element_counts)

            element_counts.update(list_1d)
            # element_counts = Counter(list_1d)

            # print(element_counts)

            mutation_index_period = None
            mutation_index_unit = None

            most_frequent_element = max(element_counts, key=element_counts.get)  # type: ignore
            frequency_count = element_counts[most_frequent_element]
            while mutation_index_period == None and mutation_index_unit == None:
                for i in range(len(tempChromosome)):
                    for j in range(len(tempChromosome[i])):
                        if tempChromosome[i][j] == most_frequent_element:
                            if random.random() < (1 / frequency_count):
                                mutation_index_period = i
                                mutation_index_unit = j

            del tempChromosome[mutation_index_period][mutation_index_unit]  # type: ignore
            least_frequent_element = min(element_counts, key=element_counts.get)  # type: ignore # type: ignore

            chromosome[mutation_index_period][
                mutation_index_unit
            ] = least_frequent_element

    return chromosome


# Roulette Wheel
def rouletteWheelSelection(fitnessScorePopulation):
    total_fitness = sum(fitnessScorePopulation)
    selection_probs = [score / total_fitness for score in fitnessScorePopulation]
    selected_indices = random.choices(
        population=range(len(fitnessScorePopulation)), weights=selection_probs, k=2
    )
    return selected_indices


# Check duplicates
def hasDuplicates(my_list):
    return len(my_list) != len(set(my_list))


# Crossover funct
def crossOver(chromosome1, chromosome2):
    if random.random() < 0.5:
        # tentukan crossover point
        crossover_point = random.randint(1, 6)

        firstChromosome = chromosome1
        secondChromosome = chromosome2

        for i in range(crossover_point, 6):
            temp = firstChromosome[i]
            firstChromosome[i] = secondChromosome[i]
            secondChromosome[i] = temp

        finalChromosome = [firstChromosome, secondChromosome]
    else:
        finalChromosome = [chromosome1, chromosome2]

    return finalChromosome


# Random Population Generator
def generateRandomPopulation(population_size, listrik_instance):
    population = []
    for _ in range(population_size):
        solution = generateRandomChromosome(listrik_instance)  # Random Chromosome
        population.append(solution)
    return population


# Valid Population Generator
def generatePopulation(population_size, listrik_instance): #INCOMPLETE. DON'T USE. CHECK REFERENCE
    population = []
    for _ in range(population_size):
        solution = generateChromosome(listrik_instance)  #VALID
        population.append(solution)
    return population


# Child Chromosome
def fillChild(population, currentPopulation, fitnessScorePopulation, mutationRate):
    # /2 karena sekali crossover bikin 2 anak
    # operasi // itu pembagian, hanya saja resultnya nanti integer.

    # for i in range(len(population)//2):
    tempCurrentPopulation = copy.deepcopy(currentPopulation)
    for i in range(9):
        hasilRoulette = rouletteWheelSelection(fitnessScorePopulation)
        indexParent1 = hasilRoulette[0]
        indexParent2 = hasilRoulette[1]
        # print("parent1 : ", population[indexParent1])
        # print("parent2 : ", population[indexParent2])

        hasilCrossover1, hasilCrossover2 = crossOver(
            population[indexParent1], population[indexParent2]
        )

        # print("hasilCrossover1 before mutate = ", hasilCrossover1)
        # print("hasilCrossover2 before mutate = ", hasilCrossover2)

        hasilCrossover1 = mutate(hasilCrossover1, mutationRate)
        hasilCrossover2 = mutate(hasilCrossover2, mutationRate)
        # print("hasilCrossover1 after mutate : ", hasilCrossover1)
        # print("hasilCrossover2 after mutate: ", hasilCrossover2)

        tempCurrentPopulation.append(hasilCrossover1)
        tempCurrentPopulation.append(hasilCrossover2)

    return tempCurrentPopulation


# array = [[0 for j in range(6)] for i in range(2)]
# array = generateChromosome(test1)


def generateMaintenanceSchedule(listrik_instance):
    # dimensi 1 mewakili periode (bulan), dimensi 2 mewakili pembangkit listrik yang di maintenance
    population = generateRandomPopulation(20, listrik_instance)
    firstPopulation = population
    fitnessScore = []
    bestChromosome = []
    iterationCount = 0
    bestChromosomeHistory = []
    bestChromosomeFitnessHistory = []

    while True:
        currentPopulation = []
        iterationCount += 1

        # print("\n\n\n EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"EPOCH COUNT = ", iterationCount,"ITERATION COUNT = ", iterationCount,"ITERATION COUNT = ", iterationCount,"ITERATION COUNT = ", iterationCount, "\n\n\n")

        fitnessScorePopulation = calculateFitnessScore(
            population, tablePembangkitListrik, listrik_instance
        )

        # print("fitnessScorePopulation after calculate fitness score = ", fitnessScorePopulation)

        mutationRate = 0.5

        currentPopulation = fillElite(
            population, currentPopulation, fitnessScorePopulation
        )
        # print("currentPopulation after fillElite", currentPopulation)

        currentPopulation = fillChild(
            population, currentPopulation, fitnessScorePopulation, mutationRate
        )
        fitnessScoreCurrentPopulation = calculateFitnessScore(
            currentPopulation, tablePembangkitListrik, listrik_instance
        )
        # print("currentPopulation after fillChild", currentPopulation)
        # print("fitnessScoreCurrentPopulation: ", fitnessScoreCurrentPopulation)
        # kalo udah cukup bagus chromosomenya, return.
        bestChromosome, bestChromosomeFitness = takeBestChromosome(currentPopulation, fitnessScoreCurrentPopulation)  # type: ignore
        population = currentPopulation
        bestChromosomeHistory.append(bestChromosome)
        bestChromosomeFitnessHistory.append(bestChromosomeFitness)
        # print(bestChromosomeFitnessHistory)
        population = copy.deepcopy(currentPopulation)
        if terminate(bestChromosomeFitness, iterationCount):
            # print("bestChromosomeFitness", bestChromosomeFitness)
            # print("History Chromosome : ", bestChromosomeHistory)
            # print("History Fitness: ", bestChromosomeFitnessHistory)
            # print("First Population :", firstPopulation)
            # indices = list(range(len(bestChromosomeFitnessHistory)))

            # plt.plot(indices, bestChromosomeFitnessHistory)

            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.title('Graph based on a list')

            # plt.show()
            return bestChromosome, bestChromosomeFitness, bestChromosomeFitnessHistory
            break  # shuu
            break


def plot(bestChromosomeFitnessHistory):
    indices = list(range(len(bestChromosomeFitnessHistory)))
    plt.plot(indices, bestChromosomeFitnessHistory)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Graph based on a list")

    plt.show()
    
tablePembangkitListrik = []
def run(tablePembangkitListrikk):
    tablePembangkitListrik = tablePembangkitListrikk
    print(type(tablePembangkitListrik))
    test1 = Listrik(tablePembangkitListrik, 6, 110)
    runCount = 0
    bestHistory = []
    print(tablePembangkitListrik)
    runCount = 0
    print("runCount:", runCount)
    while True:
        schedule, fitness, bestChromosomeFitnessHistory = generateMaintenanceSchedule(test1)
        print(test1.initListrik)
        print(schedule)
        runCount += 1
        bestHistory.append(fitness)
        if fitness == 100:
            output = schedule + schedule  # type: ignore
            print("\n Maintenance Schedule: \n")
            row_names = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]
            df = pd.DataFrame(
                output, columns=["Generator 1 ", "Generator 2 "], index=row_names
            ).T
            print(df)
            print("\n The Fitness: ", fitness)
            print("\n Run Count: ", runCount)
            print("\n Best History: ", bestHistory)
            plot(bestChromosomeFitnessHistory)
            break
