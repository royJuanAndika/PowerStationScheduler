from ast import List
import matplotlib.pyplot as plt
import copy
import random
from collections import Counter
from itertools import chain
import pandas as pd


class Listrik:
    def __init__(self, tabelPembangkitListrik, periode, minListrik, maxMT, SelectionMethod):
        self.TabelPembangkitListrik = tabelPembangkitListrik
        self.SelectionMethod = SelectionMethod
        self.periode = periode
        self.minListrik = minListrik
        self.initListrik = sum(tabelPembangkitListrik[0])
        self.maxValue, self.maxIndex = self.findMaxAndIndex()
        self.maxMT = maxMT

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
        
    def getMaxMT(self):
        return self.maxMT

    def setMaxMT(self, maxMT):
        self.maxMT = maxMT
        
    def getSelectionMethod(self):
        return self.SelectionMethod

    def setSelectionMethod(self, SelectionMethod):
        self.SelectionMethod = SelectionMethod


# tablePembangkitListrik = [
#     [0, 0],
#     [20, 2],
#     [15, 2],
#     [35, 1],
#     [40, 1],
#     [15, 1],
#     [15, 2],
#     [10, 1],
# ]
tablePembangkitListrik = [
    [10, 0],
    [20, 2],
    [15, 2],
    [35, 1],
    [40, 1],
    [1, 1],
    [1, 2],
    [1, 1],
]
test1 = Listrik(tablePembangkitListrik, 6, 110, 150, 2, 1) #UTAMA


tablePembangkitListrik2 = [
    [10, 1],
    [20, 2],
    [15, 3],
    [35, 1],
    [40, 1],
    [15, 1],
    [15, 2],
    [10, 1],
    # [30, 2],
    # [25, 4],
]
test2 = Listrik(tablePembangkitListrik2, 6, 110, 150, 2, 2) #EXPERIMENTAL

#loadFile punya William




# Random Chromosome
def generateRandomChromosome(listrik_instance):
    unit = [row[0] for row in listrik_instance.getTabelPembangkitListrik()]
    interval = [row[1] for row in listrik_instance.getTabelPembangkitListrik()]
    capacity = [listrik_instance.getInitListrik()] * listrik_instance.getPeriode()
    arr = [[0 for j in range(listrik_instance.getPeriode())] for i in range(listrik_instance.getMaxMT())]

    for i in range(listrik_instance.getMaxMT()):
        for j in range(listrik_instance.getPeriode()):
            num = random.randint(0, len(unit) - 1)
            if num != 0 and capacity[j] >= unit[num]:
                interval[num - 1] -= 1
                capacity[j] -= unit[num - 1]
                arr[i][j] = num
            else:
                arr[i][j] = 0

    # Transpose the list of lists
    transposed_arr = [[arr[j][i] for j in range(listrik_instance.getMaxMT())] for i in range(listrik_instance.getPeriode())]
    # print(transposed_arr)
    return transposed_arr



# Random Chromosome
def check(interval):
    # Return True if all intervals are not zero
    for i in range(len(interval)):
        if interval[i] == 0:
            return False
    return True

def hasDuplicates(my_list):
    return len(my_list) != len(set(my_list))


def first_number_with_multiple_appearances(arr):
    appearance_count = {}
    for num in arr:
        if num in appearance_count:
            return num
        else:
            appearance_count[num] = 1
    return None

def fixDuplicate(populationIJ):
    tempPopulationIJ = copy.deepcopy(populationIJ)
    numberWithDuplicate = first_number_with_multiple_appearances(tempPopulationIJ)
    if(numberWithDuplicate == None):
        return tempPopulationIJ
    
    
    randomNumber = random.randint(0,1)
    if(randomNumber == 0):
        for i in range(len(tempPopulationIJ)):
            if tempPopulationIJ[i] == numberWithDuplicate:
                tempPopulationIJ[i] = 0
                break
    else:
        for i in range(len(tempPopulationIJ)-1, 0, -1):
            if tempPopulationIJ[i] == numberWithDuplicate:
                tempPopulationIJ[i] = 0
                break
    return tempPopulationIJ

def clearDuplicates(population):
    for i in range(0, len(population)):
        for j in range(len(population[i])): #for chromosome
            if hasDuplicates(population[i][j]): #for period -> list of unit
                population[i][j] = fixDuplicate(population[i][j])
    return population

#NO DUPE
def generateChromosome(listrik_instance):
    unit = [row[0] for row in listrik_instance.getTabelPembangkitListrik()]
    interval = [row[1] for row in listrik_instance.getTabelPembangkitListrik()]
    capacity = [listrik_instance.getInitListrik()] * listrik_instance.getPeriode()
    arr = [[0 for j in range(listrik_instance.getPeriode())] for i in range(listrik_instance.getMaxMT())]

    for i in range(listrik_instance.getMaxMT()):
        for j in range(listrik_instance.getPeriode()):
            num = random.randint(0, len(unit) - 1)
            if num != 0 and capacity[j] >= unit[num]:
                interval[num - 1] -= 1
                capacity[j] -= unit[num - 1]
                arr[i][j] = num
            else:
                arr[i][j] = 0

    # Transpose the list of lists
    transposed_arr = [[arr[j][i] for j in range(listrik_instance.getMaxMT())] for i in range(listrik_instance.getPeriode())]
    return transposed_arr

array = [[0 for j in range(test2.getPeriode())] for i in range(test2.getMaxMT())]
array = generateChromosome(test2)


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
    # if not listOfFitness or all(v is None for v in listOfFitness):
    #     return None, None

    maxIndexOfFitness = max(listOfFitness)
    # print("max index of fitness", maxIndexOfFitness)
    if maxIndexOfFitness is None:
        # choosen = random
        # return currentPopulation[choosen], listOfFitness[choosen]
        maxIndexOfFitness = listOfFitness[0]
    # return valuenya, bukan index.
    # DEBUG
    # print("max index of fitness",maxIndexOfFitness)
    for i in range(0, len(listOfFitness) - 1):
        if listOfFitness[i] == maxIndexOfFitness:
            return currentPopulation[i], listOfFitness[i]
        else:
            return currentPopulation[0], listOfFitness[0]


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
            if sisaMegaWatt < listrik_instance.getMinListrik():
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
            print("tempChromosome = ", tempChromosome)
            list_1d = list(chain.from_iterable(tempChromosome))
            print("list_1d = ", list_1d)
            element_counts = Counter({i: 0 for i in range(0, 8)})
            print("element_counts = ", element_counts)
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
    if total_fitness == 0:
        total_fitness = 1
    selection_probs = [score / total_fitness for score in fitnessScorePopulation]
    selected_indices = random.choices(
        population=range(len(fitnessScorePopulation)), weights=selection_probs, k=2
    )
    return selected_indices


# Check duplicates
def hasDuplicates(my_list):
    return len(my_list) != len(set(my_list))


# Crossover funct
def crossOver(chromosome1, chromosome2, listrik_instance):
    if random.random() < 0.5:
        # tentukan crossover point
        crossover_point = random.randint(1, listrik_instance.getPeriode())

        firstChromosome = chromosome1
        secondChromosome = chromosome2

        for i in range(crossover_point, listrik_instance.getPeriode()):
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
    populationNeo = clearDuplicates(population)
    return populationNeo


# Child Chromosome
def fillChild(population, currentPopulation, fitnessScorePopulation, mutationRate, listrik_instance):
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
            population[indexParent1], population[indexParent2], listrik_instance
        )

        # print("hasilCrossover1 before mutate = ", hasilCrossover1)
        # print("hasilCrossover2 before mutate = ", hasilCrossover2)

        hasilCrossover1 = mutate(hasilCrossover1, mutationRate)
        hasilCrossover2 = mutate(hasilCrossover2, mutationRate)
        # print("hasilCrossover1 after mutate : ", hasilCrossover1)
        # print("hasilCrossover2 after mutate: ", hasilCrossover2)

        tempCurrentPopulation.append(hasilCrossover1)
        tempCurrentPopulation.append(hasilCrossover2)
        tempCurrentPopulationNeo = clearDuplicates(tempCurrentPopulation)

    return tempCurrentPopulationNeo

def tournament_selection(population, fitness_scores, tournament_size):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(range(len(population)), tournament_size)
        best_individual = max(tournament, key=lambda i: fitness_scores[i])
        selected.append(population[best_individual])
    return selected

def fillChild2(population, currentPopulation, fitnessScorePopulation, mutationRate, listrik_instance):
    tournament_size = 3
    for _ in range(9):
        selected_parents = tournament_selection(population, fitnessScorePopulation, tournament_size)
        parent1 = selected_parents[0]
        parent2 = selected_parents[1]

        hasilCrossover1, hasilCrossover2 = crossOver(parent1, parent2, listrik_instance)

        hasilCrossover1 = mutate(hasilCrossover1, mutationRate)
        hasilCrossover2 = mutate(hasilCrossover2, mutationRate)

        currentPopulation.append(hasilCrossover1)
        currentPopulation.append(hasilCrossover2)
        print(currentPopulation)

    return clearDuplicates(currentPopulation)



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

        # print("\n", currentPopulation, "\n")
        
        if(listrik_instance.getSelectionMethod() == 2):
            currentPopulation = fillChild2(
                population, currentPopulation, fitnessScorePopulation, mutationRate, listrik_instance
            )
        elif(listrik_instance.getSelectionMethod() == 1):
                    currentPopulation = fillChild(
            population, currentPopulation, fitnessScorePopulation, mutationRate, listrik_instance
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


def plot(bestChromosomeFitnessHistory):
    indices = list(range(len(bestChromosomeFitnessHistory)))
    plt.plot(indices, bestChromosomeFitnessHistory)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Graph based on a list")

    plt.show()


runCount = 0
bestHistory = []
while True:
    schedule, fitness, bestChromosomeFitnessHistory = generateMaintenanceSchedule(test2)
    runCount += 1
    bestHistory.append(fitness)
    if fitness >=97:
        output = schedule  # type: ignore
        num_rows = len(output)
        num_columns = len(output[0]) if output else 0
        row_names = [str(i) for i in range(1, num_rows + 1)]
        column_names = [f"Generator {i}" for i in range(1, num_columns + 1)]

        df = pd.DataFrame(output, columns=column_names, index=row_names).T

        print(df)
        print("\n The Fitness: ", fitness)
        print("\n Run Count: ", runCount)
        print("\n Best History: ", bestChromosomeFitnessHistory)
        plot(bestChromosomeFitnessHistory)
        break
