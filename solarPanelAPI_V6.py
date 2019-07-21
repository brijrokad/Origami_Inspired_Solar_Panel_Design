###imports
import numpy as np
import pandas as pd
import random
import math
import statistics

from pvlib import modelchain as mc
from pvlib import pvsystem


def makeCutBitString(l, w, maxCuts, res=6):
    """
    HELPER FOR MAKING AN INDIVIDUAL
    inputs- l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            maxCuts- int - maximum amount of cuts on the initial solar panel
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of bits where a bit represents an inch along the solar panel. the first
            'l' bits represent vertical positions, and the next 'w' bits represent horizontal positions,
            such that each position happens every 'res' inches.
    """
    #figure out the number of the bit strings
    numBits = int((l-1)/res) + int((w-1)/res)

    #fill the list with 0's
    bitList = [0 for i in range(numBits)]

    #get the number of cuts
    numCuts = random.randint(1, maxCuts)

    #Lets put in the 1's that represent the cuts
    for i in range(numCuts):
        val = random.randint(0,numBits-1)

        #make sure that it doesn't replace a value in bitstring which is already a 1
        while bitList[val] == 1:
            val = random.randint(0,numBits-1)

        #replace the appropriate 0 with a 1
        bitList[val] = 1
        
    #return final bitstring
    return bitList

def getFaceListBitString(l, w, aBitString, res=6):
    """
    HELPER FOR FINDING FACES
    inputs- l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            aBitString- a list of bits - with a length equal to (l+w)/res
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of tuples where each tuple has 3 ints representing the info for each face. 
            The ints are the length, width, and area of the face. The whole list contains all of the faces
    """
    # error check
    if len(aBitString) != int((l-1)/res) + int((w-1)/res):
        return "ERROR: the 'l' and 'w' parameters do not correspond to the given bit string"
    
    # get the total number of bits
    numBits = len(aBitString)

    bitLength = int(l/res)
    bitWidth = int(w/res)

    # make bit strings that represent just the length and the width of the solar panel respectively
    lenBitString = aBitString[0:bitLength]
    widBitString = aBitString[bitLength:]
    
    # initialize list of faces
    faceList = []

    ## get list of the positions where a cut will happen
    # vertical cuts
    vCuts = [i+1 for i in range(len(lenBitString)) if lenBitString[i] == 1]
    # horizontal cuts
    hCuts = [i+1 for i in range(len(widBitString)) if widBitString[i] == 1]

    ## initialize some variables
    vLow = 0
    vHigh = bitLength
    hLow = 0
    hHigh = bitWidth

    # make lists storing at what inches chuts are being made
    vCuts = [0] + vCuts + [bitLength]
    hCuts = [0] + hCuts + [bitWidth]

    # initalize some vars
    lengths = []
    widths = []

    # fill 'lengths' with the length of each face
    for i in range(len(vCuts)-1):
        vLow = vCuts[0]
        vHigh = vCuts[1]
        length = vHigh - vLow
        lengths.append(length*res)

        # update vCuts
        vCuts = vCuts[1:]

    # fill 'widths' with the width of each face
    for i in range(len(hCuts)-1):
        hLow = hCuts[0]
        hHigh = hCuts[1]
        width = hHigh - hLow
        widths.append(width*res)

        # update vCuts
        hCuts = hCuts[1:]

    ## fill faceList with areas
    #for every length
    for i in range(len(lengths)):
        length = lengths[i]
        # for every width
        for j in range(len(widths)):
            width = widths[j]
            area = length*width
            faceList.append((length, width, area))

    return faceList

def makeIndividual(l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=6):
    """
    MAKES AN INDIVIDUAL- HELPER FOR MAKING A POPULATION
    inputs- l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            maxCuts- int - maximum amount of cuts on the initial solar panel
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of ints - random individual whose first int((l-1)/res) + int((w-1)/res) elements are bits 
            representing where a cut is made the other ((maxCuts/2)+1)*((maxCuts/2)+1) elements (assuming maxCuts is even)
            represent the z, theta, and alpha values. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
    """
    # error check to see if maxCuts can be defined
    if (maxCuts % 2) == 0:
        maxFaces = ((maxCuts/2)+1)*((maxCuts/2)+1)
    else:
        return "ERROR: You need to figure out how many maxFaces will result when there is an odd number of maxCuts"
    
    aBitString = makeCutBitString(l, w, maxCuts, res)
    
    # initalize list
    paramList = []

    for i in range(int(maxFaces)):
        zVal = random.randint(zMin, zMax)
        tVal = random.randint(tMin, tMax)
        aVal = random.randint(aMin, aMax)
        paramList.append(zVal)
        paramList.append(tVal)
        paramList.append(aVal)
        
    individual = aBitString + paramList
    
    #make sure these bits are 0.
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    bitLength = int(l/res)

    individual[bitLength] = 0
    individual[bitStringLen-1] = 0
    
    #resolution
    individual = bitStringResolution(individual, bitStringLen, maxCuts)
    
    return individual

def makePopulation(popSize, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=6):
    """
    MAKES A POPULATION
    inputs- popSize - int - the number of individuals in the population
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            maxCuts- int - maximum amount of cuts on the initial solar panel
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of lists of ints - A list of individuals. Each individual's first ((l-1)+(w-1))/res elements a
            re bits representing where a cut is made the other ((maxCuts/2)+1)*((maxCuts/2)+1) elements 
            (assuming maxCuts is even) represent the z, theta, and alpha values. 
            It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
    """
    population = []
    for i in range(popSize):
        individ = makeIndividual(l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res)
        population.append(individ)
        
    return population

def getFaceList(individ, l, w, res=6):
    """
    GETS LIST OF FACES AND MEASUREMENTS/AREAS
    inputs- individ - list of ints - an individual
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of tuples where each tuple has 3 ints representing the info for each face. 
            The ints are the length, width, and area of the face. The whole list contains all of the faces
    """
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    aBitString = individ[:bitStringLen]
    faceList = getFaceListBitString(l, w, aBitString, res)
    
    return faceList

def getFitness(individ, numConflicts, l, w, sapm, cec, df, latitude, longitude, currFitCount, res=6, conflictDeduction=50):
    '''
    TO BE USED TO GRAB THE FITNESS OF AN INDIVIDUAL
    USED AS A HELPER FUNCTION TO CREATE THE FITNESS OF THE POPULATION
    GETS THE FITNESS OF THE INDIVIDUAL
    inputs- individ - list of ints - an individual
            numConflicts - int - the number of conflicts an individual has. This value comes from conflictCount
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
            conflictDeduction - int - the amount an individual's fitness should be decreased for every conflict it has
    Output- (float) the scaled fitness value for the individual
          - (int) the new currFitCount
    '''


    #Dropping the cuts
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    indAngles = individ[bitStringLen:]
    
    #Int to keep track of which face
    face_position = 0
    indFitness = 0
    FaceList = getFaceList(individ, l, w, res)
    
    #Calculating the fitness and scaling
    for face in FaceList:
        
        
        #It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
        #Our basic chain model.
        #Surface_tilt is our theta ==> indAngles[face_position+1]
        #Surface_arimuth is our alpha ==> indAngles[face_position+2]
        model = mc.basic_chain(df.index, latitude, longitude , module_parameters = sapm, inverter_parameters = cec,
               surface_tilt=indAngles[face_position+1], surface_azimuth=indAngles[face_position+2])
        #Droping the columns we dont need
        power = model[0].drop(columns = ['i_sc','i_mp','v_oc','i_x','i_xx'])

        #grabbing the face's maximum power before scaling
        facePower = power['2012-07'].sum(0)[1]
        
        #scale the face's fitness based on area and add it to the overall sum
        indFitness += facePower*(face[2]/(l*w))
        #v_mp is maximum voltage
        #p_mp is maximum power
        
        face_position += 3
        
    #find conflict deduction amount
    deduction = numConflicts*conflictDeduction
    
    #update fitness
    indFitness = indFitness - deduction

    currFitCount += 1
    
    return indFitness, currFitCount
    

def getPopFitness(population, l, w, zMin, zMax, tMin, tMax, aMin, aMax, currFitCount, sapm, cec, df, latitude, longitude,
            res=6, zThresh=20, tThresh=90, aThresh=45, conflictDeductions=50):
    
    #initialize the fitness list
    popFitness = []
    
    for ind in population:
        #get the number of conflicts
        numConflicts = conflictCount(ind, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
        #append fitness of individual to the list
        fitness, currFitCount = getFitness(ind, numConflicts, l, w, sapm, cec, df, latitude, longitude, res, conflictDeductions)
        popFitness.append(fitness)
        
    return popFitness, currFitCount
        

def conflictCount(individual, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res=6, zThresh=20, tThresh=90, aThresh=45):
    """
    MAKES A POPULATION
    inputs- individual - list of ints - an individual. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
            zThresh- the threshold that describes if two solar panels are closer then zThresh inches, and the
                     other thresholds are passed, then that face will be mutated
            tThresh- the threshold that describes if two solar panels are closer then tThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
            aThresh- the threshold that describes if two solar panels are closer then aThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
    output- takes in an individual and returns a new individual
    
    NOTE: Currently the function does not mutate the t param or look at the tDiff
    """
    #initialize the conflict count
    conflictCount = 0
    #get the length of the bitString
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    paramList = individual[bitStringLen:]
    #for every index in the paramList
    for i1 in range(len(paramList[0:1])):
        #if param is a z value
        if (i1 % 3) == 0:
            #get values
            zVal = paramList[i1]
            tVal = paramList[i1+1]
            aVal = paramList[i1+2]
            #for every index in the rest of the paramList
            for i2 in range(len(paramList[i1+3:])):
                nextIndex = i1+i2
                #if nextParam is a z-value
                if (nextIndex % 3) == 0:
                    nextZVal = paramList[i1+3:][nextIndex]
                    nextTVal = paramList[i1+3:][nextIndex+1]
                    nextAVal = paramList[i1+3:][nextIndex+2]
                    #compare nextParams to params
                    zDiff = abs(zVal - nextZVal)
                    tDiff = abs(tVal - nextTVal)
                    aDiff = abs(aVal - nextAVal)
                    #while all thresholds are crossed
                    if zDiff <= zThresh and aDiff <= aThresh:
                        conflictCount+= 1

    #return result
    return conflictCount

def bitStringResolution(individual, bitStringLen, maxCuts):
    """
    inputs - individual - list of ints - an individual. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
           - bitStringLen - int - length of bitstring at beginning of individual. equal to int((l-1)/res) + int((w-1)/res)
           - maxCuts - int - max number of cuts allowed
    """
    #get the bit string
    bitString = individual[:bitStringLen]

    # vars to store number of 1's, number of extra cuts, and list of the indexes of the ones
    numCuts = 0
    extraCuts = 0
    indexList = []

    index = 0
    for bit in bitString:
        if bit == 1:
            numCuts += 1
            indexList.append(index)
            if numCuts > maxCuts:
                extraCuts += 1
        index += 1

    if numCuts == 0:
        indexToFlip = random.randint(0, len(bitString)-1)
        individual[indexToFlip] = 1
        
    #for every extra cut, make one 1 into a zero
    for i in range(extraCuts):

        #get get index of the 1 to flip to a 0
        randomIndex = random.randint(0, len(indexList)-1)
        indexToFlip = indexList[randomIndex]

        #flip a 1 to a zero
        individual[indexToFlip] = 0

        #update indexList so you don't reflip any bits
        del indexList[randomIndex]

    return individual

def uniformCrossover(p1, p2):
    """
    CROSSOVER
    inputs- p1 - list of intss - an individual 
            p2 - list of intss - an individual 
    outputs - child1 - list of floats- a new individual that is made from uniform crossover from parents
            - child2 - list of floats- a new individual that is made from uniform crossover from parents
    """
    #error check
    if len(p1) != len(p2):
        return "ERROR: parents are not the same length"
    
    #initialize children
    child1 = []
    child2 = []
    
    #do uniform crossover
    for i in range(len(p1)):
        coinFlip = round(random.uniform(0,1), 3)
        if coinFlip < 0.5:
            child1.append(p1[i])
            child2.append(p2[i])
        else:
            child1.append(p2[i])
            child2.append(p1[i])
    return child1, child2

def onePointCrossover(p1, p2):
    """
    CROSSOVER
    inputs- p1 - list of ints - an individual 
            p2 - list of ints - an individual 
    outputs - child1 - list of ints- a new individual that is made from uniform crossover from parents
            - child2 - list of ints- a new individual that is made from uniform crossover from parents
    """
    #error check
    if len(p1) != len(p2):
        return "ERROR: parents are not the same length"
    
    #initialize children
    child1 = []
    child2 = []
    
    #get the split point
    splitPoint = random.randint(1, len(p1)-1)
    
    startIndex = 0
    endIndex = splitPoint
    
    #append the beginning part of the parents to the children
    child1.extend(p1[startIndex: endIndex])
    child2.extend(p2[startIndex: endIndex])
    
    #update start and end points
    startIndex = splitPoint
    endIndex = len(p1)
    
    #append the ending part of the parents to the children
    child1.extend(p2[startIndex: endIndex])
    child2.extend(p1[startIndex: endIndex])

    return child1, child2

def tournamentSelection(population, fitnessPop, optimize='max'):
    """
    PARENT SELECTION
    inputs - population - a list of lists of integers. Each list of integers is an individual.
             fitnessPop - list of floats - list of fitness such that the index of a fitness matches the index of the 
                                           individual from 'population' 
             optimize - string - either 'min' or 'max'. If 'max', then a higher fitness will be preferred. If 'min'
                                 then a lower fitness will be preferred
    outputs - parent1, parent2 - these will be two parents chosen from the population
    
    NOTE: this assumes only two combatants in the tournament
    """
    #initialize list of parents (should be of length 2)
    parents = []
    
    #get number of individuals in population
    numOfIndividuals = len(population)
    
    #to get two parents
    for i in range(2):
        #get index of both combatants
        index1 = random.randint(0, numOfIndividuals-1)
        index2 = random.randint(0, numOfIndividuals-1)

        #stops the same individual from battling itself
        while index1==index2:
            index2 = random.randint(0, numOfIndividuals-1)

        #get fighters
        fighter1 = population[index1]
        fighter2 = population[index2]
        ##get fitnesses
        fitness1 = fitnessPop[index1]
        fitness2 = fitnessPop[index2]
        ##find winner
        if optimize=='max':
            if fitness1 >= fitness2:
                parents.append(fighter1)
            else:
                parents.append(fighter2)
                
        elif optimize == 'min':
            if fitness1 <= fitness2:
                parents.append(fighter1)
            else:
                parents.append(fighter2)
        
        else:
            return "ERROR: the param 'optimize' is not given a valid value. It must be empty, 'max', or 'min'"
    
    #return result
    return parents[0], parents[1]

def tournamentSelectionGenerational(population, fitnessPop, optimize='max'):
    """
    PARENT SELECTION
    inputs - population - a list of lists of integers. Each list of integers is an individual.
             fitnessPop - list of floats - list of fitness such that the index of a fitness matches the index of the 
                                           individual from 'population' 
             optimize - string - either 'min' or 'max'. If 'max', then a higher fitness will be preferred. If 'min'
                                 then a lower fitness will be preferred
    outputs - parent1, parent2 - these will be two parents chosen from the population
    
    NOTE: this assumes only two combatants in the tournament
    """
    #initialize list of parents (should be of length 2)
    parents = []
    
    #get number of individuals in population
    numOfIndividuals = len(population)
    
    #to get two parents
    for i in range(2):
        #get index of both combatants
        index1 = random.randint(0, numOfIndividuals-1)
        index2 = random.randint(0, numOfIndividuals-1)

        #stops the same individual from battling itself
        while index1==index2:
            index2 = random.randint(0, numOfIndividuals-1)

        #get fighters
        fighter1 = population[index1]
        fighter2 = population[index2]
        ##get fitnesses
        fitness1 = fitnessPop[index1]
        fitness2 = fitnessPop[index2]
        ##find winner
        if optimize=='max':
            if fitness1 >= fitness2:
                parents.append(fighter1)
            else:
                parents.append(fighter2)
                
        elif optimize == 'min':
            if fitness1 <= fitness2:
                parents.append(fighter1)
            else:
                parents.append(fighter2)
        
        else:
            return "ERROR: the param 'optimize' is not given a valid value. It must be empty, 'max', or 'min'"
    
    #return result
    return parents[0], parents[1], fitness1, fitness2

def replaceWorst(population, newIndividual1, newIndividual2, fitnessPop, l, w, zMin, zMax, tMin, tMax, aMin, aMax, 
                sapm, cec, df, latitude, longitude, currFitCount, res=6, zThresh=20, tThresh=90, 
                aThresh=45, conflictDeduction=50, optimize='max'):
    """
    SURVIVAL SELECTION- STEADY STATE
    inputs - population - a list of lists of integers. Each list of integers is an individual and
                          is assumed to be a permutation.
             newIndividual1 -  a list of integers -This will replace the worst out of the population
             newIndividual2 -  a list of integers - This will replace the 2nd worst out of the population
             fitnessPop - list of floats - list of fitness such that the index of a fitness matches the index of the 
                                           individual from 'population'
             l- int - length of the initial solar panel
             w- int - width of the initial solar panel
             zMin - int - min possible value of z
             zMax - int - min possible value of z
             tMin - int - min possible value of t
             tMax - int - min possible value of t
             aMin - int - min possible value of a
             aMax - int - min possible value of a
             res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
             zThresh- the threshold that describes if two solar panels are closer then zThresh inches, and the
                      other thresholds are passed, then that face will be mutated
             tThresh- the threshold that describes if two solar panels are closer then tThresh degrees, and the
                      other thresholds are passed, then that face will be mutated
             aThresh- the threshold that describes if two solar panels are closer then aThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
             conflictDeduction - int - the amount an individual's fitness should be decreased for every conflict it has
             optimize - string - either 'min' or 'max'. If 'max', then a higher fitness will be preferred. If 'min'
                                 then a lower fitness will be preferred
    output - a new population where 'newIndividual1' and 'newIndividual2' has replaced the worst 2 individuals.
           - a new fitnessPop where the appropriate fitnesses have been replaced with the fitnesses of 
                   'newIndividual1' and 'newIndividual2'.
    
    NOTE: this function takes in two children and they will replace the two worst people in the given population
    NOTE: this function assumes that the possible fitnesses are between -9999999999999 and 9999999999999.
    """
    #initialize list of fitness values
    fitnessList = fitnessPop
    
    #get the fitness values of the newIndividuals
    numConflicts1 = conflictCount(newIndividual1, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
    newFitness1, currFitCount = getFitness(newIndividual1, numConflicts1, l, w, sapm, cec, df, latitude, longitude, 
                            currFitCount, res, conflictDeduction)
    
    numConflicts2 = conflictCount(newIndividual1, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
    newFitness2, currFitCount = getFitness(newIndividual2, numConflicts2, l, w, sapm, cec, df, latitude, longitude, 
                            currFitCount, res, conflictDeduction)
    

    if optimize == 'min':
        #find biggest(worst) fitness
        maximum1 = max(fitnessList)
        #find location of individual with worst fitness
        maxIndex1 = fitnessList.index(maximum1)
        #make the worst fitness, not the worst anymore, so i can find the 2nd worst
        fitnessList[maxIndex1] = -9999999999999

        #find 2nd worst fitness
        maximum2 = max(fitnessList)
        #find location of individual with 2nd worst fitness
        maxIndex2 = fitnessList.index(maximum2)

        #replace worst indiviudals in population and replace their fitnesses
        population[maxIndex1] = newIndividual1
        population[maxIndex2] = newIndividual2
        
        fitnessList[maxIndex1] = newFitness1
        fitnessList[maxIndex2] = newFitness2
        
    elif optimize == 'max':
        #find smallest(worst) fitness
        minimum1 = min(fitnessList)
        #find location of individual with worst fitness
        minIndex1 = fitnessList.index(minimum1)
        #make the worst fitness, not the worst anymore, so i can find the 2nd worst
        fitnessList[minIndex1] = 9999999999999

        #find 2nd worst fitness
        minimum2 = min(fitnessList)
        #find location of individual with 2nd worst fitness
        minIndex2 = fitnessList.index(minimum2)

        #replace worst indiviudals in population and replace their fitnesses
        population[minIndex1] = newIndividual1
        population[minIndex2] = newIndividual2
        
        fitnessList[minIndex1] = newFitness1
        fitnessList[minIndex2] = newFitness2
        
    else:
        return "ERROR: the param 'optimize' is not given a valid value. It must be empty, 'max', or 'min'"

    #return results
    return population, fitnessList, currFitCount

def mutation(individual, pm, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res=6, zThresh=20, tThresh=90, aThresh=45):
    """
    MAKES A POPULATION
    inputs- individual - list of ints - an individual. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
            pm - float between 0.0 and 1.0 - the probability a mutation will occur for an individual
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            maxCuts - int - the maximum number of allowed cuts
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
            zThresh- the threshold that describes if two solar panels are closer then zThresh inches, and the
                     other thresholds are passed, then that face will be mutated
            tThresh- the threshold that describes if two solar panels are closer then tThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
            aThresh- the threshold that describes if two solar panels are closer then aThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
    output- takes in an individual and returns a new individual that has been mutated
    """

    #get the lenth of the bit string
    bitStringLen = int((l-1)/res) + int((w-1)/res)

    #see if mutation is done
    check = random.uniform(0, 1)
    if check <= pm:
        #get point where mutation will happen
        mutationPoint = random.randint(0, len(individual)-1)

        #if mutation point is within the bitString, do a bit flip
        if mutationPoint < bitStringLen:
            if individual[mutationPoint] == 1:
                individual[mutationPoint] = 0
            elif individual[mutationPoint] == 0:
                individual[mutationPoint] = 1
        #else if the mutation point is within the param string
        else:
            #get index within the param string so i can check if it is z,t, or a val
            paramIndex = mutationPoint - bitStringLen
            #if a z value is to be mutated
            if (paramIndex % 3) == 0:
                individual[mutationPoint] = random.randint(zMin, zMax)
            #if a t value is to be mutated
            elif (paramIndex % 3) == 1:
                individual[mutationPoint] = random.randint(tMin, tMax)
            #if an a value is to be mutated
            elif (paramIndex % 3) == 2:
                individual[mutationPoint] = random.randint(aMin, aMax)
    
    #This is to reassure that there will be no panel of area 0

    bitLenght = int(l/res)

    individual[bitLenght] = 0
    individual[bitStringLen-1] = 0

    #return result
    return individual

def getAvgPopFitness(fitnessPop):
    """
    input - fitnessPop - list of floats - list of all of the fitnesses
    output - float - the average fitness of the input
    """
    #initialize the sum
    finalSum = 0
    popSize = len(fitnessPop)
    #add each fitness to the sum
    for fitness in fitnessPop:
        finalSum += fitness
    avg = finalSum/popSize
    
    #return result
    return avg

def mutationGuassian(individual, pm, sigma, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=6, zThresh=20, tThresh=90, aThresh=45):
    """
    MAKES A POPULATION
    inputs- individual - list of ints - an individual. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
            pm - float between 0.0 and 1.0 - the probability a mutation will occur for an individual
            sigma- int - standard deviation for the normal distribution
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            maxCuts - int - the maximum number of allowed cuts
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
            zThresh- the threshold that describes if two solar panels are closer then zThresh inches, and the
                     other thresholds are passed, then that face will be mutated
            tThresh- the threshold that describes if two solar panels are closer then tThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
            aThresh- the threshold that describes if two solar panels are closer then aThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
    output- takes in an individual and returns a new individual that has been mutated
    """
    for i in range(len(individual)):
        
            
        #get the lenth of the bit string
        bitStringLen = int((l-1)/res) + int((w-1)/res)

        #if mutation point is within the bitString, do a bit flip
        if i < bitStringLen:
            #see if mutation is done
            check = random.uniform(0, 1)
            if check <= pm:
                if individual[i] == 1:
                    individual[i] = 0
                elif individual[i] == 0:
                    individual[i] = 1
        #else if the mutation point is within the param string
        else:
            #get index within the param string so i can check if it is z,t, or a val
            paramIndex = i - bitStringLen
            #if a z value is to be mutated
            if (paramIndex % 3) == 0:

                randomInt = int(np.random.normal(0,sigma))
                newZ = individual[paramIndex] + randomInt

                #checking to see if newZ falls in appropriate range
                while newZ < zMin or newZ > zMax:
                    randomInt = int(np.random.normal(0,sigma))
                    newZ = individual[paramIndex] + randomInt

                individual[i] = newZ
                
            #if a t value is to be mutated
            elif (paramIndex % 3) == 1:

                randomInt = int(np.random.normal(0,sigma))
                newT = individual[paramIndex] + randomInt

                #checking to see if newZ falls in appropriate range
                while newT < TMin or newT > TMax:
                    randomInt = int(np.random.normal(0,sigma))
                    newT = individual[paramIndex] + randomInt

                individual[i] = newT
                
            #if an a value is to be mutated
            elif (paramIndex % 3) == 2:

                randomInt = int(np.random.normal(0,sigma))
                newT = individual[paramIndex] + randomInt

                while newA < TMin or newA > TMax:
                    randomInt = int(np.random.normal(0,sigma))
                    newA = individual[paramIndex] + randomInt

                individual[i] = newA
    
    #due bit resolution on individual
    individual = bitStringResolution(individual, bitStringLen, maxCuts)
    #return result
    return individual, sigma

def initializeSigmaList(panelParams, pop):
    """
    inputs - panelParams - list of parameters - of form [l, w, maxCuts, res, zMin, zMax, tMin, tMax, aMin, 
                                                            aMax, zThresh, tThresh, aThresh]
           - pop - list of lists of ints - a list of individuals
    output - list of lists of floats - each list is sigmaList for an individual. each float is a sigma/standard deviation
    that corresponds to the standard devation of the gene in the param list with the same index
    """
    l = panelParams[0]
    w = panelParams[1]

    #cut parameters
    maxCuts = panelParams[2]
    res = panelParams[3]

    #param min/max values
    zMin = panelParams[4]
    zMax = panelParams[5]

    tMin = panelParams[6]
    tMax = panelParams[7]

    aMin = panelParams[8]
    aMax = panelParams[9]

    #thresholds
    zThresh = panelParams[10] #inches
    tThresh = panelParams[11] #degrees
    aThresh = panelParams[12] #degrees
    
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    
    finalList = []
    for k in range(len(pop)):
        sigmaList = []

        dictOfSDLists = {}
        #for the length of an individual's paramList.'i' is the index of the current parameter 
        for i in range(len(pop[0]) - bitStringLen):
            listToAdd = []
            #for every individual in the population. j is the index of the current individual
            for j in range(len(pop)):
                individ = pop[j]
                paramList = individ[bitStringLen:]
                listToAdd.append(paramList[i])

            dictOfSDLists['paramNo_%s' % (i+1)] = listToAdd

        #fill sigma list with appropriate sigmas
        for param in dictOfSDLists:
            sigma = statistics.stdev(dictOfSDLists[param])
            sigmaList.append(sigma)
        
        #append the sigmaList
        finalList.append(sigmaList)
        
    #return result
    return finalList


def mutationGuassianList(individual, panelParams, pm, sigmaList):
    """
    MUTATION FOR EP
    inputs- individual - list of ints - an individual. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
            panelParams - list of parameters - of form [l, w, maxCuts, res, zMin, zMax, tMin, tMax, aMin, 
                                                            aMax, zThresh, tThresh, aThresh]
            pm - float between 0.0 and 1.0 - the probability a mutation will occur for an individual
            sigmaList - list of floats - standard deviation for the normal distribution where its length is equal
                                            to the length of the number of elements
            
    output- takes in an individual and returns a new individual that has been mutated
          - a mutated sigmaList
    """
    
    l = panelParams[0]
    w = panelParams[1]

    #cut parameters
    maxCuts = panelParams[2]
    res = panelParams[3]

    #param min/max values
    zMin = panelParams[4]
    zMax = panelParams[5]

    tMin = panelParams[6]
    tMax = panelParams[7]

    aMin = panelParams[8]
    aMax = panelParams[9]

    #thresholds
    zThresh = panelParams[10] #inches
    tThresh = panelParams[11] #degrees
    aThresh = panelParams[12] #degrees
    
    learningRate = 0.2
    minSigma = 0.5
    #get the lenth of the bit string
    bitStringLen = int((l-1)/res) + int((w-1)/res)
    
    ###mutate sigmas
    
    #for every sigma
    for i in range(len(sigmaList)):
        #get sigma
        sigma = sigmaList[i]
        #mutate it with gaussian perturbation
        newSigma = sigma*(1 + (learningRate*np.random.normal(0,1)))
        
        #boundary rule
        if newSigma < minSigma:
            newSigma = minSigma
        sigmaList[i] = newSigma
    
    
    ###mutate individual
    for i in range(len(individual)):
        
        #if mutation point is within the bitString, do a bit flip
        if i < bitStringLen:
            #see if mutation is done
            check = random.uniform(0, 1)
            if check <= pm:
                if individual[i] == 1:
                    individual[i] = 0
                elif individual[i] == 0:
                    individual[i] = 1

        #else if the mutation point is within the param string
        else:
            #get index within the param string so i can check if it is z,t, or a val
            paramIndex = i - bitStringLen
            #if a z value is to be mutated
            if (paramIndex % 3) == 0:
                #get the appropriate sigma value
                sigma = sigmaList[paramIndex]

                #get value to add to current gene
                randomInt = int(sigma * np.random.normal(0,1))

                #make new gene
                newZ = individual[i] + randomInt

                #checking to see if newZ falls in appropriate range
                while newZ < zMin or newZ > zMax:
                    randomInt = int(sigma * np.random.normal(0,1))
                    newZ = individual[i] + randomInt

                #replace old gene with new mutated gene
                individual[i] = newZ

            #if a t value is to be mutated
            elif (paramIndex % 3) == 1:
                #get the appropriate sigma value
                sigma = sigmaList[paramIndex]

                #get value to add to current gene
                randomInt = int(sigma * np.random.normal(0,1))

                #make new gene
                newT = individual[i] + randomInt

                #checking to see if newT falls in appropriate range
                while newT < tMin or newT > tMax:
                    randomInt = int(sigma * np.random.normal(0,1))
                    newT = individual[i] + randomInt

                #replace old gene with new mutated gene
                individual[i] = newT

            #if an a value is to be mutated
            elif (paramIndex % 3) == 2:
                #get the appropriate sigma value
                sigma = sigmaList[paramIndex]

                #get value to add to current gene
                randomInt = int(sigma * np.random.normal(0,1))

                #make new gene
                newA = individual[i] + randomInt

                #checking to see if newZ falls in appropriate range
                while newA < aMin or newA > aMax:
                    randomInt = int(sigma * np.random.normal(0,1))
                    newA = individual[i] + randomInt

                #replace old gene with new mutated gene
                individual[i] = newA
    
    #due bit resolution on individual
    individual = bitStringResolution(individual, bitStringLen, maxCuts)
    #return result
    return individual, sigmaList


def getBest(pop, fitnessPop):
    """
    inputs - pop - list of individuals - a list of all of the individuals
           - fitnessPop - list of floats - list of each indiviudal's fitness. The index in fitnessPop corresponds with the 
                index of pop.
    outputs - list of ints - the best individual
            - float - the best individual's fitness 
    """
    #get max value
    maxFitness = max(fitnessPop)
    index = fitnessPop.index(maxFitness)
    bestIndividual = pop[index]
    
    #return solutions
    return bestIndividual, maxFitness

def steadyStateGA(panelParams, popSize, pm, numMutations, fitnessCountLimit, attempts, checkIteration, conflictDeduction, toPrint=True):
    """
    inputs - panelParams - list of parameters - of form [l, w, maxCuts, res, zMin, zMax, tMin, tMax, aMin, 
                                                            aMax, zThresh, tThresh, aThresh]
           - popSize - int - size of a population
           - pm - float between 0.0 and 1.0 - probability that a gene will be mutated
           - numMutations - int - the number of parameters of an individual that could be mutated in one iteration
           - fitnessCountLimit - int - the max number of fitness evaluations
           - attempts - int - the number of attempts/number of times the whole process is done
           - checkIteration - int - how often you check the avg fitness and best individual
           - conflictDeductions - int - the amount that should be deduced from fitness if there is a conflict
           - toPrint - boolean -  if true, then results will be printed as function runs
    outputs - dfBest - pandas dataframe storing the best individual and their fitnesses
            - dfAvg - pandas dataframe storing the average fitness and the associated iteration for each attempt
    """
    l = panelParams[0]
    w = panelParams[1]

    #cut parameters
    maxCuts = panelParams[2]
    res = panelParams[3]

    #param min/max values
    zMin = panelParams[4]
    zMax = panelParams[5]

    tMin = panelParams[6]
    tMax = panelParams[7]

    aMin = panelParams[8]
    aMax = panelParams[9]

    #thresholds
    zThresh = panelParams[10] #inches
    tThresh = panelParams[11] #degrees
    aThresh = panelParams[12] #degrees

    #parameters for pvlib
    #Athens, Ga Lat/long
    latitude = 33.957409
    longitude = -83.376801

    #Creating DateTimeIndex
    df = pd.DataFrame({'date': ['2012-07-01 11:00:00','2012-07-01 12:00:00',
                                '2012-07-01 13:00:00','2012-07-01 14:00:00','2012-07-01 15:00:00','2012-07-01 16:00:00'
                               '2012-07-01 17:00:00','2012-07-01 18:00:00','2012-07-01 19:00:00'],
                      }).set_index('date')

    #Creating DatetimeIndex Object
    df.index = pd.DatetimeIndex(df.index)


    #Needed paramters for the basic chain model
    #These are constant through the whole simulation
    cec_mod = pvsystem.retrieve_sam('SandiaMod')
    cec_inv = pvsystem.retrieve_sam('CECInverter')
    sapm = cec_mod['Advent_Solar_AS160___2006_']
    cec = cec_inv['ABB__MICRO_0_3_I_OUTD_US_208_208V__CEC_2014_']
    
    #initializing the fitness evaluations
    currFitCount = 0
    
    #initialize the list of best fitnesses and individuals
    bestIndList = []
    bestFitnessList = []

    #initialize the list lists of of average fitnesses
    listOfAvgFitnessLists = []
    for i in range(attempts): 

        ###initialization
        
        #initialize the list of average fitnesses
        avgFitnessList = []

        #initial population
        pop = makePopulation(popSize, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res)

        #initial fitness list
        fitnessList, currFitCount = getPopFitness(pop, l, w, zMin, zMax, tMin, tMax, aMin, aMax, currFitCount, 
                                    sapm, cec, df, latitude, longitude, res, zThresh, tThresh, aThresh, conflictDeduction)

        ###iterative part
        
        #initialize the best individual and fitness
        bestGlobalInd = []
        bestGlobalFitness = 0.0

        ##begin to iterate through

        iteration = 0
        while currFitCount <= fitnessCountLimit:
            ##parent selection
            parent1, parent2 = tournamentSelection(pop, fitnessList, optimize='max')
            ##crossover
            child1, child2 = onePointCrossover(parent1, parent2)
            #mutation of children based on probability. up to 'numMutations' could happen
            for i in range(numMutations):
                #child1
                child1 = mutation(child1, pm, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
                #child2
                child2 = mutation(child2, pm, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
            ##survival selection
            pop, fitnessList, currFitCount  = replaceWorst(pop, child1, child2, fitnessList, l, w, zMin, zMax, tMin, 
                                            tMax, aMin, aMax, sapm, cec, df, latitude, longitude, currFitCount, res, 
                                            zThresh, tThresh, aThresh, conflictDeduction, optimize='max')
            if (iteration % checkIteration) == 0:
                avgFitness = getAvgPopFitness(fitnessList)
                bestInd, bestFitness = getBest(pop, fitnessList)
                
                if toPrint == True:
                    print("iteration " + str(iteration))
                    print("current average fitness: " + str(avgFitness))
                    print("  ")
                    
                #append avg fitness to a list
                avgFitnessList.append(avgFitness)

                if bestFitness > bestGlobalFitness:
                    bestGlobalInd = bestInd
                    bestGlobalFitness = bestFitness
            iteration += 1

        if toPrint == True:
            print("finished")
            print("total number of iterations: " + str(iteration))
            print("best individual: " + str(bestGlobalInd))
            print("best fitness: " + str(bestGlobalFitness))

        bestIndList.append(bestGlobalInd)
        bestFitnessList.append(bestGlobalFitness)
        listOfAvgFitnessLists.append(avgFitnessList)

    if toPrint == True:
        print("  ")
        print("list of best individuals found for each attempt: " + str(bestIndList))
        print("list of best fitnesses found for each attempt: " + str(bestFitnessList))
        print("list of average fitness lists: " + str(listOfAvgFitnessLists))
        print(" ")

    #make pandas dataframe for best individuals and their fitnesses
    bestDf = pd.DataFrame()
    #add columns
    bestDf['best individuals'] = bestIndList
    bestDf['best Fitnesses'] = bestFitnessList

    #make pandas dataframe for average fitness values
    avgDf = pd.DataFrame()
    
    #add colums
    #make iteration List
    iterationList = []
    for i in range(len(listOfAvgFitnessLists[0])):
        iterationList.append(i*checkIteration)
    
    #make iteration column
    avgDf['iterations'] = iterationList
    #make avg fitness columns
    for i in range(len(listOfAvgFitnessLists)):
        fitnessListToAdd = listOfAvgFitnessLists[i]
        avgDf[str(i+1)] = fitnessListToAdd
        
    #return results
    return bestDf, avgDf

def generationalGA(panelParams, popSize, pm, numMutations, fitnessCountLimit, attempts, checkGeneration, conflictDeduction, toPrint=True):
    """
    inputs - panelParams - list of parameters - of form [l, w, maxCuts, res, zMin, zMax, tMin, tMax, aMin, 
                                                            aMax, zThresh, tThresh, aThresh]
           - popSize - int - size of a population
           - pm - float between 0.0 and 1.0 - probability that a gene will be mutated
           - numMutations - int - the number of parameters of an individual that could be mutated in one iteration
           - fitnessCountLimit - int - the max number of fitness evaluations
           - attempts - int - the number of attempts/number of times the whole process is done
           - checkGeneration - int - how often you check the avg fitness and best individual
           - conflictDeductions - int - the amount that should be deduced from fitness if there is a conflict
           - toPrint - boolean -  if true, then results will be printed as function runs
    outputs - dfBest - pandas dataframe storing the best individual and their fitnesses
            - dfAvg - pandas dataframe storing the average fitness and the associated iteration for each attempt
    """
    l = panelParams[0]
    w = panelParams[1]

    #cut parameters
    maxCuts = panelParams[2]
    res = panelParams[3]

    #param min/max values
    zMin = panelParams[4]
    zMax = panelParams[5]

    tMin = panelParams[6]
    tMax = panelParams[7]

    aMin = panelParams[8]
    aMax = panelParams[9]

    #thresholds
    zThresh = panelParams[10] #inches
    tThresh = panelParams[11] #degrees
    aThresh = panelParams[12] #degrees

    #parameters for pvlib
    #Athens, Ga Lat/long
    latitude = 33.957409
    longitude = -83.376801

    #Creating DateTimeIndex
    df = pd.DataFrame({'date': ['2012-07-01 11:00:00','2012-07-01 12:00:00',
                                '2012-07-01 13:00:00','2012-07-01 14:00:00','2012-07-01 15:00:00','2012-07-01 16:00:00'
                               '2012-07-01 17:00:00','2012-07-01 18:00:00','2012-07-01 19:00:00'],
                      }).set_index('date')

    #Creating DatetimeIndex Object
    df.index = pd.DatetimeIndex(df.index)


    #Needed paramters for the basic chain model
    #These are constant through the whole simulation
    cec_mod = pvsystem.retrieve_sam('SandiaMod')
    cec_inv = pvsystem.retrieve_sam('CECInverter')
    sapm = cec_mod['Advent_Solar_AS160___2006_']
    cec = cec_inv['ABB__MICRO_0_3_I_OUTD_US_208_208V__CEC_2014_']
    
    #initializing the fitness evaluations
    currFitCount = 0
    
    #initialize the list of best fitnesses and individuals
    bestIndList = []
    bestFitnessList = []

    #initialize the list lists of of average fitnesses
    listOfAvgFitnesses = []
    for i in range(attempts): 

        ###initialization
        
        #initialize the list of average fitnesses
        avgFitnessList = []

        #initial population
        pop = makePopulation(popSize, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res)

        #initial fitness list
        fitnessList, currFitCount = getPopFitness(pop, l, w, zMin, zMax, tMin, tMax, aMin, aMax, currFitCount, 
                                    sapm, cec, df, latitude, longitude, res, zThresh, tThresh, aThresh, conflictDeduction)
        ###iterative part
        
        #initialize the best individual and fitness
        bestGlobalInd = []
        bestGlobalFitness = 0.0

        ##begin generations

        generation = 0
        
        #initialize next generations
        newPop = []
        newFitnessList = []
        while currFitCount <= fitnessCountLimit:
            #for the length of a population- make a new generation
            while len(newPop) < len(pop):
                ##parent selection
                parent1, parent2, fitnessToAdd1, fitnessToAdd2 = tournamentSelectionGenerational(pop, fitnessList, optimize='max')
                #add fitness count from selecting the two parents
                currFitCount += 2
                ##crossover
                child1, child2 = onePointCrossover(parent1, parent2)
                #mutation of children based on probability. up to 'numMutations' could happen
                for i in range(numMutations):
                    #child1
                    child1 = mutation(child1, pm, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
                    #child2
                    child2 = mutation(child2, pm, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
                
                #append children to new generation
                newPop.append(child1)
                newPop.append(child2) 
                
                #append new fitnesses to new generation
                newFitnessList.append(fitnessToAdd1)
                newFitnessList.append(fitnessToAdd2)
            
            #reset the current population and fitnessList
            pop = newPop
            fitnessList = newFitnessList
            #reset the next population and fitnessList
            newPop = []
            newFitnessList = []
            
            if (generation % checkGeneration) == 0:
                
                avgFitness = getAvgPopFitness(fitnessList)
                
                bestInd, bestFitness = getBest(pop, fitnessList)
                if toPrint == True:
                    print("generation " + str(generation))
                    print("current average fitness: " + str(avgFitness))
                    print("  ")

                #append avg fitness to a list
                avgFitnessList.append(avgFitness)

                if bestFitness > bestGlobalFitness:
                    bestGlobalInd = bestInd
                    bestGlobalFitness = bestFitness
            generation += 1

        if toPrint == True:
            print("finished")
            print("total number of generations: " + str(generation))
            print("best individual: " + str(bestGlobalInd))
            print("best fitness: " + str(bestGlobalFitness))
            print(" ")

        bestIndList.append(bestGlobalInd)
        bestFitnessList.append(bestGlobalFitness)
        listOfAvgFitnesses.append(avgFitnessList)

    if toPrint == True:
        print("  ")
        print("list of best individuals found for each attempt: " + str(bestIndList))
        print("list of best fitnesses found for each attempt: " + str(bestFitnessList))
        print("list of average fitness lists: " + str(listOfAvgFitnesses))
        print(" ")

    #make pandas dataframe for best individuals and their fitnesses
    bestDf = pd.DataFrame()
    #add columns
    bestDf['best individuals'] = bestIndList
    bestDf['best Fitnesses'] = bestFitnessList

    #make pandas dataframe for average fitness values
    avgDf = pd.DataFrame()
    #add colums
    #make generation List
    generationList = []
    for i in range(len(listOfAvgFitnesses[0])):
        generationList.append(i*checkGeneration)

    #make generations column
    avgDf['generations'] = generationList
    #make avg fitness columns
    for i in range(len(listOfAvgFitnesses)):
        fitnessListToAdd = listOfAvgFitnesses[i]
        avgDf[str(i)] = fitnessListToAdd
        
    #return results
    return bestDf, avgDf

def EP(panelParams, popSize, pm, numMutations, fitnessCountLimit, attempts, checkIteration, 
           conflictDeduction, q=10, toPrint=True):
    """
    inputs - panelParams - list of parameters - of form [l, w, maxCuts, res, zMin, zMax, tMin, tMax, aMin, 
                                                            aMax, zThresh, tThresh, aThresh]
           - popSize - int - size of a population
           - pm - float between 0.0 and 1.0 - probability that a gene will be mutated
           - numMutations - int - the number of parameters of an individual that could be mutated in one iteration
           - fitnessCountLimit - int - the max number of fitness evaluations
           - attempts - int - the number of attempts/number of times the whole process is done
           - checkIteration - int - how often you check the avg fitness and best individual
           - conflictDeductions - int - the amount that should be deduced from fitness if there is a conflict
            - q - int - the number of other individuals a indiviudal will compete against
           - toPrint - boolean -  if true, then results will be printed as function runs
    outputs - dfBest - pandas dataframe storing the best individual and their fitnesses
            - dfAvg - pandas dataframe storing the average fitness and the associated iteration for each attempt
    """
    l = panelParams[0]
    w = panelParams[1]

    #cut parameters
    maxCuts = panelParams[2]
    res = panelParams[3]

    #param min/max values
    zMin = panelParams[4]
    zMax = panelParams[5]

    tMin = panelParams[6]
    tMax = panelParams[7]

    aMin = panelParams[8]
    aMax = panelParams[9]

    #thresholds
    zThresh = panelParams[10] #inches
    tThresh = panelParams[11] #degrees
    aThresh = panelParams[12] #degrees

    #parameters for pvlib
    #Athens, Ga Lat/long
    latitude = 33.957409
    longitude = -83.376801

    #Creating DateTimeIndex
    df = pd.DataFrame({'date': ['2012-07-01 11:00:00','2012-07-01 12:00:00',
                                '2012-07-01 13:00:00','2012-07-01 14:00:00','2012-07-01 15:00:00','2012-07-01 16:00:00'
                               '2012-07-01 17:00:00','2012-07-01 18:00:00','2012-07-01 19:00:00'],
                      }).set_index('date')

    #Creating DatetimeIndex Object
    df.index = pd.DatetimeIndex(df.index)


    #Needed paramters for the basic chain model
    #These are constant through the whole simulation
    cec_mod = pvsystem.retrieve_sam('SandiaMod')
    cec_inv = pvsystem.retrieve_sam('CECInverter')
    sapm = cec_mod['Advent_Solar_AS160___2006_']
    cec = cec_inv['ABB__MICRO_0_3_I_OUTD_US_208_208V__CEC_2014_']
    
    #initializing the fitness evaluations
    currFitCount = 0
    
    #initialize the list of best fitnesses and individuals
    bestIndList = []
    bestFitnessList = []

    #initialize the list if of of average fitnesses
    listOfAvgFitnessLists = []
    for i in range(attempts): 

        ###initialization
        
        #initialize the list of average fitnesses
        avgFitnessList = []

        #initial population
        pop = makePopulation(popSize, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res)
        
        #initial fitness list
        fitnessList, currFitCount = getPopFitness(pop, l, w, zMin, zMax, tMin, tMax, aMin, aMax, currFitCount, 
                                    sapm, cec, df, latitude, longitude, res, zThresh, tThresh, aThresh, conflictDeduction)
        
        #print("initial fitnessList: " + str(fitnessList))
        #initialize the list of sigmaLists- each individual has a sigmaList
        sigmaLists = initializeSigmaList(panelParams, pop)
        
        
        ###iterative part
        
        #initialize the best individual and fitness
        bestGlobalInd = []
        bestGlobalFitness = 0.0

        ##begin to iterate through

        iteration = 0
        
        #initialize var for termination 
        
        countToTermination = 200
        
        while currFitCount <= fitnessCountLimit:
            
            #create child population
            childPop = []
            #create child sigma lists
            childSigmaLists = []
            #create child fitness list
            childFitnessList = []
            
            ##parent selection
            #for every individual in the population- make a mutated child
            #print("start pop: " + str(pop))
            for i in range(len(pop)):
                individual = pop[i]
                sigmaList = sigmaLists[i]
                
                #get mutated child and sigmaList
                child, newSigmaList = mutationGuassianList(individual, panelParams, pm, sigmaList)
                #add mutated child to list
                childPop.append(child)
                #add mutated sigma List to list
                childSigmaLists.append(newSigmaList)
                #add mutated child's fitness to fitnessList
                numConflicts = conflictCount(child, l, w, zMin, zMax, tMin, tMax, aMin, aMax, res, zThresh, tThresh, aThresh)
                childFitness, currFitCount = getFitness(child, numConflicts, l, w, sapm, cec, 
                                              df, latitude, longitude, currFitCount, res, conflictDeduction)
                childFitnessList.append(childFitness)
                
            ##survival selection
            
            #combine older generation and new generation
            
            totalPop = pop + childPop
            totalSigmaLists = sigmaLists + childSigmaLists
            totalFitnessList = fitnessList + childFitnessList
            pointList = []
            
            ##give each individual some points
            #for every individual in totalPop
            for i in range(len(totalPop)):
                points = 0
                ind = totalPop[i]
                indFitness = totalFitnessList[i]
                #compare ind to q other individuals
                for j in range(q):
                    randIndex = random.randint(0, len(totalPop)-1)
                    randInd = totalPop[randIndex]
                    randFitness = totalFitnessList[randIndex]
                    if indFitness > randFitness:
                        points += 1
                pointList.append(points)
                
                
            #get the popSize highest scoring individuals
            listOfHighestPoints = nlargest(popSize, pointList)
            
            #get the higest scoring individuals
            for i in range(len(listOfHighestPoints)):
                point = listOfHighestPoints[i]
                index = pointList.index(point)
                #make new population and things
                pop[i] = totalPop[index]
                fitnessList[i] = totalFitnessList[index]
                sigmaLists[i] = totalSigmaLists[index]
                
                #reset this point value
                listOfHighestPoints[i] = -1
            
            
            if (iteration % checkIteration) == 0:
                avgFitness = getAvgPopFitness(fitnessList)
                bestInd, bestFitness = getBest(pop, fitnessList)
                
                if toPrint == True:
                    print("iteration " + str(iteration))
                    print("current average fitness: " + str(avgFitness))
                    print("  ")
                    
                #append avg fitness to a list
                avgFitnessList.append(avgFitness)
                
                countToTermination -= 1
                
                if bestFitness > bestGlobalFitness:
                    bestGlobalInd = bestInd
                    bestGlobalFitness = bestFitness
                    countToTermination = 200
                    
                if countToTermination <= 0:
                    currFitCount = 1000000000
                    print("termination condition reached")
            iteration += 1

        if toPrint == True:
            print("finished")
            print("total number of iterations: " + str(iteration))
            print("best individual: " + str(bestGlobalInd))
            print("best fitness: " + str(bestGlobalFitness))

        bestIndList.append(bestGlobalInd)
        bestFitnessList.append(bestGlobalFitness)
        listOfAvgFitnessLists.append(avgFitnessList)

    if toPrint == True:
        print("  ")
        print("list of best individuals found for each attempt: " + str(bestIndList))
        print("list of best fitnesses found for each attempt: " + str(bestFitnessList))
        print("list of average fitness lists: " + str(listOfAvgFitnessLists))
        print(" ")

    #make pandas dataframe for best individuals and their fitnesses
    bestDf = pd.DataFrame()
    #add columns
    bestDf['best individuals'] = bestIndList
    bestDf['best Fitnesses'] = bestFitnessList

    #make pandas dataframe for average fitness values
    avgDf = pd.DataFrame()
    
    #add colums
    #make iteration List
    iterationList = []
    for i in range(len(listOfAvgFitnessLists[0])):
        iterationList.append(i*checkIteration)
    
    #make iteration column
    avgDf['iterations'] = iterationList
    #make avg fitness columns
    for i in range(len(listOfAvgFitnessLists)):
        fitnessListToAdd = listOfAvgFitnessLists[i]
        avgDf[str(i+1)] = fitnessListToAdd
        
    #return results
    return bestDf, avgDf

