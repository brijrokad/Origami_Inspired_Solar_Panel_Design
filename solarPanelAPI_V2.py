###imports
import numpy as np
import pandas as pd
import random
import math

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

    # make bit strings that represent just the length and the width of the solar panel respectively
    lenBitString = aBitString[0:l-1]
    widBitString = aBitString[l-1:]
    
    # initialize list of faces
    faceList = []

    ## get list of the positions where a cut will happen
    # vertical cuts
    vCuts = [i+1 for i in range(len(lenBitString)) if lenBitString[i] == 1]
    # horizontal cuts
    hCuts = [i+1 for i in range(len(widBitString)) if widBitString[i] == 1]

    ## initialize some variables
    vLow = 0
    vHigh = l
    hLow = 0
    hHigh = w

    # make lists storing at what inches chuts are being made
    vCuts = [0] + vCuts + [l]
    hCuts = [0] + hCuts + [w]

    # initalize some vars
    lengths = []
    widths = []

    # fill 'lengths' with the length of each face
    for i in range(len(vCuts)-1):
        vLow = vCuts[0]
        vHigh = vCuts[1]
        length = vHigh - vLow
        lengths.append(length)

        # update vCuts
        vCuts = vCuts[1:]

    # fill 'widths' with the width of each face
    for i in range(len(hCuts)-1):
        hLow = hCuts[0]
        hHigh = hCuts[1]
        width = hHigh - hLow
        widths.append(width)

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
            sapm - the dataframe of coefficients produced by the sapm model
            cec - the datafram of coefficients produced by the cec model
            latitude - float - the latitude to calculate the fitness value
            longitude - float - the longitude to calculate the fitness
            df - a dataframe of datetimeindexes used to tell pvlib aat which times to calculate the powers
            currFitCount - int - the current number of fitness evalutations (termination criteria)
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches           
            conflictDeduction - int - the amount an individual's fitness should be decreased for every conflict it has
    Output- indFitness - float - the scaled fitness value for the individual
            currFitCount - int - the updated number of fitness evalutations (termination criteria)
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
    """
    inouts - population - a list of lists of integers. Each list of integers is an individual.
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            zMin - int - min possible value of z
            zMax - int - min possible value of z
            tMin - int - min possible value of t
            tMax - int - min possible value of t
            aMin - int - min possible value of a
            aMax - int - min possible value of a
            sapm - pandas data frame - the dataframe of coefficients produced by the sapm model
            cec - pandas data frame - the datafram of coefficients produced by the cec model
            latitude - float - the latitude to calculate the fitness value
            longitude - float - the longitude to calculate the fitness
            df - pandas data frame - a dataframe of datetimeindexes used to tell pvlib aat which times to calculate the powers
            currFitCount - int - the current number of fitness evalutations (termination criteria)
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
            zThresh- the threshold that describes if two solar panels are closer then zThresh inches, and the
                     other thresholds are passed, then that face will be mutated
            tThresh- the threshold that describes if two solar panels are closer then tThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
            aThresh- the threshold that describes if two solar panels are closer then aThresh degrees, and the
                     other thresholds are passed, then that face will be mutated
    output- popFitness - list of floats - the fitness for the entire population
            currFitCount - int - the updated number of fitness evalutations (termination criteria)
    """
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
            sapm - the dataframe of coefficients produced by the sapm model
            cec - the datafram of coefficients produced by the cec model
            df - a dataframe of datetimeindexes used to tell pvlib aat which times to
                calculate the powers
            latitude - float - the latitude to calculate the fitness value
            longitude - float - the longitude to calculate the fitness
             currFitCount - int - the current number of fitness evalutations (termination criteria)
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
            currFitCount - int - the updated number of fitness evalutations (termination criteria)
    
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
    #see if mutation is done
    check = random.uniform(0, 1)
    if check <= pm:
        #get point where mutation will happen
        mutationPoint = random.randint(0, len(individual)-1)

        #get the lenth of the bit string
        bitStringLen = int((l-1)/res) + int((w-1)/res)

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

def getBestInd(pop, fitnessPop):
    """
    inputs - pop - list of individuals - a list of all of the individuals
           - fitnessPop - list of floats - list of each indiviudal's fitness. The index in fitnessPop corresponds with the 
                index of pop.
    """
    #get max value
    maxFitness = max(fitnessPop)
    index = fitnessPop.index(maxFitness)
    
    #return solution
    return pop[index]