def makeCutBitString(l, w, maxCuts, res=1):
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
    numBits = int(round(((l-1)+(w-1)/res)))

    #fill the list with 0's
    bitList = [0 for i in range(numBits)]

    #get the number of cuts
    numCuts = random.randint(1, maxCuts)

    #Lets put in the 1's that represent the cuts
    for i in range(numCuts):
        val = random.randint(0,numBits-1)

        #make sure that it doesn't replace a value in bitstring which is already a 1
        while bitList[val] == 1:
            val = random.randint(0,numBits)

        #replace the appropriate 0 with a 1
        bitList[val] = 1
        
    #return final bitstring
    return bitList

def getFaceListBitString(l, w, aBitString, res=1):
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
    if len(aBitString) != int(round(((l-1)+(w-1)/res))):
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

def makeIndividual(l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=1):
    """
    MAKES AN INDIVIDUAL
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
    output- a list of ints - individual whose first ((l-1)+(w-1))/res elements are bits representing where a cut is made
            the other ((maxCuts/2)+1)*((maxCuts/2)+1) elements (assuming maxCuts is even) represent the z, theta, and alpha 
            values. It looks like: [bit1 bit2 bit2 ... z1, t1, a1, z2, t2, a2...]
    """
    # error check to see if maxCuts can be defined
    if (maxCuts % 2) == 0:
        maxFaces = ((maxCuts/2)+1)*((maxCuts/2)+1)
    else:
        return "ERROR: You need to figure out how many maxFaces will result when there is an odd number of maxCuts"
    
    aBitString = makeCutBitString(l, w, maxCuts, res=1)
    
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

def makePopulation(popSize, l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=1):
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
        individ = makeIndividual(l, w, maxCuts, zMin, zMax, tMin, tMax, aMin, aMax, res=1)
        population.append(individ)
        
    return population

def getFaceList(individ, l, w, res=1):
    """
    GETS LIST OF FACES AND MEASUREMENTS/AREAS
    inputs- individ - list of ints - an individual
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
    output- a list of tuples where each tuple has 3 ints representing the info for each face. 
            The ints are the length, width, and area of the face. The whole list contains all of the faces
    """
    bitStringLen = int(round(((l-1)+(w-1)/res)))
    aBitString = individ[:bitStringLen]
    faceList = getFaceListBitString(l, w, aBitString, res=1)
    
    return faceList

def getFitness(individ, l, w, sapm, cec, df, latitude, longitude, currFitCount, res=1):
    '''
    TO BE USED TO GRAB THE FITNESS OF AN INDIVIDUAL
    USED AS A HELPER FUNCTION TO CREATE THE FITNESS OF THE POPULATION
    GETS THE FITNESS OF THE INDIVIDUAL
    inputs- individ - list of ints - an individual
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            sapm - the dataframe of coefficients produced by the sapm model
            cec - the datafram of coefficients produced by the cec model
            df - a dataframe of datetimeindexes used to tell pvlib aat which times to
                calculate the powers
            currFitCount - int - the current number of fitness evalutations (termination criteria)
    Output- float - the scaled fitness value for the individual
          - int - the currFitCount + 1
    '''


    #Dropping the cuts
    bitStringLen = int(round(((l-1)+(w-1))/res))
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
        
        indFitness += facePower*(face[2]/(l*w))
        #v_mp is maximum voltage
        #p_mp is maximum power
        
        face_position += 3
        
    currFitCount += 1 
    return indFitness, currFitCount
    

def getPopFitness(population, l, w, sapm, cec, df, currFitCount,res=1):
    '''
    CALCULATES THE FITNESS OF THE POPULATION
        inputs- individ - list of ints - an individual
            l- int - length of the initial solar panel
            w- int - width of the initial solar panel
            sapm - the dataframe of coefficients produced by the sapm model
            cec - the datafram of coefficients produced by the cec model
            df - a dataframe of datetimeindexes used to tell pvlib aat which times to
                calculate the powers
            currFitCount - int - the current number of fitness evalutations (termination criteria)
            res- int - resolution of the cut. i.e. the cut will happen every 'res' inches
        outputs - list - the populations' fitness values
                - int - the updated count of currFitCount
    '''
    
    
    popFitness = []
    
    for ind in population:
        fitness, currFitCount = getFitness(ind, l, w, sapm, cec, df, currFitCount, res)
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
    paramList = individ1[bitStringLen:]
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