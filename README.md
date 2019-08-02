# Origami_Inspired_Solar_Panel_Design
Optimized solar panel shape using genatic algorithms

## Approach
This repository uses Evolutionary Strategies to find the optimal design for a 3-D solar panel design. Origami can take a flat sheet of paper and make pretty unique shapes if folded correctly, so this takes that idea to make cuts and strategically place the smaller panels. We took a standard solar panel and broke it into pieces and placed it on an axle to see if a 3 dimensional topology will perform equivalently or better than a flat 2-dimensional panel. Flat solar panels can also use a motor to track the sun to get as much sun light as possible. Since the motor itself will consume some energy, this 3-D model could potentially make the need to track the sun obsolete. Ideally this Shown below is our best fit individual after 30 trials.

![alt-text](/3D Model/Capture4.pnAsg)

### Individual
In order to implement these evolutionary strategies, we had to represent the 3-D panel as an array of cut positions and angles. A standard solar panel is 65inX39in and allowed a cut to be made on every inch, which results in 64 cut points length wise and 38 cut points height wise to ignore making a cut at the edges. With a maximum of 16 individual sub-panels, we can create an array of 3 different sections. The first 16 positions of the Individual are 0 and 1s where a 1 indicates a cut along the length and height. The next part will be 16 groups of 3 numbers that represent the height position (z: 36in to 72in), the angle relative to the horizontal position (Theta: 0 to 90), and lastly the angle of the panel about the z-axis (Alpha: North, south, east, and west perspectives).

*Include an example of an individual*

By randomly making these individuals, it can cause for some incorrect individuals, so when individuals were created, we resolved any errors to enforce our max of 6 cuts (16 faces).
### Fitness and Conflicts
Our fitness function was implemented by the [pvlib.modelchain](https://pvlib-python.readthedocs.io/en/latest/generated/pvlib.modelchain.basic_chain.html#pvlib.modelchain.basic_chain). This experimental power model can take an array of times and locations. We kept it simple and used Athens, Ga [33.957409, -83.376801] with a day in August summed the power at each hour from 11:00AM to 7:00PM. Then the output was scaled to the area of each panel since the function treats each call for a standard size panel.

We had to have a conflict for any panels that were overlapping, thus reducing the amount of energy stored. We kept it simple and for the number of overlaps (conflicts) we deducted 50 Watts from the individual's overall fitness.

### Strategies
  * crossover
    * One-Point Crossover
    * Uniform Crossover
  * Mutation
  * Tournament Selection
  * replace worst
  * Steady State
  * Evolutionary Strategies
  * Evolutionary Programming

## API
  Add individual functions
  Add pipeline descriptions
