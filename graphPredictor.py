# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 16:46:23 2019

@author: janwa
"""

import tkinter as tk
import numpy as np
import math
import operator 
import sys

highestPower = 5            #highest possible power in every population, it also defines how many populations there are going to be (highestPower + 1)                                                           
coefficientsRange = [10, 6, 3.5, 1.5, 0.6, 0.3, 0.2, 0.05, 0.02, 0.005, 0.002] #initial range in which coefficients for each power can be created (-coeffcientsRange[power], +coefficientsRange[power])
nbPerSample = 1200          #it defines how big the values on screen are going to be 
WIDTH = 800                 #width of the window
HEIGHT = 800                #height of the window
YSCALE = 2                  #how much X-axis is going to be scaled
XSCALE = 2                  #how much Y-axis is going to be scaled
SIZE = 5                    #brush size
populationSize = 100        #how many functions are in each population
mutationRate = 0.1          #the chance that a mixed function is going to be completely random
nBestCopied = 10            #how many best functions we mix to create new population
nGenerations = 10000        #maximum number of generations
sampleDiff = 0.01           #we calulate difference for each funtion every sampleDifference x
mutationDifference = 0.3    #the range of coefficients mutation in each new function, new coefficient is inherited from one of the parents and then to this value a random value from this range is added



lowBound = -(nbPerSample * sampleDiff / 2)
YSCALE = int(YSCALE)
XSCALE = int(XSCALE)

if len(coefficientsRange) - 1 < highestPower:
    sys.exit('CHECK COEFFCIENTS')
    
coordsData = np.zeros((1,2))
lowPixBound = 0
uppPixBound = 0
root = tk.Tk()
canvas = tk.Canvas(root, width = WIDTH, height = HEIGHT)
singleDistPix = .0
generation = 0
species = list()
class Function(object):
    
    def __init__(self, nPower, cRange, mDiff):
        self.highestPower = nPower
        self.coefficients = np.zeros((nPower+1))
        self.coefficientRange = cRange
        self.difference = 0.0
        self.mutationDifference = mDiff
        
    def mix(self, c1, c2):
        for i in range(self.highestPower + 1):
            r = np.random.randint(0,2)
            if r == 0:
                self.coefficients[i] = c1[i] + (np.random.uniform(low = -self.mutationDifference, high = self.mutationDifference) * self.coefficientRange[i]) / max(self.coefficientRange)
            else:
                self.coefficients[i] = c2[i] + (np.random.uniform(low = -self.mutationDifference, high = self.mutationDifference) * self.coefficientRange[i]) / max(self.coefficientRange)
        return self.coefficients
    
    def createNewFunction(self):
        for power in range(self.highestPower + 1):
            self.coefficients[power] = np.random.uniform(low = -self.coefficientRange[power], high = self.coefficientRange[power])
            



def draw_line(event):
    global coordsData
    if str(event.type) == 'ButtonPress':
        canvas.old_coords = event.x, event.y
    elif str(event.type) == 'ButtonRelease':
        x,y = event.x, event.y
        x1,y1 = canvas.old_coords
        canvas.create_line(x,y,x1,y1, fill = 'black',width = SIZE)
    coordsData = np.concatenate((coordsData,[[event.x,event.y]]))
    
def draw2(event):
    global coordsData
    x, y = event.x, event.y

    coordsData = np.concatenate((coordsData,[[event.x,event.y]]))
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x,y,x1,y1, fill = 'black', width = SIZE)
    canvas.old_coords = x,y 
    
def reset_coords(event):
    global lowPixBound
    global uppPixBound
    global coordsData
    global generation
    
    
    species = list()
    for j in range(highestPower + 1):
        population = list()
        for i in range(populationSize):
            function = Function(highestPower - j, coefficientsRange, mutationDifference)
            function.createNewFunction()
            population.append(function)
        species.append(population)
    
    canvas.old_coords = None
    coordsData = np.delete(coordsData, 0, axis = 0)
    drawnLine = np.copy(coordsData)
    intXCount = int((-lowBound - lowBound)*XSCALE)
    xDist = WIDTH/intXCount
    intYCount = int((-lowBound - lowBound)*YSCALE)
    yDist = HEIGHT/intYCount
    lowPixBound = coordsData[0][0]
    for rec in range(len(coordsData)):
        coordsData[rec][0] = round((coordsData[rec][0] - WIDTH/2) / xDist,2)
        coordsData[rec][1] = (HEIGHT/2 - coordsData[rec][1]) / yDist
    
    convCoordsData = list() 
    
    for rec in range(len(coordsData) - 1):
        xDiff = int(round((coordsData[rec + 1][0] - coordsData[rec][0]) * 100,3))
        if xDiff < 0:
            coordsData = np.zeros((1,2))
            reset()
            return
        yDiff = coordsData[rec + 1][1] - coordsData[rec][1]
        
        for x in range(xDiff):
            convCoordsData.append([coordsData[rec][0] + x * sampleDiff,coordsData[rec][1] + x * (yDiff / xDiff)])
    coordsData = np.copy(np.array(convCoordsData))
    
    convCoordsData.clear()     
    
    import matplotlib.pyplot as plt
    plt.plot(coordsData[:,1])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    
    coordsData = coordsData[:,1]
    coordsData = np.reshape(coordsData, (1, len(coordsData)))
    bestDiff = 100000.
    realValues = np.zeros((len(coordsData[0]), ))
    generation = 0
    
    fLowerBound = (lowPixBound - WIDTH/2) / xDist
    while generation < nGenerations:
        bestSpecies = -1
        generation += 1
        for pop in range(highestPower + 1):
            population = species[pop].copy()
            
            for i in range(len(population)):
                
                population[i].difference = 0.
                for j in range(len(coordsData[0])):
                    x = fLowerBound + j * sampleDiff
                    fx = 0.
                    
                    for power in range(population[0].highestPower + 1):
                        fx += population[i].coefficients[power] * math.pow(x, power)
                        
                    population[i].difference += abs(fx - coordsData[0][j])
                    
            
            sortedPopulation = sorted(population, key = operator.attrgetter('difference'))
            print('Population number ' + str(highestPower - pop) + ' average best difference: {:.5f}'.format(sortedPopulation[0].difference / len(coordsData[0])))
            if sortedPopulation[0].difference <= bestDiff:
                message = 'f(x) = '
                for power in reversed(range(sortedPopulation[0].highestPower + 1)):
                    if power > 0:
                        message += '{:.05f}'.format(sortedPopulation[0].coefficients[power]) + 'x^' + str(power) + ' + '
                    else:
                        message += '{:.05f}'.format(sortedPopulation[0].coefficients[power])
                for i in range(len(coordsData[0])):
                    x = fLowerBound + i * sampleDiff
                    fx = .0
                    for power in range(sortedPopulation[0].highestPower + 1):
                        fx += sortedPopulation[0].coefficients[power] * math.pow(x, power)
                    realValues[i] = fx
                bestDiff = sortedPopulation[0].difference
                bestSpecies = pop
                    
            population.clear()
            
            for i in range(nBestCopied):
                population.append(sortedPopulation[i])
            
            for i in range(populationSize - nBestCopied):
                rnd = np.random.rand()
                function = Function(population[0].highestPower, coefficientsRange, mutationDifference)
                if rnd < mutationRate:
                    function.createNewFunction()
                else:
                    p1 = np.random.randint(0, nBestCopied)
                    p2 = np.random.randint(0, nBestCopied)
                    while p2 == p1:
                        p2 = np.random.randint(0, nBestCopied)
                    function.mix(sortedPopulation[p1].coefficients, sortedPopulation[p2].coefficients)
                population.append(function)
            species[pop] = population.copy()
        
        
        
        xOld = 0
        yOld = 0
        xNew = 0
        yNew = 0
        for i in range(len(coordsData[0])):
            if i == 0:
                yOld = int(HEIGHT/2 - realValues[i] * yDist)
                xOld = int(i * sampleDiff * xDist + lowPixBound)
                yNew = int(HEIGHT/2 - realValues[i] * yDist)
                xNew = int(i * sampleDiff * xDist + lowPixBound)
            else:
                yNew = int(HEIGHT/2 - realValues[i] * yDist)
                xNew = int(i * sampleDiff * xDist + lowPixBound)
                canvas.create_line(xOld, yOld, xNew, yNew, fill = 'blue', width = SIZE)
                
                xOld = xNew
                yOld = yNew
        
        
        if generation <= nGenerations:
            print('Generation: ' + str(generation) + ' Avg Best Error: {:.05f}'.format(bestDiff / len(coordsData[0])) + ' Best Species: ' + str(highestPower - bestSpecies))
            print('Function: ' + message)
            print('')
        root.update()
        if generation < nGenerations:
            reset()
            drawPreviousLine(drawnLine)
    if generation > nGenerations:
        reset()
    coordsData = np.zeros((1,2))
    
def reset():
    global lowPixBound
    global uppPixBound
    global coordsData
    global generation
    
    #coordsData = np.zeros((1,2))
    canvas.delete('all')
    
    canvas.create_line(WIDTH/2,0,WIDTH/2,HEIGHT)
    canvas.create_line(0,HEIGHT/2,WIDTH,HEIGHT/2)
    
    intXCount = int((-lowBound - lowBound)*XSCALE)
    xDist = WIDTH/intXCount
    for i in range(intXCount):
        canvas.create_line(i*xDist,HEIGHT/2+10,i*xDist,HEIGHT/2-10)
        if i == intXCount/2 + lowBound:
            canvas.create_text(i*xDist, HEIGHT/2+20, text = str(int(lowBound)))
            #lowPixBound = i*xDist
        elif i == intXCount/2 - lowBound:
            canvas.create_text(i*xDist, HEIGHT/2+20, text = str(int(-lowBound)))
            uppPixBound = i*xDist
    
    intYCount = int((-lowBound - lowBound)*YSCALE)
    yDist = HEIGHT/intYCount
    for i in range(intYCount):
        canvas.create_line(WIDTH/2+10,i*yDist,WIDTH/2-10,i*yDist)

def drawPreviousLine(coords):
    xOld = coords[0][0]
    yOld = coords[0][1]
    for i in range(1, len(coords)):
        xNew = coords[i][0]
        yNew = coords[i][1]
        canvas.create_line(xOld,yOld,xNew,yNew, fill = 'black', width = SIZE)
        xOld = xNew
        yOld = yNew

def stopGenerations():
    global generation
    generation = nGenerations + 1
    print('RESET')

def combineFunctions():
    stopGenerations()
    reset()
    
buttonReset = tk.Button(root, text = 'Reset', command = combineFunctions)
reset()
canvas.pack()
canvas.bind('<ButtonPress-1>', draw_line)
canvas.bind('<ButtonRelease-1>', draw_line)
canvas.bind('<B1-Motion>', draw2)
canvas.bind('<ButtonRelease-1>', reset_coords)
buttonReset.pack()
root.mainloop()
        
        
