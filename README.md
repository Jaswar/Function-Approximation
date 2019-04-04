# Function-Approximation
Description: 
This repository contains a genetic algorithm to approximate drawn by hand function. What it does is it tries to find the most optimal function for the one we have drawn in a window. What we do first is we draw a function in a simple graphic designer window I have created. Next, my algoritm tries to describe this line mathematically. As said I used a genetic algorithm which consists of a population of polynomial functions, best of which are then mixed to create an offspring.

Required libraries:
- numpy

My specs:
- i7-7700HQ Processor
- 16GB RAM
- NVIDIA Quadro M1200 4GB graphics card

Training:
As mentioned what we do first is we draw a line in a window which is coloured in black. My code then starts to mathematically approximate this line to some polynomial equation. To do that it uses a genetic algorithm with a species feature. Species are separated from each other and contain a population of polynomial function. If we set variable called highestPower to some n, that means we have n + 1 populations. Population x consists of certain amount of polynomial functions, each with their highest power set to this x. So population 0 has function that can be described as f(x) = a, population 1 has functions like f(x) = ax + b and so on. From each population we then select certain number of best function which we then mix to create offsprings. Mixing happens in every species seperately, there is no mixing between species. After around 15-20 minutes my code can find really good approximations and after some more time it approximates them nearly perfectly. 
