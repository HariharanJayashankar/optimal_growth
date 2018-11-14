# Scripts Description

1. analytic_bellman.py - Has the function which defines a bellman with a defined value function.
2. bellman.py - Has the function which updates the bellman one step. Basically has the 1 step mapping Tw for a guess w.
3. comparing_bellmans.py - This compares visually the iterated solution and the true solution, since we actually know what the true solution to the analytic bellman is
4. interp.py - Just a small script which goes through what interpolation does visually in a 2 dimensional space
5. value_iter.py - Actually goes through solving the bellman problem in "analytic_bellman.py" numerically.

These aren't very well structured and some of them have overlapping functions. All these served was to properly structure my thought process when figuring this out.

## General Flow

I'm going to try and show what the flow is these scripts and in turn also show what the flow of solving the bellman looks like in this case.

1. Define a problem to which we know the solution - Most bellman problems don't have analytical solutions, but we have a couple of cases which do. I employ one here just as a benchmark
2. Define a bellman updater - Theoretically we want to follow the Contract Mapping theorem by using the Bellman equation as an operator on a vector w. We want to define how we would carry out 1 step in this updating process.
3. Visually see if the code works - We want to see if we even start converging to any solution. This is mostly preliminary.
4. Actually solve the bellman - In essence we wrap the bellman updater in a loop to get to the true value solution.
