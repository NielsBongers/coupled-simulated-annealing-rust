# Rust implementation of Coupled Simulated Annealing

![Rust Main](https://github.com/NielsBongers/rust-file-indexing/actions/workflows/rust.yml/badge.svg?branch=main&event=push)

## Overview 

This is a simple Rust implementation of the paper [<i>Coupled Simulated Annealing</i>](https://ieeexplore.ieee.org/document/5184877) by Xavier-de-Sousa et al. Compared to regular parallel Simulated Annealing (SA) implementations, instead of running multiple in parallel to explore the search space, the parallel runs are instead coupled together through a parameter $\gamma$, which modifies the acceptance probability, hence the name Coupled Simulated Annealing (CSA)

In regular SA, we have $0 \leq A(x \to y) \leq 1$ as the acceptance probability. In CSA, we have $0 \leq A_\Theta (\gamma, x_i \to y_i) \leq 1$. The parameter $\gamma$ is dependent on the energies of the set of current states $\Theta \equiv \{x_i\}_{i=1}^m$, so $\gamma = f [E(x_1, E(x_2), \ldots, E(x_m)]$. 

Essentially, the coupling term indicates the energy levels of the overall population across all coupled systems (the parallel runs), which gives a better indication the energy level of a given state $x_i$ and hence whether $x_i \to y_i$ should be accepted. 

The paper derives a number of methods: CSA-MuSA, -BA, -M and -MwVC. I have implemented the first three in the code. 

## Performance 

On the [Ackley test function](https://en.wikipedia.org/wiki/Ackley_function) in three dimensions, with a total of 40.000 evaluations across all threads used, using more threads seems to certainly help, despite there being fewer function evaluations per individual thread. The information sharing between threads is having a noticeable effect. 

<img src="images/18082024 - Coupled simulated annealing - performance comparison - 40.000 total.png" width="600" alt="Example usage of the tool">

## Features 

The underlying algorithm is implemented entirely using generics. By defining a generation function (which takes the current state, the temperature, and a random number generator object), you are free to define a sampling process that generates a new object of the same type. Usually, that would be a `Vec<f64>` object, but the code is written in a way that categoricals will also work, because I wanted this to work for a travelling salesman/[vehicle routing-type problem](https://en.wikipedia.org/wiki/Vehicle_routing_problem) too. 

Following sampling, the current state $x$ and new state $y$ have their energies calculated using a generic function that maps $x, y \to \mathbb R$. Using those, the SA code is ran and $\gamma$ is shared between the parallel processes. After running, the best-performing (minimum-energy) samples are returned per thread and the best of _those_ is returned by the `coupled_simulated_annealing` function. 

The code includes three of the methods derived in the paper, their associated method for calculating $\gamma$, and a few standard annealing schedules: 

- Exponential: $T^{k+1} = \gamma t^{(k)}$
- Fast annealing: $T^{k+1} = t^{(1)} / k$
- Logarithmic: $T^{(1)} \cdot \ln (2) / \ln (k + 1)$

Additionally, there is a benchmark function with some Python code included to evaluate different parameters. After modifications, be sure to test the code with ```cargo test```. 

## Installation 

Simply clone the repo and run ```cargo run --release```. ```main.rs``` has an example ready using some convenience functions, and can directly be modified. 