use rand::prelude::*;

use simulated_annealing::simulated_annealing::{AnnealingSchedules, CoupledSAMethods};
use simulated_annealing::utils::utils::floating_distributions;
use simulated_annealing::utils::DistributionType;

use simulated_annealing::simulated_annealing::simulated_annealing::coupled_simulated_annealing;

fn f64_generation_function(x: &Vec<f64>, rng: &mut ThreadRng) -> Vec<f64> {
    let vector_length = x.len();

    let random_perturbations: Vec<f64> = (0..vector_length)
        .map(|_| floating_distributions(rng, DistributionType::Uniform))
        .collect();

    let x_prime: Vec<f64> = x
        .iter()
        .zip(random_perturbations.iter())
        .map(|(x, epsilon)| x + epsilon)
        .collect();

    x_prime
}

fn f64_energy_function(x: &Vec<f64>) -> f64 {
    let target_function: Vec<f64> = vec![0.0; x.len()];

    x.iter()
        .zip(target_function)
        .map(|(x, target)| f64::powi(x - target, 2))
        .sum()
}

fn ackley(x: &Vec<f64>) -> f64 {
    use std::f64::consts::E;
    use std::f64::consts::PI;

    let a = 20.0;
    let b = 0.2;
    let c = 2.0 * PI;

    let n = x.len() as f64;
    let sum1: f64 = x.iter().map(|&xi| xi.powi(2)).sum();
    let sum2: f64 = x.iter().map(|&xi| (c * xi).cos()).sum();

    let term1 = -a * (-b * (sum1 / n).sqrt()).exp();
    let term2 = -(sum2 / n).exp();

    term1 + term2 + a + E
}

fn main() {
    // coupled_simulated_annealing(5);

    let x_0 = vec![5.0, 5.0, 5.0];
    let temperature_0 = 1.0;

    let coupled_sa_method = CoupledSAMethods::CSA_MuSA;
    let annealing_schedule = AnnealingSchedules::Fast();
    let max_iterations: i64 = 100000;
    let number_threads = 20;

    let x = coupled_simulated_annealing(
        f64_generation_function,
        ackley,
        coupled_sa_method,
        annealing_schedule,
        x_0,
        temperature_0,
        max_iterations,
        number_threads,
    );

    // let x = anneal(
    //     f64_generation_function,
    //     f64_energy_function,
    //     acceptance_function,
    //     annealing_schedule,
    //     x_0,
    //     temperature_0,
    //     max_iterations,
    // );

    println!("x: {:?}", x);
}
