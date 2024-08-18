use rand::prelude::*;

use simulated_annealing::simulated_annealing::{AcceptanceFunctions, AnnealingSchedules};
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

fn main() {
    // coupled_simulated_annealing(5);

    let x_0 = vec![1.0, 2.0, 3.0];
    let temperature_0 = 10.0;

    let acceptance_function = AcceptanceFunctions::MuSA;
    let annealing_schedule = AnnealingSchedules::Exponential(0.5);
    let max_iterations: i64 = 1000;
    let number_threads = 10;

    let x = coupled_simulated_annealing(
        f64_generation_function,
        f64_energy_function,
        acceptance_function,
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
