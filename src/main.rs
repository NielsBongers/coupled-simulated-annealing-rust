use std::fs::OpenOptions;
use std::io::Write;

use rand::prelude::*;

use simulated_annealing::simulated_annealing::{AnnealingSchedules, CoupledSAMethods};
// use simulated_annealing::utils::utils::floating_distributions;
// use simulated_annealing::utils::DistributionType;

use simulated_annealing::simulated_annealing::simulated_annealing::coupled_simulated_annealing;

fn f64_generation_function(x: &Vec<f64>, temperature: f64, rng: &mut ThreadRng) -> Vec<f64> {
    let vector_length = x.len();

    let random_perturbations: Vec<f64> = (0..vector_length)
        .map(|_| (temperature) * (std::f64::consts::PI * (rng.gen::<f64>() - 0.5)).tan())
        .collect();

    let x_prime: Vec<f64> = x
        .iter()
        .zip(random_perturbations.iter())
        .map(|(x, epsilon)| x + epsilon)
        .collect();

    x_prime
}

fn _f64_energy_function(x: &Vec<f64>) -> f64 {
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
    let total_evaluations = 40000;
    let number_runs = 100;

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("results/ackley_benchmark.csv")
        .expect("Failed to open or create file");

    // Write the CSV header
    writeln!(file, "number_threads,max_iterations,performance")
        .expect("Failed to write header to file");

    for number_threads in 1..=20 {
        let mut performance_vector = Vec::<f64>::new();

        let max_iterations = total_evaluations;
        // let max_iterations = total_evaluations / number_threads;
        let number_threads = number_threads as usize;

        for _ in 0..number_runs {
            let x_0 = vec![10.0, 10.0, 10.0];
            let temperature_0 = 5.0;

            let coupled_sa_method = CoupledSAMethods::CSA_MuSA;
            let annealing_schedule = AnnealingSchedules::Exponential(0.999);

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

            let performance: f64 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();

            writeln!(
                file,
                "{},{},{}",
                number_threads, max_iterations, performance
            )
            .expect("Failed to write results to file");

            performance_vector.push(performance);
        }

        let mean_performance: f64 = performance_vector.iter().sum::<f64>() / number_runs as f64;

        println!(
            "Threads: {}. Per thread: {}. Mean performance over {} runs: {}.",
            number_threads, max_iterations, number_runs, mean_performance
        )
    }
}
