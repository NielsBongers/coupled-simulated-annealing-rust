use crate::{
    simulated_annealing::{
        simulated_annealing::coupled_simulated_annealing, AnnealingSchedules, CoupledSAMethods,
    },
    utils::utils::f64_generation_function,
};

use std::io::Write;
use std::{
    fs::{create_dir_all, OpenOptions},
    path::Path,
};

pub fn ackley(x: &Vec<f64>) -> f64 {
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

pub fn benchmark(
    x_0: Vec<f64>,
    temperature_0: f64,
    coupled_sa_method: CoupledSAMethods,
    annealing_schedule: AnnealingSchedules,
    max_number_threads: usize,
    total_evaluations: i64,
    number_runs: i64,
    total_evaluations_overall: bool,
) {
    create_dir_all(Path::new("results")).expect("Failed to create results folder");

    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open("results/ackley_benchmark.csv")
        .expect("Failed to open or create file");

    // Write the CSV header
    writeln!(file, "number_threads,max_iterations,performance")
        .expect("Failed to write header to file");

    for number_threads in 1..=max_number_threads {
        let mut performance_vector = Vec::<f64>::new();

        let max_iterations = if total_evaluations_overall {
            total_evaluations / number_threads as i64
        } else {
            total_evaluations
        };

        for _ in 0..number_runs {
            let x = coupled_simulated_annealing(
                f64_generation_function,
                ackley,
                coupled_sa_method.clone(),
                annealing_schedule.clone(),
                x_0.clone(),
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
