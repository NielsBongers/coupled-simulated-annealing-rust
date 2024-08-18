use simulated_annealing::simulated_annealing::{AnnealingSchedules, CoupledSAMethods};
use simulated_annealing::utils::benchmarking::ackley;
use simulated_annealing::utils::utils::f64_generation_function;

use simulated_annealing::simulated_annealing::simulated_annealing::coupled_simulated_annealing;

fn main() {
    let total_evaluations = 40000;
    let number_threads = 10;

    let max_iterations = total_evaluations / number_threads;
    let number_threads = number_threads as usize;

    let x_0: Vec<f64> = vec![10.0, 10.0, 10.0];
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

    println!("Performance: {}", performance);
}
