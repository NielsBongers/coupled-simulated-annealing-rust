use simulated_annealing::{
    simulated_annealing::{
        simulated_annealing::coupled_simulated_annealing, AnnealingSchedules, CoupledSAMethods,
    },
    utils::benchmarking::ackley,
    utils::utils::f64_generation_function,
};

#[test]
fn test_ackley() {
    let x_0 = vec![5.0, 5.0, 5.0];
    let temperature_0 = 5.0;

    let number_threads = 10;
    let max_iterations = 40000;

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

    // Ackley's function is defined as having its global minimum at (0, 0, 0), so this is just the distance from there.
    let performance: f64 = x.iter().map(|&xi| xi.powi(2)).sum::<f64>().sqrt();

    assert!(performance < 1e-5);
}
