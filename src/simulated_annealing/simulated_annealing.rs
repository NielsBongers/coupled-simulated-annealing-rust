use crate::simulated_annealing::AcceptanceFunctions;
use rand::prelude::*;

use super::AnnealingSchedules;

fn generation<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    x: &T,
    rng: &mut ThreadRng,
) -> T {
    generation_function(&x, rng)
}

fn energy<T>(energy_function: fn(&T) -> f64, x: &T) -> f64 {
    energy_function(x)
}

fn acceptance_probability(
    acceptance_function: &AcceptanceFunctions,
    energy_x: f64,
    energy_y: f64,
    temperature: f64,
) -> f64 {
    match acceptance_function {
        AcceptanceFunctions::Metropolis => f64::exp((energy_x - energy_y) / temperature),
        AcceptanceFunctions::Logistic => 1.0 / (1.0 + f64::exp(energy_y - energy_x) / temperature),
    }
}

fn acceptance<T>(
    energy_function: fn(&T) -> f64,
    acceptance_function: &AcceptanceFunctions,
    rng: &mut ThreadRng,
    x: T,
    y: T,
    temperature: f64,
) -> T {
    let energy_x = energy(energy_function, &x);
    let energy_y = energy(energy_function, &y);

    if energy_y <= energy_x
        || acceptance_probability(acceptance_function, energy_x, energy_y, temperature)
            > rng.gen::<f64>()
    {
        y
    } else {
        x
    }
}

fn update_temperature(
    annealing_schedule: &AnnealingSchedules,
    iteration: i64,
    temperature: f64,
    temperature_0: f64,
) -> f64 {
    match annealing_schedule {
        AnnealingSchedules::Exponential(gamma) => gamma * temperature,
        AnnealingSchedules::Fast() => temperature_0 / iteration as f64,
    }
}

pub fn anneal<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    acceptance_function: AcceptanceFunctions,
    annealing_schedule: AnnealingSchedules,
    x_0: T,
    temperature_0: f64,
    max_iterations: u64,
) -> T {
    let mut rng = rand::thread_rng();

    let mut iteration = 0;
    let mut x = x_0;
    let mut temperature = temperature_0;

    for _ in 0..max_iterations {
        let y = generation(generation_function, &x, &mut rng);

        temperature =
            update_temperature(&annealing_schedule, iteration, temperature, temperature_0);

        x = acceptance(
            energy_function,
            &acceptance_function,
            &mut rng,
            x,
            y,
            temperature,
        );

        iteration += 1;
    }

    x
}
