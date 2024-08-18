use core::f64;

use crate::simulated_annealing::AcceptanceFunctions;
use rand::prelude::*;

use crate::simulated_annealing::AnnealingSchedules;

/// Generates a stochastically modified version of the input state using a generic generation function.
fn generation<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    x: &T,
    rng: &mut ThreadRng,
) -> T {
    generation_function(&x, rng)
}

/// Calculates the energy for a given state.
fn energy<T>(energy_function: fn(&T) -> f64, x: &T) -> f64 {
    energy_function(x)
}

/// Returns the acceptance probability for two given states based on common functions.
fn acceptance_probability(
    acceptance_function: &AcceptanceFunctions,
    energy_x: f64,
    energy_y: f64,
    temperature: f64,
) -> f64 {
    match acceptance_function {
        AcceptanceFunctions::Metropolis => f64::exp((energy_x - energy_y) / temperature),
        AcceptanceFunctions::Logistic => {
            1.0 / (1.0 + f64::exp((energy_y - energy_x) / temperature))
        }
    }
}

/// Adjusts the temperature for a given annealing schedule based on the iteration, temperature, and initial temperature.
fn update_temperature(
    annealing_schedule: &AnnealingSchedules,
    iteration: i64,
    temperature: f64,
    temperature_0: f64,
) -> f64 {
    match annealing_schedule {
        AnnealingSchedules::Exponential(gamma) => gamma * temperature,
        AnnealingSchedules::Fast() => temperature_0 / iteration as f64,
        AnnealingSchedules::Logistic => {
            temperature_0 * f64::ln(2.0) / f64::ln(iteration as f64 + 1.0)
        }
    }
}

/// Performs simulated annealing.
/// A generic generation function which creates new probing states, an energy function determining the energy level of the states, and an acceptance function and annealing schedule have to be passed, as well as the initial temperature and state, and the maximum number of iterations.
pub fn anneal<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    acceptance_function: AcceptanceFunctions,
    annealing_schedule: AnnealingSchedules,
    x_0: T,
    temperature_0: f64,
    max_iterations: i64,
) -> T
where
    T: Clone,
{
    let mut rng = rand::thread_rng();

    let mut x = x_0;
    let mut x_best = x.clone();
    let mut energy_best = f64::MAX;

    let mut temperature = temperature_0;

    for iteration in 0..max_iterations {
        let y = generation(generation_function, &x, &mut rng);

        temperature =
            update_temperature(&annealing_schedule, iteration, temperature, temperature_0);

        let energy_x = energy(energy_function, &x);
        let energy_y = energy(energy_function, &y);

        // Accepting directly if E(y) <= E(x), otherwise accepting if A(x, y) > r, with r randomly sampled.
        x = if energy_y <= energy_x
            || acceptance_probability(&acceptance_function, energy_x, energy_y, temperature)
                > rng.gen::<f64>()
        {
            // Only checked if E(y) <= E(x), which the best will necessarily also be. Reduces expensive clones.
            if energy_y < energy_best {
                energy_best = energy_y;
                x_best = y.clone();
            }
            y
        } else {
            x
        };
    }

    x_best
}
