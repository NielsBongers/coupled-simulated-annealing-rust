use crate::simulated_annealing::AcceptanceFunctions;
use core::f64;
use rand::prelude::*;
use std::{
    result,
    sync::{Arc, Barrier, Mutex},
    thread::{self, JoinHandle},
};

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
fn energy<T>(energy_function: fn(T) -> f64, x: T) -> f64 {
    energy_function(x)
}

/// Returns the acceptance probability for two given states based on common functions.
fn acceptance_probability(
    acceptance_function: &AcceptanceFunctions,
    energy_x: f64,
    energy_y: f64,
    gamma: f64,
    temperature: f64,
) -> f64 {
    match acceptance_function {
        // AcceptanceFunctions::Metropolis => f64::exp((energy_x - energy_y) / temperature),
        // AcceptanceFunctions::Logistic => {
        //     1.0 / (1.0 + f64::exp((energy_y - energy_x) / temperature))
        // }
        AcceptanceFunctions::MuSA => {
            f64::exp(-energy_y / temperature) / (f64::exp(-energy_y / temperature) + gamma)
        }
        AcceptanceFunctions::BA => 1.0 - (f64::exp(-energy_x / temperature)) / gamma,
    }
}

fn coupling_term(
    acceptance_function: &AcceptanceFunctions,
    energy_x: f64,
    temperature: f64,
) -> f64 {
    match acceptance_function {
        AcceptanceFunctions::MuSA => f64::exp(-energy_x / temperature),
        AcceptanceFunctions::BA => f64::exp(-energy_x / temperature),
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

pub fn coupled_simulated_annealing<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    acceptance_function: AcceptanceFunctions,
    annealing_schedule: AnnealingSchedules,
    x_0: T,
    temperature_0: f64,
    max_iterations: i64,
    number_threads: usize,
) -> T
where
    T: Clone + std::marker::Send + 'static,
{
    // This is gamma in Xavier-de-Sousa's paper. This is a mutex-protected vector where all the threads write their coupling contributions to.
    // The memory barrier ensures they all finish writing the current iteration before they start reading again.
    // They can never overwrite because of the mutexes but they could desync.
    let coupling_terms: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(vec![0.0; number_threads]));
    let memory_barrier = Arc::new(Barrier::new(number_threads));

    let mut handles = Vec::<JoinHandle<T>>::new();

    // Creating all the threads. The anneal(...) function returns the type T result per thread.
    for thread_id in 0..number_threads {
        let coupling_terms: Arc<Mutex<Vec<f64>>> = Arc::clone(&coupling_terms);
        let memory_barrier_clone = Arc::clone(&memory_barrier);

        let acceptance_function_clone = acceptance_function.clone();
        let annealing_schedule_clone = annealing_schedule.clone();
        let x_0_clone = x_0.clone();

        let handle = thread::spawn(move || {
            anneal(
                generation_function,
                energy_function,
                acceptance_function_clone,
                annealing_schedule_clone,
                coupling_terms,
                memory_barrier_clone,
                thread_id,
                x_0_clone,
                temperature_0,
                max_iterations,
            )
        });

        handles.push(handle);
    }

    // Taking all the handles, joining and unwrapping them, getting their best states out, then finding the minimum energy, and returning the associated state.
    handles
        .into_iter()
        .map(|handle| {
            let x = handle.join().unwrap();
            let energy_x = energy(energy_function, &x);
            (x, energy_x)
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("Unwrapping the results failed.")
        .0
}

/// Performs simulated annealing.
/// A generic generation function which creates new probing states, an energy function determining the energy level of the states, and an acceptance function and annealing schedule have to be passed, as well as the initial temperature and state, and the maximum number of iterations.
pub fn anneal<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    acceptance_function: AcceptanceFunctions,
    annealing_schedule: AnnealingSchedules,
    coupling_terms: Arc<Mutex<Vec<f64>>>,
    memory_barrier: Arc<Barrier>,
    thread_id: usize,
    x_0: T,
    temperature_0: f64,
    max_iterations: i64,
) -> T
where
    T: Clone,
{
    // This is thread-local with different seeds per thread.
    let mut rng = rand::thread_rng();

    // Initializing the state and the temperature.
    let mut x = x_0;
    let mut temperature = temperature_0;

    // Tracking the best performance for this instance.
    let mut x_best = x.clone();
    let mut energy_best = f64::MAX;

    for iteration in 0..max_iterations {
        // New trial state.
        let y = generation(generation_function, &x, &mut rng);

        // Energies for the current and trial state.
        let energy_x = energy(energy_function, &x);
        let energy_y = energy(energy_function, &y);

        // Updating the coupling terms for the current thread.
        // This sets gamma based on the current acceptance function used (see Table 1 in Xavier-de-Sousa).
        coupling_terms.lock().unwrap()[thread_id] =
            coupling_term(&acceptance_function, energy_x, temperature);

        // Waiting for all the others to complete this step too.
        memory_barrier.wait();

        let gamma = coupling_terms.lock().unwrap().iter().sum();

        // Accepting directly if E(y) <= E(x), otherwise accepting if A(x, y) > r, with r randomly sampled.
        x = if energy_y <= energy_x
            || acceptance_probability(&acceptance_function, energy_x, energy_y, gamma, temperature)
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

        // Updating the temperature for the next iteration.
        temperature =
            update_temperature(&annealing_schedule, iteration, temperature, temperature_0);
    }

    x_best
}
