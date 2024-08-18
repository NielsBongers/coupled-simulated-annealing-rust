use crate::simulated_annealing::CoupledSAMethods;
use core::f64;
use rand::prelude::*;
use std::{
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
/// Different acceptance functions are supported: see Xavier-de-Sousa2010, Table 1.
/// These are connected to the coupling terms.
fn acceptance_probability(
    coupled_sa_method: &CoupledSAMethods,
    energy_x: f64,
    energy_y: f64,
    gamma: f64,
    temperature: f64,
) -> f64 {
    match coupled_sa_method {
        CoupledSAMethods::CSA_MuSA => {
            f64::exp(-energy_y / temperature) / (f64::exp(-energy_y / temperature) + gamma)
        }
        CoupledSAMethods::CSA_BA => 1.0 - (f64::exp(-energy_x / temperature)) / gamma,
        CoupledSAMethods::CSA_M => f64::exp(energy_x / temperature) / gamma,
    }
}

/// Calculates the coupling terms to calculate gamma with.
/// See Table 1 one Xavier-de-Sousa2010.
fn coupling_term(coupled_sa_method: &CoupledSAMethods, energy_x: f64, temperature: f64) -> f64 {
    match coupled_sa_method {
        CoupledSAMethods::CSA_MuSA => f64::exp(-energy_x / temperature),
        CoupledSAMethods::CSA_BA => f64::exp(-energy_x / temperature),
        CoupledSAMethods::CSA_M => f64::exp(energy_x / temperature),
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

/// Performs coupled simulated annealing over a specified number of threads.
/// Coupling occurs through the acceptance function and a coupling parameter gamma.
/// See Xavier-de-Sousa2010 for algorithm details.
pub fn coupled_simulated_annealing<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    coupled_sa_method: CoupledSAMethods,
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

    // Instantiating the handles.
    let mut handles = Vec::<JoinHandle<(T, f64)>>::new();

    // Creating all the threads. The anneal(...) function returns the type T result per thread.
    for thread_id in 0..number_threads {
        let coupling_terms: Arc<Mutex<Vec<f64>>> = Arc::clone(&coupling_terms);
        let memory_barrier_clone = Arc::clone(&memory_barrier);

        let coupled_sa_method_clone = coupled_sa_method.clone();
        let annealing_schedule_clone = annealing_schedule.clone();
        let x_0_clone = x_0.clone();

        let handle = thread::spawn(move || {
            anneal(
                generation_function,
                energy_function,
                coupled_sa_method_clone,
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
        .map(|handle| handle.join().unwrap())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .expect("Unwrapping the results failed.")
        .0
}

/// Performs simulated annealing.
/// A generic generation function which creates new probing states, an energy function determining the energy level of the states, and an acceptance function and annealing schedule have to be passed, as well as the initial temperature and state, and the maximum number of iterations.
pub fn anneal<T>(
    generation_function: fn(&T, &mut ThreadRng) -> T,
    energy_function: fn(&T) -> f64,
    coupled_sa_method: CoupledSAMethods,
    annealing_schedule: AnnealingSchedules,
    coupling_terms: Arc<Mutex<Vec<f64>>>,
    memory_barrier: Arc<Barrier>,
    thread_id: usize,
    x_0: T,
    temperature_0: f64,
    max_iterations: i64,
) -> (T, f64)
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
        // This sets gamma based on the current coupled SA method used (see Table 1 in Xavier-de-Sousa).
        coupling_terms.lock().unwrap()[thread_id] =
            coupling_term(&coupled_sa_method, energy_x, temperature);

        // Waiting for all the others to complete this step too.
        memory_barrier.wait();

        let gamma = coupling_terms.lock().unwrap().iter().sum();

        // Accepting directly if E(y) <= E(x), otherwise accepting if A(x, y) > r, with r randomly sampled.
        x = if energy_y <= energy_x
            || acceptance_probability(&coupled_sa_method, energy_x, energy_y, gamma, temperature)
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

    // Returning this thread's best-performing state.
    (x_best, energy_best)
}
