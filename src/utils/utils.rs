use crate::utils::DistributionType;
use rand::prelude::*;

pub fn floating_distributions(rng: &mut ThreadRng, distribution_type: DistributionType) -> f64 {
    match distribution_type {
        DistributionType::Uniform => 1.0 - 2.0 * rng.gen::<f64>(),
        DistributionType::Normal => todo!("Haven't implemented inverse transform sampling yet."),
    }
}

pub fn f64_generation_function(x: &Vec<f64>, temperature: f64, rng: &mut ThreadRng) -> Vec<f64> {
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
