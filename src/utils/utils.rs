use crate::utils::DistributionType;
use rand::prelude::*;

/// Simple distributions for use in the generation code.
pub fn floating_distributions(rng: &mut ThreadRng, distribution_type: DistributionType) -> f64 {
    match distribution_type {
        DistributionType::Uniform => 1.0 - 2.0 * rng.gen::<f64>(),
        DistributionType::Cauchy => std::f64::consts::PI * (rng.gen::<f64>() - 0.5).tan(),
    }
}

/// Simple generation function for f64s, taking a vector and returning a slightly adjusted one. This samples a random value from Cauchy and multiplies that by the temperature to get some basic form of temperature dependence (less movement when temperature is lower).
pub fn f64_generation_function(x: &Vec<f64>, temperature: f64, rng: &mut ThreadRng) -> Vec<f64> {
    let vector_length = x.len();

    let random_perturbations: Vec<f64> = (0..vector_length)
        .map(|_| (temperature) * (floating_distributions(rng, DistributionType::Cauchy)))
        .collect();

    let y: Vec<f64> = x
        .iter()
        .zip(random_perturbations.iter())
        .map(|(x, epsilon)| x + epsilon)
        .collect();

    y
}
