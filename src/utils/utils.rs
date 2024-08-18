use crate::utils::DistributionType;
use rand::prelude::*;

pub fn floating_distributions(rng: &mut ThreadRng, distribution_type: DistributionType) -> f64 {
    match distribution_type {
        DistributionType::Uniform => 1.0 - 2.0 * rng.gen::<f64>(),
        DistributionType::Normal => todo!("Haven't implemented inverse transform sampling yet."),
    }
}
