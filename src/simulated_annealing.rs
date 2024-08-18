pub mod simulated_annealing;

#[allow(non_camel_case_types)]
#[derive(Clone)]
/// Acceptance functions from Table 1 of Xavier-de-Sousa2010.
pub enum CoupledSAMethods {
    CSA_MuSA,
    CSA_BA,
    CSA_M,
}

#[derive(Clone)]
pub enum AnnealingSchedules {
    Exponential(f64),
    Fast(),
    Logarithmic,
}
