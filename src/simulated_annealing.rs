pub mod simulated_annealing;

pub enum AcceptanceFunctions {
    Metropolis,
    Logistic,
}

pub enum AnnealingSchedules {
    Exponential(f64),
    Fast(),
    Logistic,
}
