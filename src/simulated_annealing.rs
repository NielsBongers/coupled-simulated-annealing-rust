pub mod simulated_annealing;

#[derive(Clone)]
pub enum AcceptanceFunctions {
    MuSA,
    BA,
}

#[derive(Clone)]
pub enum AnnealingSchedules {
    Exponential(f64),
    Fast(),
    Logistic,
}
