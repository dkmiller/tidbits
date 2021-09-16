use clap::Clap;
use serde::{Deserialize, Serialize};

/// Strongly-typed object model for configuration.
#[derive(Serialize, Deserialize, Debug)]
pub struct Config {
    pub a: String,
    pub b: String,
}

/// Command-line options for this code.
#[derive(Clap)]
pub struct Opts {
    #[clap(default_value = "config.yml")]
    pub config: String,
}
