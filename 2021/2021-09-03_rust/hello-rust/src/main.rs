use clap::Clap;
use serde_yaml::from_reader;
use std::error::Error;
use std::fs::{metadata, read_dir, File};
use std::process::Command;

mod models;
use models::{Config, Opts};

/// Load a strongly typed configuration object from the specified path.
fn load_config(config_path: String) -> Result<Config, Box<dyn Error>> {
    // https://stackoverflow.com/a/55125216
    // https://stackoverflow.com/a/53243962/
    let f = File::open(config_path)?;
    let c: Config = from_reader(f)?;

    Ok(c)
}

/// Get the access token corresponding to the users Azure CLI login
/// and specified resource identifier.
fn azure_access_token(resource: String) -> Result<String, Box<dyn Error>> {
    // az account get-access-token --resource https://vault.azure.net/ --query accessToken -o tsv

    // https://stackoverflow.com/a/42993724/
    let output = Command::new("az")
        .args([
            "account",
            "get-access-token",
            "--resource",
            &resource,
            "--query",
            "accessToken",
            "-o",
            "tsv",
        ])
        .output()?;

    let access_token = String::from_utf8_lossy(&output.stdout);

    Ok(access_token.to_string())
}

fn main() {
    println!("Hello, world!");

    let entries = read_dir(".").unwrap();

    for entry in entries {
        let path = entry.unwrap().path();
        let meta = metadata(&path).unwrap();

        let prefix = if meta.is_dir() { "(dir)" } else { "     " };

        // Sad, would prefer:
        // https://rust-lang.github.io/rfcs/2795-format-args-implicit-identifiers.html
        match path.to_str() {
            Some(s) => println!("{} {}", prefix, s),
            None => println!("{} {:?} (invalid utf-8)", prefix, path),
        };
    }

    // https://github.com/clap-rs/clap
    let opts: Opts = Opts::parse();
    println!("Configuration file: {}", opts.config);

    let config = load_config(opts.config);

    match config {
        Ok(c) => println!("{:?}", c),
        Err(e) => println!("{}", e),
    };

    let access_token = azure_access_token("https://vault.azure.net/".to_string());
    match access_token {
        Ok(s) => println!("{}", s),
        Err(e) => println!("{}", e),
    };
}
