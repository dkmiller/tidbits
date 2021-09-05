use clap::Clap;
use serde::{Serialize, Deserialize};
use serde_yaml::from_reader;
use std::fs::{File, metadata, read_dir};
use std::error::Error;

#[derive(Serialize, Deserialize, Debug)]
struct Config {
    a: String,
    b: String,
}

#[derive(Clap)]
struct Opts {
    #[clap(default_value="config.yml")]
    config: String
}

fn load_config(config_path: String) -> Result<Config, Box<Error>> {
    // https://stackoverflow.com/a/53243962/
    let f = File::open(config_path)?;
    let c: Config = from_reader(f)?;

    Ok(c)


    // // let f = File::open(opts.config).unwrap();
    // // // let conf: Config = serde_yaml::from_reader(f).unwrap();
    // // // match f {
    // // //     Ok(ff) => serde_yaml::from_reader(ff),
    // // //     Err(e) => 1
    // // // };
    // // // let d = serde_yaml::from_reader(f);
    // match f {
    //     Ok(ff) => serde_yaml::from_reader(ff),
    //     _ => _
    //     Err(e) => 1
    // };
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
        Err(e) => println!("{}", e)
    };
}
