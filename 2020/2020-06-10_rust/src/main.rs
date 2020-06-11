use std::io;

fn main() {
    // ! means a macro is being invoked.
    println!("Please enter a number...");

    let mut num = String::new();

    io::stdin()
        .read_line(&mut num)
        .expect("You didn't enter a line");

    println!("You entered {}", num);
}
