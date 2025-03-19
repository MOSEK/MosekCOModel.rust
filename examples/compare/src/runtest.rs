extern crate mosekcomodel;

mod tests;
use tests::*;

fn main() {
    let mut args = std::env::args();
    _ = args.next();
    if let Some(a) = args.next() {
        let time = 
            match a.as_str() {
                "mul1" => tests::mul1(),
                "mul2" => tests::mul2(),
                "mul3" => tests::mul3(),
                "mul4" => tests::mul4(),
                "mul5" => tests::mul5(),
                "mul6" => tests::mul6(),
                "mul7" => tests::mul7(),
                "mul8" => tests::mul8(),
                _ => 0.0
            };
        println!("Time {}: {:.2}",a,time);
    }
}

