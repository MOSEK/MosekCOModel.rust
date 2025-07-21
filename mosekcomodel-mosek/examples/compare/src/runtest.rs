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
                "sumon1" => tests::sumon1(),
                "sumon2" => tests::sumon2(),
                "sumon3" => tests::sumon3(),
                "sumon4" => tests::sumon4(),
                "sumon1s" => tests::sumon1s(),
                "sumon2s" => tests::sumon2s(),
                "sumon3s" => tests::sumon3s(),
                "sumon4s" => tests::sumon4s(),
                _ => 0.0
            };
        println!("Time {}: {:.2}",a,time);
    }
    else {
        println!("runtest TESTNAME");
        println!("    TESTNAME: ");
        for name in [ "mul1", "mul2", "mul3", "mul4", "mul5", "mul6", "mul7", "mul8", "sumon1", "sumon2", "sumon3", "sumon4", "sumon1s", "sumon2s", "sumon3s", "sumon4s" ].iter() {
            println!("      {}",name);
        }
    }
}

