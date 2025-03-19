//! # Comparable times
//! 
//! Comparing runtimes of simple expressions as implemented in MosekCOModel (--release mode) and in
//! MOSEK Java Fusion.
//! 
//! Date: March 13, 2025
//!|                         | Rust  | Java  |
//!| Stacking, mixed         |  0.19 |  0.55 |
//!| Stacking, dense         |  0.09 |  0.20 |
//!| Stacking, sparse        |  0.02 |  0.06 |
//!| Mul dense X * dense M   |  0.14 |  0.26 |
//!| Mul sparse X * dense M  |  0.17 |  0.07 |
//!| Mul dense X * sparse M  |  0.12 |  0.03 |
//!| Mul sparse X * sparse M |  0.10 |  0.06 |
//!| Mul dense M * dense X   |  0.14 |  0.33 |
//!| Mul dense M * sparse X  |  0.24 |  0.09 |
//!| Mul sparse M * dense X  |  0.14 |  0.04 |
//!| Mul sparse M * sparse X |  0.28 |  0.06 |
//!
extern crate mosekcomodel;
mod tests;
use std::{collections::HashMap};
use tests::*;

pub fn main() {
    let mut classpath = None;
    let mut compare = false;
    let mut table_col_sep = "\t";
    let mut table_row_pfx = "";
    let mut table_row_sfx = "";
    let mut table_align = false;


    let mut args = std::env::args();
    println!("args = {:?}",std::env::args().collect::<Vec<String>>());
    _  = args.next(); // pop executable name
    while let Some(a) = args.next() {
        match a.as_str() {
            "--style" => {
                if let Some(v) = args.next() {
                    match v.as_str() {
                        "csv" => (table_align,table_col_sep,table_row_pfx,table_row_sfx) = (false,",","",""),
                        "md"  => (table_align,table_col_sep,table_row_pfx,table_row_sfx) = (true," | ","| "," |"),
                        "tab" => (table_align,table_col_sep,table_row_pfx,table_row_sfx) = (false,"\t", "",""),
                        _ => {}
                    }
                }
            },
            "--compare" => compare = true,
            "--classpath"|"-cp" => if let Some(v) = args.next() { classpath = Some(v) },
            "--help"|"-h" => {
                println!("compare [--style STYLE] [--compare] [--classpath PATH | -cp PATH] [--help]");
                println!("  --compare run Java fusion implementation for comparison");
                println!("  --classpath|-cp PATH provide CLASSPATH for compiling and running java");
                println!("  --style (csv|markdown|tab) select output style");
                return;
            }
            _ => panic!("Unexpected argument {}",a)
        }
    }

    let mut rundata : HashMap<String,f64> = HashMap::new();

    if compare {
        println!("Compile Java...");
        let mut cmd1 = std::process::Command::new("javac");
            if let Some(ref cp) = classpath {
                cmd1.arg("-cp").arg(cp);
            } 
            cmd1.arg("-d").arg("target")
                .arg("java/timing.java");
        cmd1.status().expect("Failed to compile java example");
        println!("    Ok");

        println!("Run Java examples...");
        for name in ["stacking1","stacking2","stacking3",
                     "mul1","mul2","mul3","mul4",
                     "mul5","mul6","mul7","mul8"].iter() {
            let mut cmd2 = std::process::Command::new("java");
            cmd2.arg("-Xmx8192G")
                .arg("-Xms4096G");
            if let Some(ref cp) = classpath {
                cmd2.arg("-cp").arg(format!("target:{}",cp));
            } 
            else {
                cmd2.arg("-cp").arg("target");
            };
            cmd2.arg("com.mosek.fusion.examples.timing");
            cmd2.arg(name);
            let output = cmd2.output().expect("Failure to run Java tests.");
            println!("    {} Ok", name);
            let data = std::str::from_utf8(output.stdout.as_slice()).unwrap();
            if let Ok(v) = data.parse::<f64>() { _ = rundata.insert(name.to_string(),v); }
        }
    }

    let tabledata = vec![
        ("Stacking, mixed",        stacking1(),rundata.get("stacking1")),
        ("Stacking, dense",        stacking2(),rundata.get("stacking2")),
        ("Stacking, sparse",       stacking3(),rundata.get("stacking3")),
        ("Mul dense X * dense M",  mul1(),     rundata.get("mul1")),
        ("Mul sparse X * dense M", mul2(),     rundata.get("mul2")),
        ("Mul dense X * sparse M", mul3(),     rundata.get("mul3")),
        ("Mul sparse X * sparse M",mul4(),     rundata.get("mul4")),
        ("Mul dense M * dense X",  mul5(),     rundata.get("mul5")),
        ("Mul dense M * sparse X", mul6(),     rundata.get("mul6")),
        ("Mul sparse M * dense X", mul7(),     rundata.get("mul7")),
        ("Mul sparse M * sparse X",mul8(),     rundata.get("mul8")),
    ];

    let width = tabledata.iter().map(|(n,_,_)| n.len()).max().unwrap();

    if table_align {
        println!("{}{:<width$}{}{:5}{}{:5}{}", table_row_pfx,"",table_col_sep,"Rust",table_col_sep,"Java",table_row_sfx, width = width );
        for (n,v,v2) in tabledata {
            if let Some(v2) = v2 { println!("{}{:<width$}{}{:5.2}{}{:5.2}{}", table_row_pfx,n,table_col_sep,v,table_col_sep,v2,table_row_sfx, width = width  ); }
            else                 { println!("{}{:<width$}{}{:5.2}{}  -  {}", table_row_pfx,n,table_col_sep,v,table_col_sep,table_row_sfx, width = width  ); }
        }
    }
    else {
        println!("{}{}{}{:5}{}{:5}{}", table_row_pfx,"",table_col_sep,"Rust",table_col_sep,"Java",table_row_sfx );
        for (n,v,v2) in tabledata {
            if let Some(v2) = v2 { println!("{}{}{}{}{}{}{}", table_row_pfx,n,table_col_sep,v,table_col_sep,v2,table_row_sfx ); }
            else                 { println!("{}{}{}{}{}-{}", table_row_pfx,n,table_col_sep,v,table_col_sep,table_row_sfx ); }
        }
    }

   

/*
    println!("Stacking, mixed\t{:.2}\t{}",        stacking1(),rundata.get("stacking1").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string().to_string()));
    println!("Stacking, dense\t{:.2}\t{}",        stacking2(),rundata.get("stacking2").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Stacking, sparse\t{:.2}\t{}",       stacking3(),rundata.get("stacking3").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul dense X * dense M\t{:.2}\t{}",  mul1(),     rundata.get("mul1").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul sparse X * dense M\t{:.2}\t{}", mul2(),     rundata.get("mul2").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul dense X * sparse M\t{:.2}\t{}", mul3(),     rundata.get("mul3").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul sparse X * sparse M\t{:.2}\t{}",mul4(),     rundata.get("mul4").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul dense M * dense X\t{:.2}\t{}",  mul5(),     rundata.get("mul5").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul dense M * sparse X\t{:.2}\t{}", mul6(),     rundata.get("mul6").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul sparse M * dense X\t{:.2}\t{}", mul7(),     rundata.get("mul7").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
    println!("Mul sparse M * sparse X\t{:.2}\t{}",mul8(),     rundata.get("mul8").map(|v| format!("{:.2}",v)).unwrap_or("-".to_string()));
*/
}
