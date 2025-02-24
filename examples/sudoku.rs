///
/// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
///
/// File:      sudoku.rs
///
/// Purpose:  A MILP-based SUDOKU solver
///

extern crate mosekmodel;
use itertools::{iproduct, izip};
use mosekmodel::*;

fn main() {
    //fixed cells in human readable (i.e. 1-based) format
    let hr_fixed = [ 
        [1, 5, 4],
        [2, 2, 5], [2, 3, 8], [2, 6, 3],
        [3, 2, 1], [3, 4, 2], [3, 5, 8], [3, 7, 9],
        [4, 2, 7], [4, 3, 3], [4, 4, 1], [4, 7, 8], [4, 8, 4],
        [6, 2, 4], [6, 3, 1], [6, 6, 9], [6, 7, 2], [6, 8, 7],
        [7, 3, 4], [7, 5, 6], [7, 6, 5], [7, 8, 8],
        [8, 4, 4], [8, 7, 1], [8, 8, 6],
        [9, 5, 9]
    ];    


    let m = 3;
    let n = m * m;

    let fixed = hr_fixed.map(|v| v.map(|w| w - 1));

    let mut model = Model::new(Some("SUDOKU"));
    
    model.set_log_handler(|msg| print!("{}",msg));
    let x = model.variable(None, nonnegative().with_shape(&[n,n,n]).integer());
    model.constraint(None,&x.clone(),less_than(1.0).with_shape(&[n,n,n]));

    // each value only once per dimension
    model.constraint(None, &x.clone().sum_on(&[1,2]), equal_to(1.0).with_shape(&[n,n]));
    model.constraint(None, &x.clone().sum_on(&[0,2]), equal_to(1.0).with_shape(&[n,n]));
    model.constraint(None, &x.clone().sum_on(&[0,1]), equal_to(1.0).with_shape(&[n,n]));

    // each number must appear only once in a block
    for k in 0..n {
        for i in 0..m {
            for j in 0..m {
              model.constraint(None,
                               &(&x).index([i*m..(i+1)*m, j*m..(j+1)*m, k..k+1]).sum(),
                               equal_to(1.0));
            }
        }
    }

    model.constraint(None, 
                     &stackvec(0,fixed.iter().map(|&i| (&x).index(i).reshape(&[1]) ).collect::<Vec<Variable<1>>>()), 
                     equal_to(1.0).with_shape(&[fixed.len()]));

    model.solve();

    //print the solution, if any...
    
    let (psta,_) = model.solution_status(SolutionType::Default);

    if let SolutionStatus::Optimal = psta {
        let mut unfilled = vec![0usize;n*n];
        for item in hr_fixed.iter() {
            unfilled[ (item[0]-1)*n + (item[1]-1)] = item[2];
        }
        println!("Puzzle:");
        print_solution(m, &unfilled.as_slice());

        let res = model.primal_solution(SolutionType::Default, &x).unwrap();
        let mut filled = vec![0usize;n*n];
    
        for ((i,j),k,r) in 
            izip!(iproduct!(0..n,0..n),
                  res.chunks(n).map(|vv| vv.iter().enumerate().find_map(|(i,&v)| if v > 0.5 { Some(i+1) } else { None }).unwrap_or(0)),
                  filled.iter_mut()) {
            *r = k;
        }
        println!("Solution:");
        print_solution(m,&filled);
    }
    else {
      println!("No solution found!");
    }
}

fn print_solution(m : usize, data : &[usize]) {
    let n = m * m;
    
    for ((i,j),&v) in iproduct!(0..n,0..n).zip(data.iter()) {
        if i % m == 0 && j % n == 0 { println!(" +-------+-------+-------+"); }
        if j % m == 0 { print!(" |"); }
        if v > 0 { print!(" {}",v); }
        else { print!("  ") }
        if (j+1) % n == 0 { println!(" |"); }
    }
    println!(" +-------+-------+-------+");
}

/*
fn print_solution(model : &Model, m : usize, x : &Variable<3>, puzzle : &[[usize;3]) {
    let n = m * m;
    println!();

    {
        let mut unfilled = vec![0;n*n];
        for item in hr_fixed.iter() {
            unfilled[ item[0]*n + item[1]] = item[2];
        }
        println!(" +-------+-------+-------+");
        for i in 0..n {
            for j in 0..n {
                if j % m == 0 { 
                    print!(" |");
                }
                for k in 0..n {
                    if res[[i,j,k]] > 0.5 {
                        print!(" {}",k+1);
                    }
                }
            }
            println!(" |");
            if (i + 1) % m == 0 {
                println!(" +-------+-------+-------+");
            }
        }
    }


    let res = NDArray::new([n,n,n],None,model.primal_solution(SolutionType::Default, x).unwrap()).unwrap();

    println!("Solution:")
    println!(" +-------+-------+-------+");
    for i in 0..n {
        for j in 0..n {
            if j % m == 0 { 
                print!(" |");
            }
            for k in 0..n {
                if res[[i,j,k]] > 0.5 {
                    print!(" {}",k+1);
                }
            }
        }
        println!(" |");
        if (i + 1) % m == 0 {
            println!(" +-------+-------+-------+");
        }
    }
}
*/


