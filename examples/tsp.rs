//
//  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
//  File:      tsp.py
//
//  Purpose: Demonstrates a simple technique to the TSP
//           usign the Fusion API.
//
//
extern crate mosekmodel;


use mosekmodel::*;
use mosekmodel::matrix::*;

#[allow(non_snake_case)]
fn tsp(n : usize, A : & NDArray<2>, C : &NDArray<2>, remove_selfloops: bool, remove_2_hop_loops: bool) -> (Vec<f64>,Vec<[usize;2]>) {
    let mut M  = Model::new(None);

    let x = M.variable(None, domain::nonnegative().with_shape(&[n,n]).integer());
    _ = M.constraint(None, &x, domain::less_than(A.clone()));

    _ = M.constraint(None, & x.clone().sum_on(&[1]), equal_to(1.0).with_shape(&[n]));
    _ = M.constraint(None, & x.clone().sum_on(&[0]), equal_to(1.0).with_shape(&[n]));

    M.objective(None, Sense::Minimize, &C.clone().dot(x.clone()));

    if remove_2_hop_loops {
        M.constraint(None, & x.clone().add(x.clone().transpose()), less_than(1.0).with_shape(&[n,n]));
    }

    if remove_selfloops {
        M.constraint(None, & x.clone().diag(), equal_to(0.0).with_shape(&[n]));
    }

    //M.write_problem(format!("tsp-0-{}-{}.ptf",if remove_selfloops {'t'} else {'f'}, if remove_2_hop_loops {'t'} else {'f'}));
   
    for it in 0.. {
        println!("--------------------\nIteration {}",it);
        M.solve();

        println!("\nsolution cost: {}", M.primal_objective(SolutionType::Default).unwrap());
        println!("\nsolution:");

        let mut cycles : Vec<Vec<[usize;2]>> = Vec::new();

        for i in 0..n {
            let xi = M.primal_solution(SolutionType::Default, &(&x).index([i..i+1, 0..n])).unwrap();
            println!("x[{{}},:] = {:?}",xi);

            for (j,xij) in xi.iter().enumerate().filter(|(_,&v)| v > 0.5) {
                if let Some(c) = cycles.iter_mut()
                    .find(|c| c.iter().filter(|ij| ij[0] == i || ij[1] == i || ij[0] == j || ij[1] == j ).count() > 0) {
                    c.push([i,j]);
                }
                else {
                    cycles.push(vec![ [ i,j ] ]);
                }
            }
        }

        println!("\ncycles: {:?}",cycles);

        if cycles.len() == 1 {
            return (M.primal_solution(SolutionType::Default, &x).unwrap(), cycles[0].clone())
        }

        for c in cycles.iter_mut() {
            c.sort_by_key(|i| i[0]*n+i[1]);
            let ni = c.len();
            M.constraint(Some(format!("cycle-{:?}",c).as_str()), &x.clone().dot(matrix::sparse([n,n], c.clone(), vec![1.0; ni])), less_than((ni-1) as f64));
        }
    }
    (vec![],vec![])
}

#[allow(non_snake_case)]
fn main() {
    let A_ij : &[[usize;2]] = &[[0,1],[0,2],[0,3],[1,0],[1,2],[2,1],[2,3],[3,0]];
    let C_v : &[f64]        = &[   1.,  0.1,  0.1,  0.1,   1.,  0.1,   1.,   1.];

    let n = *A_ij.iter().flat_map(|r| r.iter()).max().unwrap()+1;
    let costs = matrix::sparse([n,n],A_ij.to_vec(),C_v);
    {
        println!("TSP, remove self loops");
        let (x,c) = tsp(n, &matrix::sparse([n,n],A_ij.to_vec(),vec![1.0; C_v.len()]), &costs , true, true);
        println!("x = {:?}, c = {:?}",x,c);
    }
    {
        println!("TSP, remove self loops and 2-hop loops");
        let (x,c) = tsp(n, &matrix::sparse([n,n],A_ij.to_vec(),vec![1.0; C_v.len()]), &costs , true, false);
        println!("x = {:?}, c = {:?}",x,c);
    }
}
#[test]
fn test() { main() }
