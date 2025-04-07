//!
//!  Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//!  File:      `tsp.py`
//!
//!  Purpose: Demonstrates a simple technique to the TSP
//!           usign the Fusion API.
//!
//!
extern crate mosekcomodel;
extern crate rand;
extern crate itertools;


use std::{cell::RefCell, sync::mpsc::{self, Sender}, thread::JoinHandle};

use itertools::iproduct;
use mosekcomodel::*;
use rand::Rng;
use utils::AppliedPermutation;

const APP_ID : &str = "com.mosek.example.tsp";




struct DrawData {
    points : Vec<[f64;2]>,
}






fn main() {
    let mut n : usize = 15;
    let remove_2_hop_loops = false;
    let remove_selfloops = false;

    let mut args = std::env::args(); args.next();
    while let Some(a) = args.next() {
        match s.as_str() {
            "-n" => if let Some(v) = args.next() { n = v.parse::<usize>() },
            "--remote-2-hop-loops" => remove_selfloops = true,
            "--remove-self-loops" => remove_selfloops = true,
            _ => {},
        }
    }

    let mut rng = rand::rng();

    let points = (0..n).map(|_| [ rng.random_range(0..1000) as f64 / 1000.0, rng.random_range(0..1000) as f64 / 1000.0]).collect();



    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    let threads = Rc::new(RefCell::new(Vec::new()));

    {
        let threads = threads.clone();
        app.connect_activate(move | app : &Application | build_ui(app,&points,threads.clone()));
    }

    let r = app.run_with_args::<&str>(&[]);
    for t in threads.borrow_mut().iter() {
        t.join(); 
    }
    println!("Main loop exit!");
}

fn build_ui(app : &Application, points : &Vec<[f64;2]>, threads : Rc<RefCell<Vec<JoinHandle<()>>>>) {
    let drawdata = DrawData{ points : points.clone() };

    let darea = DrawingArea::builder()
        .width_request(1500) 
        .height_request(1500)
        .build();

    // Redraw callback
    {
        let data = data.clone();
        darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
    }

    let current_solution = Box::new(RefCell::new(Vec::new()));
    let (tx,rx) = mpsc::channel();

    {
        let points = points.clone();
        threads.borrow_mut().push(std::thread::spawn(move|| optimize(points,tx)));
    }


    let window = ApplicationWindow::builder()
        .application(app)
        .title("Hello Löwner-John")
        .child(&darea)
        .build();
}

fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h);

    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    context.translate(s/2.0, s/2.0);
    context.scale(0.8*s, 0.8*s);
    context.set_source_rgb(0.0, 0.0, 0.0);


}

fn optimize(points : Vec<[f64;2]>,tx : Sender<Vec<usize>>) {
    let arc_w = matrix::dense([n,n], iproduct!(points.iter(),points.iter()).map(|(p0,p1)| ((p0[0]-p1[0]).pow(2.0) + (p0[1]-p1[1]).pow(2)).sqrt()).collect());

    let mut model  = Model::new(None);

    let x = model.ranged_variable(None, in_range(0.0,1.0).with_shape(&[n,n]).integer()).0;

    _ = model.constraint(None,  x.sum_on(&[1]), equal_to(1.0));
    _ = model.constraint(None,  x.sum_on(&[0]), equal_to(1.0));

    model.objective(None, Sense::Minimize, arc_w.dot(&x));

    if remove_2_hop_loops {
        model.constraint(None,  x.add(x.transpose()), less_than(1.0));
    }

    if remove_selfloops {
        model.constraint(None, x.diag(), equal_to(0.0));
    }

    {
        let x = x.clone();
        model.set_solution_callback(move |model| 
            if let Ok(xx) = model.primal_solution(SolutionType::Integer, &x) {
                println!("New Solution: {:?}",xx)
            });
    }
   
    for it in 0.. {
        println!("--------------------\nIteration {}",it);
        model.solve();

        println!("\nsolution cost: {}", model.primal_objective(SolutionType::Default).unwrap());
        println!("\nsolution:");

        let mut cycles : Vec<Vec<[usize;2]>> = Vec::new();

        for i in 0..n {
            let xi = model.primal_solution(SolutionType::Default, &(&x).index([i..i+1, 0..n])).unwrap();
            println!("x[{{}},:] = {:?}",xi);

            for (j,_xij) in xi.iter().enumerate().filter(|(_,&v)| v > 0.5) {
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
            return (model.primal_solution(SolutionType::Default, &x).unwrap(), cycles[0].clone())
        }

        for c in cycles.iter_mut() {
            c.sort_by_key(|i| i[0]*n+i[1]);
            let ni = c.len();
            model.constraint(Some(format!("cycle-{:?}",c).as_str()), 
                         x.dot(matrix::sparse([n,n], c.to_vec(), vec![1.0; ni])), 
                         less_than((ni-1) as f64));
        }
    }
    (vec![],vec![])
}



//    let A_ij : &[[usize;2]] = &[[0,1],[0,2],[0,3],[1,0],[1,2],[2,1],[2,3],[3,0]];
//    let C_v : &[f64]        = &[   1.,  0.1,  0.1,  0.1,   1.,  0.1,   1.,   1.];
//
//    let n = *A_ij.iter().flat_map(|r| r.iter()).max().unwrap()+1;
//    let costs = matrix::sparse([n,n],A_ij.to_vec(),C_v);
//    {
//        println!("TSP, remove self loops");
//        let (x,c) = tsp(n, &matrix::sparse([n,n],A_ij.to_vec(),vec![1.0; C_v.len()]), &costs , true, true);
//        println!("x = {:?}, c = {:?}",x,c);
//    }
//    {
//        println!("TSP, remove self loops and 2-hop loops");
//        let (x,c) = tsp(n, &matrix::sparse([n,n],A_ij.to_vec(),vec![1.0; C_v.len()]), &costs , true, false);
//        println!("x = {:?}, c = {:?}",x,c);
//    }
}




fn random_n_swap<T>(source : & mut Vec<T>, n : usize) {
    let mut rng = rand::rng();
    for k in 0..n.max(source.len()) {
        let i = rng.random_range(k..source.len());
        source.swap(k,i);
    }
}
















/*
#[allow(non_snake_case)]
fn tsp(n : usize, A : & NDArray<2>, C : &NDArray<2>, remove_selfloops: bool, remove_2_hop_loops: bool) -> (Vec<f64>,Vec<[usize;2]>) {
    let mut M  = Model::new(None);


    let x = M.variable(None, domain::nonnegative().with_shape(&[n,n]).integer());
    _ = M.constraint(None, &x, domain::less_than(A.clone()));

    _ = M.constraint(None,  x.sum_on(&[1]), equal_to(1.0).with_shape(&[n]));
    _ = M.constraint(None,  x.sum_on(&[0]), equal_to(1.0).with_shape(&[n]));

    M.objective(None, Sense::Minimize, C.dot(&x));

    if remove_2_hop_loops {
        M.constraint(None,  x.add(x.transpose()), less_than(1.0).with_shape(&[n,n]));
    }

    if remove_selfloops {
        M.constraint(None, x.diag(), equal_to(0.0).with_shape(&[n]));
    }

    //M.write_problem(format!("tsp-0-{}-{}.ptf",if remove_selfloops {'t'} else {'f'}, if remove_2_hop_loops {'t'} else {'f'}));
    {
        let x = x.clone();
        M.set_solution_callback(move |M| 
            if let Ok(xx) = M.primal_solution(SolutionType::Integer, &x) {
                println!("New Solution: {:?}",xx)
            });
    }
   
    for it in 0.. {
        println!("--------------------\nIteration {}",it);
        M.solve();

        println!("\nsolution cost: {}", M.primal_objective(SolutionType::Default).unwrap());
        println!("\nsolution:");

        let mut cycles : Vec<Vec<[usize;2]>> = Vec::new();

        for i in 0..n {
            let xi = M.primal_solution(SolutionType::Default, &(&x).index([i..i+1, 0..n])).unwrap();
            println!("x[{{}},:] = {:?}",xi);

            for (j,_xij) in xi.iter().enumerate().filter(|(_,&v)| v > 0.5) {
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
            M.constraint(Some(format!("cycle-{:?}",c).as_str()), 
                         x.dot(matrix::sparse([n,n], c.to_vec(), vec![1.0; ni])), 
                         less_than((ni-1) as f64));
        }
    }
    (vec![],vec![])
}
*/
