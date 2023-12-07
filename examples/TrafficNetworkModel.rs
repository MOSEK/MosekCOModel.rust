//
// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:      TrafficNetworlModel.rs
//
// Purpose:   Demonstrates a traffic network problem as a conic quadratic problem.
//
// Source:    Robert Fourer, "Convexity Checking in Large-Scale Optimization",
//            OR 53 --- Nottingham 6-8 September 2011.
//
// The problem:
//            Given a directed graph representing a traffic network
//            with one source and one sink, we have for each arc an
//            associated capacity, base travel time and a
//            sensitivity. Travel time along a specific arc increases
//            as the flow approaches the capacity.
//
//            Given a fixed inflow we now wish to find the
//            configuration that minimizes the average travel time.

extern crate mosekmodel;

use mosekmodel::*;
use mosekmodel::matrix::SparseMatrix;


/// Solve traffix network model.
/// # Arguments
///
/// * `number_of_nodes` - Number of nodes in graph
/// * `source_idx` - Index of the source node
/// * `sink_idx` - Index of the sink node
/// * `arcs` - List of arcs with arc data.
/// * `T` -
fn traffic_network_model( number_of_nodes : usize,
                          source_idx : usize,
                          sink_idx   : usize,
                          arcs : &[Arc],
                          T : f64) -> Result<Vec<f64>,String> {
    let mut model = Model::new(Some("Traffic Network Model"));
    let n = number_of_nodes;
    let m = arcs.len();
    
    let basetime = SparseMatrix::from_iterator(n,n,arcs.iter().map(|arc| (arc.i,arc.j,arc.basetime)));

    let sparsity : Vec<[usize;2]> = arcs.iter().map(|arc| [arc.i,arc.j]).collect();
    let cs_inv = SparseMatrix::from_iterator(n,n,arcs.iter().map(|arc| (arc.i,arc.j,1.0 / (arc.sensitivity * arc.capacity))));
    let s_inv  = SparseMatrix::from_iterator(n,n,arcs.iter().map(|arc| (arc.i,arc.j,1.0/arc.sensitivity)) );

    let x = model.variable(Some("traffic_flow"), greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));
    let d = model.variable(Some("d"),            greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));
    let z = model.variable(Some("z"),            greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));
    let x_arc = model.variable(Some("traffic_flow_arc"),greater_than(0.0).with_shape_and_sparsity(&[n,n],&[ [sink_idx,source_idx] ]));


    // Set the objective:
    // (<basetime,x> + <e,d>) / T
    model.objective(Some("Average travel time"),
                    Sense::Minimize,
                    &x.clone().mul_elm(basetime).sum().add(d.clone().sum()));

    // Set up constraints
    // Constraint (1a)
    model.constraint(Some("(1a)"),
                     &hstack![d.clone().gather().into_column(),
                              z.clone().gather().into_column(),
                              x.clone().gather().into_column()],
                     in_rotated_quadratic_cones(&[m,3], 1));

    // Constraint (1b)
    // Bound flows on each arc
    model.constraint(Some("(1b)"),
                     &x.clone().mul_elm(cs_inv).dynamic().add(x.clone().dynamic()).sub(s_inv).gather(),
                     equal_to(vec![0.0; m]));
    
    // Constraint (2)
    // Network flow equations
    model.constraint(Some("(2)"),
                     &x.clone().add(x_arc.clone()).sum_on(&[1])
                        .sub((x.clone().add(x_arc.clone())).sum_on(&[0])),
                     equal_to(vec![0.0; n]));

    // Constraint (3)
    model.constraint(Some("(3)"),
                     &x_arc.clone().gather().with_shape(&[]), equal_to(T));

    model.solve();
    
    model.primal_solution(SolutionType::Default,&x)
}

struct Arc {
    i : usize,
    j : usize,
    basetime : f64,
    capacity : f64,
    sensitivity : f64
}
impl Arc {
    fn new(i : usize, j : usize, basetime : f64, capacity : f64, sensitivity : f64) -> Arc { Arc {i,j,basetime,capacity,sensitivity} }
}

fn main() {
    let n = 4;

    let arcs = &[ Arc::new( 0,  1,  4.0, 10.0, 0.1 ),
                  Arc::new( 0,  2,  1.0, 12.0, 0.7 ),
                  Arc::new( 2,  1,  2.0, 20.0, 0.9 ),
                  Arc::new( 1,  3,  1.0, 15.0, 0.5 ),
                  Arc::new( 2,  3,  6.0, 10.0, 0.1 ) ];

    let T = 20.0;
    let source_idx : usize = 0;
    let sink_idx   : usize = 3;


    let flow = traffic_network_model(n, source_idx, sink_idx,
                                     arcs,
                                     T).unwrap();

    println!("Optimal flow:");

  
    for (arc,&f) in arcs.iter().zip(flow.iter()) {
      println!("\tflow node {} -> node {} = {:.4}", arc.i,arc.j,f);
    }
}
