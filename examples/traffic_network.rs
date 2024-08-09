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
use mosekmodel::matrix::*;



pub fn traffic_network_model(
    arcs   : &[Arc], 
    nodes  : &[Node]) -> (Vec<f64>,Vec<[usize;2]>,Vec<f64>) 
{
    let mut model = Model::new(Some("TrafficNetwork"));
    let m = arcs.len();
    let n = nodes.len();
    if  n < arcs.iter().map(|a| a.i).chain(arcs.iter().map(|a| a.j)).max().unwrap()+1 {
        panic!("Number of nodes does not match arc definitions");
    }

    let basetime = NDArray::from_iter([n,n],arcs.iter().map(|arc| ([arc.i,arc.j],arc.base_travel_time))).unwrap();
    let sparsity : Vec<[usize;2]> = arcs.iter().map(|arc| [arc.i,arc.j]).collect();
    let cs_inv = NDArray::from_iter([n,n],arcs.iter().map(|arc| ([arc.i,arc.j],1.0 / (arc.traffic_sensitivity * arc.capacity)))).unwrap();
    let s_inv  = NDArray::from_iter([n,n],arcs.iter().map(|arc| ([arc.i,arc.j],1.0/arc.traffic_sensitivity))).unwrap();

    let x = model.variable(Some("traffic_flow"), greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));
    let d = model.variable(Some("d"),            greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));
    let z = model.variable(Some("z"),            greater_than(0.0).with_shape_and_sparsity(&[n,n],sparsity.as_slice()));

    // Set the objective:
    // (<basetime,x> + <e,d>) / T
    model.objective(Some("Average travel time"),
                    Sense::Minimize,
                    &x.clone().mul_elem(basetime).sum().add(d.clone().sum()));

    // Set up constraints
    // Constraint (1a)
    // 2 d_ij z_ij > x_ij^2
    model.constraint(Some("(1a)"),
                     &hstack![d.clone().gather().into_column(),
                              z.clone().gather().into_column(),
                              x.clone().gather().into_column()],
                     in_rotated_quadratic_cones(&[m,3], 1));

    // Constraint (1b)
    // Bound flows on each arc
    // 2 z_ij + x_ij / (s_ij c_ij) = 1/s_ij
    model.constraint(Some("(1b)"),
                     &z.clone().mul(2.0).add(x.clone().mul_elem(cs_inv)).sub(s_inv).gather(),
                     equal_to(vec![0.0; m]));
    
    // Constraint (2)
    // Network flow equations
    model.constraint(Some("(2)"),
                     &x.clone().sum_on(&[1])
                        .sub(x.clone().sum_on(&[0])),
                     equal_to(nodes.iter().map(|n| - n.sink_source).collect::<Vec<f64>>()));

    //model.write_problem("trafficnetwork.ptf");
    model.solve();

    let (xsol,sp) = model.sparse_primal_solution(SolutionType::Default, &x).unwrap();
    let tsol = xsol.iter().zip(arcs.iter()).map(|(&x,a)| a.base_travel_time + a.traffic_sensitivity * x / (1.0 - x/a.capacity)).collect(); 

    (xsol,sp,tsol)
}


/// Arc data
#[derive(Clone)]
pub struct Arc {
    /// Node start index
    pub i                   : usize,
    /// Node end index
    pub j                   : usize,
    /// Base travel tile for the arc
    pub base_travel_time    : f64,
    /// Arc traffic capacity
    pub capacity            : f64,
    /// Arc traffic sensitivity
    pub traffic_sensitivity : f64
}


/// Node data
#[derive(Clone)]
pub struct Node {
    /// Sink or source value. Normal nodes have `sink_source=0` indicating inflow equals outflow.
    /// Positive indicates a source node with the given inflow, negative indicates a sink node with the
    /// given outflow.
    pub sink_source : f64,
    /// Node label, used for the UI.
    pub label : Option<String>
}

impl Arc {
    pub fn new(i : usize,
               j : usize,
               base_travel_time    : f64,
               capacity            : f64,
               traffic_sensitivity : f64) -> Arc {
        Arc{ i,j,base_travel_time,capacity,traffic_sensitivity }
    }
}

impl Node {
    pub fn new(label : Option<&str>,contribute : f64) -> Node {
        Node {
            sink_source : contribute,
            label : label.map(|n| n.to_string())
        }
    }
}

fn main() {
    let nodes = &[ Node::new(None, 20.0),
                   Node::new(None, 0.0),
                   Node::new(None, 0.0),
                   Node::new(None, -20.0) ];
    let arcs  = &[ Arc::new( 0,  1,  4.0, 10.0, 0.1 ),
                   Arc::new( 0,  2,  1.0, 12.0, 0.7 ),
                   Arc::new( 2,  1,  2.0, 20.0, 0.9 ),
                   Arc::new( 1,  3,  1.0, 15.0, 0.5 ),
                   Arc::new( 2,  3,  6.0, 10.0, 0.1 ) ];

    let (flow,sp,t) = traffic_network_model(arcs,nodes);

    println!("Optimal flow:");

    for (&ij,&f,) in sp.iter().zip(flow.iter()) {
      println!("\tflow node {} -> node {} = {:.4}", ij[0],ij[1],f);
    }
}
