//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! Purpose: Demonstrates a simple technique to the TSP.
//!
//!
extern crate mosekcomodel;
extern crate rand;
extern crate itertools;

use std::{cell::RefCell, rc::Rc, sync::mpsc::{self, Receiver, Sender}, thread::JoinHandle, time::Duration};

use cairo::Context;
use gtk::{glib::{self, ControlFlow}, prelude::{ApplicationExt, ApplicationExtManual, DrawingAreaExtManual, WidgetExt}, Application, ApplicationWindow, DrawingArea};
use itertools::iproduct;
use mosekcomodel::*;
use rand::Rng;
use utils::AppliedPermutation;

const APP_ID : &str = "com.mosek.example.tsp";


struct DrawData {
    points : Vec<[f64;2]>,
    sol : Vec<(usize,usize)>,
}

enum Command {
    Terminate
}
#[derive(Copy,Clone)]
struct Config {
    n : usize,
    remove_2_hop_loops : bool,
    remove_selfloops : bool,
}

fn main() {
    let mut conf = Config{
        n : 15,
        remove_2_hop_loops : false,
        remove_selfloops : false,
    };

    let mut args = std::env::args(); args.next();
    while let Some(a) = args.next() {
        match a.as_str() {
            "-n" => if let Some(v) = args.next() { if let Ok(v) = v.parse::<usize>() { conf.n = v }},
            "--remove-2-hop-loops" => conf.remove_selfloops = true,
            "--remove-self-loops" => conf.remove_selfloops = true,
            _ => {},
        }
    }

    let mut rng = rand::rng();

    let points = (0..conf.n).map(|_| [ rng.random_range(0..1000) as f64 / 1000.0, rng.random_range(0..1000) as f64 / 1000.0]).collect();

    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    let threads = Rc::new(RefCell::new(Vec::new()));

    {
        let threads = threads.clone();
        app.connect_activate(move | app : &Application | build_ui(app,conf,&points,threads.clone()));
    }

    let r = app.run_with_args::<&str>(&[]);
    for t in threads.borrow_mut().iter() {
        t.join(); 
    }
    println!("Main loop exit!");
}

fn build_ui(app : &Application, conf : Config,points : &Vec<[f64;2]>, threads : Rc<RefCell<Vec<JoinHandle<()>>>>) {
    let n = conf.n;
    let drawdata = Rc::new(RefCell::new(DrawData{ 
        points : points.clone(),
        sol    : Vec::new(),
    }));

    let darea = DrawingArea::builder()
        .width_request(1500) 
        .height_request(1500)
        .build();

    // Redraw callback
    {
        let data = drawdata.clone();
        darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
    }

    let (tx,rx) = mpsc::channel();
    let (rtx,rrx) = mpsc::channel();

    {
        let points = points.clone();
        threads.borrow_mut().push(std::thread::spawn(move|| optimize(&conf,points,tx)));
    }

    let window = ApplicationWindow::builder()
        .application(app)
        .title("TSP")
        .child(&darea)
        .build();

    { // Time callback
      // For each tick we check if any solutions were sent from the solver thread until the solver
      // thread closes the pipe.
        let darea = darea.clone();
        glib::source::timeout_add_local(
            Duration::from_millis(50), 
            move || {
                loop {
                    match rx.try_recv() {
                        Ok(data) => {
                            let dd = drawdata.borrow_mut();
                            dd.sol.clear();
                            dd.sol.extend_from_slice(data.as_slice());
                          
                            darea.queue_draw();
                        },
                        Err(mpsc::TryRecvError::Empty) => return ControlFlow::Continue,
                        Err(mpsc::TryRecvError::Disconnected) => return ControlFlow::Break,
                    }
                }
            });
    }
    
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

    context.set_source_rgb(0.8, 0.8, 0.8);
    for (p0,p1) in iproduct!(data.points.iter(),data.points.iter()) {
        context.move_to(p0[0],p0[1]);
        context.line_to(p1[0],p1[1]);
    }
    context.paint();
    
    context.set_source_rgb(0.0, 0.0, 0.0);
    for (&i,&j) in data.sol.iter() {
        let p0 = data.points[i];
        let p1 = data.points[j];
        context.move_to(p0[0],p0[1]);
        context.line_to(p1[0],p1[1]);
    }
    context.paint();
}

fn optimize(conf : &Config, points : Vec<[f64;2]>,tx : Sender<Vec<(usize,usize)>>,rx : Receiver<Command>) {
    let n = conf.n;
    let arc_w = matrix::dense([n,n], iproduct!(points.iter(),points.iter()).map(|(p0,p1)| ((p0[0]-p1[0]).powi(2) + (p0[1]-p1[1]).powi(2)).sqrt()).collect::<Vec<f64>>());

    let mut model  = Model::new(None);

    let x = model.ranged_variable(None, in_range(0.0,1.0).with_shape(&[n,n]).integer()).0;

    _ = model.constraint(None,  x.sum_on(&[1]), equal_to(1.0));
    _ = model.constraint(None,  x.sum_on(&[0]), equal_to(1.0));

    model.objective(None, Sense::Minimize, arc_w.dot(&x));

    if conf.remove_2_hop_loops {
        model.constraint(None,  x.add(x.transpose()), less_than(1.0));
    }

    if conf.remove_selfloops {
        model.constraint(None, x.diag(), equal_to(0.0));
    }

    {
        let x = x.clone();
        let tx = tx.clone();
        model.set_solution_callback(move |model| 
            if let Ok(xx) = model.primal_solution(SolutionType::Integer, &x) {
                tx.send(iproduct!(0..n,0..n).zip(xx.iter()).filter_map(|((i,j),&x)| if x > 0.5 { Some((i,j)) } else { None } ).collect::<Vec<(usize,usize)>>());
            });
        model.set_callback(move || {
            loop {
                match rx.try_recv() {
                    Ok(Command::Terminate) => return Callback::Break,
                    Err(mpsc::TryRecvError::Empty) => return Callback::Continue,
                    Err(mpsc::TryRecvError::Disconnected) => return Callback::Break,
                }
            }
            Callback::Continue
        });
    }
   
    for it in 0.. {
        println!("--------------------\nIteration {}",it);
        model.solve();

        //println!("\nsolution cost: {}", model.primal_objective(SolutionType::Default).unwrap());
        //println!("\nsolution:");

        let mut cycles : Vec<Vec<[usize;2]>> = Vec::new();

        for i in 0..n {
            let xi = model.primal_solution(SolutionType::Default, &(&x).index([i..i+1, 0..n])).unwrap();
            //println!("x[{{}},:] = {:?}",xi);

            for (j,_xij) in xi.iter().enumerate().filter(|(_,v)| **v > 0.5) {
                if let Some(c) = cycles.iter_mut()
                    .find(|c| c.iter().filter(|ij| ij[0] == i || ij[1] == i || ij[0] == j || ij[1] == j ).count() > 0) {
                    c.push([i,j]);
                }
                else {
                    cycles.push(vec![ [ i,j ] ]);
                }
            }
        }

        //println!("\ncycles: {:?}",cycles);

        if cycles.len() == 1 {
            if let Ok(xx) = model.primal_solution(SolutionType::Integer, &x) {
                tx.send(iproduct!(0..n,0..n).zip(xx.iter()).filter_map(|((i,j),&x)| if x > 0.5 { Some((i,j)) } else { None } ).collect::<Vec<(usize,usize)>>());
            }
            break;
        }

        for c in cycles.iter_mut() {
            c.sort_by_key(|i| i[0]*n+i[1]);
            let ni = c.len();
            model.constraint(Some(format!("cycle-{:?}",c).as_str()), 
                         x.dot(matrix::sparse([n,n], c.to_vec(), vec![1.0; ni])), 
                         less_than((ni-1) as f64));
        }
    }
}


fn random_n_swap<T>(source : & mut Vec<T>, n : usize) {
    let mut rng = rand::rng();
    for k in 0..n.max(source.len()) {
        let i = rng.random_range(k..source.len());
        source.swap(k,i);
    }
}

