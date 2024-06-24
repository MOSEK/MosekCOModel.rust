extern crate cairo;
extern crate mosekmodel;

use std::cell::RefCell;
use std::rc::Rc;
use gtk::prelude::*;
use itertools::izip;

use cairo::Context;
use gtk::{Application, DrawingArea, ApplicationWindow};
use mosekmodel::expr::*;
use mosekmodel::matrix::SparseMatrix;
use mosekmodel::{hstack, in_rotated_quadratic_cones, unbounded, nonnegative,equal_to,zero, Model, Sense, SolutionType};

use std::fs::File;
use std::io::{BufRead,BufReader};

const APP_ID : &str = "com.mosek.truss-linear";

#[allow(non_snake_case)]
#[derive(Clone,Default)]
struct DrawData {
    points         : Vec<[f64;2]>,
    node_type      : Vec<bool>,
    arcs           : Vec<(usize,usize)>,
    external_force : Vec<Vec<[f64;2]>>,
    total_material_volume : f64,
    kappa          : f64,
    arc_vol_stress : Option<(Vec<f64>,Vec<f64>)>,
}

impl DrawData {
    /// Read file. File format:
    /// ```
    /// kappa FLOAT
    /// w     FLOAT
    /// nodes
    ///     FLOAT FLOAT "X"?
    ///     ...
    /// arcs
    ///     INT INT
    ///     ...
    /// forces
    ///     INT FLOAT FLOAT
    ///     ...
    /// forces ...
    ///     ...
    /// ```
    fn from_file(filename : &str) -> DrawData {
        enum State {
            Base,
            Nodes,
            Arcs,
            Forces,
        }
        let mut dd = DrawData::default();

        let f = File::open(filename).unwrap();
        let br = BufReader::new(f);
        let mut state = State::Base;
        let mut forces : Vec<Vec<(usize,[f64;2])>> = Vec::new();

        for (lineno,l) in br.lines().enumerate() {
            let l = l.unwrap();
            //println!("{}> {}",lineno+1,l.trim_end());

            if      let Some(rest) = l.strip_prefix("kappa")  { dd.kappa = rest.trim().parse().unwrap(); state = State::Base; }
            else if let Some(rest) = l.strip_prefix("w")      { dd.total_material_volume = rest.trim().parse().unwrap(); state = State::Base; }
            else if l.starts_with("nodes")  { state = State::Nodes; }
            else if l.starts_with("arcs")   { state = State::Arcs; }
            else if l.starts_with("forces") { forces.push(Vec::new()); state = State::Forces; }
            else {
                let llstrip = l.trim_start();
                if llstrip.is_empty() || llstrip.starts_with('#') {
                    // comment
                }
                else if ! l.starts_with(' ') {
                    panic!("Invalid data at line {}: '{}'",lineno+1,l.trim_end());
                }
                else {
                    match state {
                        State::Base   => {},
                        State::Nodes  => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let x : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let y : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            if let Some("X") = it.next() { dd.node_type.push(true); }
                            else { dd.node_type.push(false); }
                            dd.points.push([x,y]);
                        },
                        State::Arcs   => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let i : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let j : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            dd.arcs.push((i,j));
                        },
                        State::Forces => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let a : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let x : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let y : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            forces.last_mut().unwrap().push((a,[x,y]));
                        }
                    }
                }
            }
        }
        
        // check

        if forces.is_empty() {
            panic!("Missing forces section");
        }
        if *dd.arcs.iter().map(|(i,j)| i.max(j)).max().unwrap() >= dd.points.len() {
            panic!("Arc end-point index out of bounds");
        }

        for ff in forces.iter() {
            if ff.iter().map(|v| v.0).max().unwrap() >= dd.points.len() {
                panic!("Force node index out of bounds");
            }

            let mut forcevec = vec![[0.0,0.0]; dd.points.len()];
            for &(i,f) in ff { forcevec[i] = f; }

            dd.external_force.push(forcevec);
        }
        println!("Truss:\n\t#nodes: {}\n\t#arcs: {}",dd.points.len(),dd.arcs.len());

        dd 
    }
}

const D : usize = 2;

pub fn main() {
    let mut args = std::env::args();

    if let None = args.next() {  
        println!("Syntax: truss filename");
    }
    let filename = if let Some (filename) = args.next() { filename } 
    else {
        println!("Syntax: truss filename");
        return;
    };

    let dosolve = if let Some(arg) = args.next() { println!("arg = '{}'",arg); arg != "-x" } else { true };

    let mut drawdata = DrawData::from_file(&filename);
    let numnodes = drawdata.points.len();
    let numarcs  = drawdata.arcs.len();

    // b is a parameter such that 
    // b is a (numarcs x (D*numnodes)) matrix. Rows are indexes by nodes, colunms are indexed by arcs,
    // so each column has an associated (i,j) ∊ A. The element b_{k,(i,j)} means row k, column
    // associated with (i,j). The matrix is build as
    //    b_{j,(ij)} = κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    b_{i,(ij)} = -κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    0 everywhere else.
    if dosolve {
        let sqrtkappa = drawdata.kappa.sqrt();
        let b = SparseMatrix::from_iterator(
            numarcs, 
            numnodes*D, 
            drawdata.arcs.iter().enumerate().flat_map(|(arci,&(i,j))| {
                let pi = drawdata.points[i];
                let pj = drawdata.points[j];
                let ti = drawdata.node_type[i];
                let tj = drawdata.node_type[j];

                
                let d = (pj[0]-pi[0], pj[1]-pi[1]);
                let sqrnormd = d.0.powi(2) + d.1.powi(2);
                
                std::iter::once(           (arci, j*D,   if !tj { sqrtkappa * d.0 / sqrnormd } else { 0.0 }))
                    .chain(std::iter::once((arci, j*D+1, if !tj { sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
                    .chain(std::iter::once((arci, i*D,   if !ti { -sqrtkappa * d.0 / sqrnormd } else { 0.0 })))
                    .chain(std::iter::once((arci, i*D+1, if !ti { -sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
            }));

        let numforceset = drawdata.external_force.len();
        let mut m = Model::new(Some("Truss"));
        let tau = m.variable(Some("tau"), unbounded());
        //let tau = m.variable(Some("tau"), equal_to(20.0));
        
        

        let sigma = m.variable(Some("sigma"), unbounded().with_shape(&[numforceset,numarcs]));
        let t = m.variable(Some("t"),unbounded().with_shape(&[numforceset,numarcs]));
        let s = m.variable(Some("s"),unbounded().with_shape(&[numforceset,numarcs]));
        let w = m.variable(Some("w"),equal_to(drawdata.total_material_volume));

        // (1)
        m.objective(None, Sense::Minimize, &tau);

        for (fi,forces) in drawdata.external_force.iter().enumerate() {
            let t     = (&t).slice(&[fi..fi+1,0..numarcs]).reshape(&[numarcs]);
            let s     = (&s).slice(&[fi..fi+1,0..numarcs]).reshape(&[numarcs]);
            let sigma = (&sigma).slice(&[fi..fi+1,0..numarcs]).reshape(&[numarcs]);

            // (2)
            m.constraint(Some("t_sigma_s"),
                         &hstack![t.clone().reshape(&[numarcs,1]),
                                  sigma.clone().reshape(&[numarcs,1]),
                                  s.clone().reshape(&[numarcs,1])],
                         in_rotated_quadratic_cones(&[numarcs,3], 1));
            // (3)
            m.constraint(Some("sum_sigma"),
                         &tau.clone().sub(sigma.clone().sum()),
                         nonnegative());
                
            // (4) 
            m.constraint(Some("total_volume"),
                         &t.clone().sum().sub(w.clone()),
                         zero());
            // (5)
            let f : Vec<f64> = forces.iter().flat_map(|row| row.iter()).cloned().collect();
            m.constraint(Some("force_balance"), 
                         &s.clone().square_diag().mul(b.clone()).sum_on(&[1]),
                         equal_to(f));
        }

        m.solve();

        m.write_problem("truss.ptf");

        if let (Ok(tsol),Ok(ssol)) = (m.primal_solution(SolutionType::Default,&t),
                                      m.primal_solution(SolutionType::Default,&s)) {
            drawdata.arc_vol_stress = Some((tsol.to_vec(),ssol.to_vec()));
        }
        else {
            println!("ERROR: No solution!");
        }
    }
    let app = Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_activate(move | app : &Application | build_ui(app,&drawdata));

    let r = app.run_with_args::<&str>(&[]);
    println!("Main loop exit!");
}


#[allow(non_snake_case)]
fn build_ui(app   : &Application,
            ddata : &DrawData)
{
    let data = Rc::new(RefCell::new(ddata.clone()));

    let darea = DrawingArea::builder()
        .width_request(800)
        .height_request(800)
        .build();

    // Redraw callback
    {
        let data = data.clone();
        darea.set_draw_func(move |widget,context,w,h| redraw_window(widget,context,w,h,&data.borrow()));
    }

    let window = ApplicationWindow::builder()
        .application(app)
        .title("Hello Löwner-John")
        .child(&darea)
        .build();


    window.present();
}

fn norm<const N : usize>(p : &[f64;N]) -> f64 { p.iter().cloned().map(f64::abs).sum::<f64>().sqrt() }
fn vecsub<const N : usize>(lhs : &[f64;N], rhs : &[f64;N]) -> [f64;N] { 
    let mut r = [0.0;N];
    for (res,&l,&r) in izip!(r.iter_mut(),lhs.iter(),rhs.iter()) {
        *res = l-r;
    }
    r
}

fn vecscale<const N : usize>(s : f64, v : &[f64;N]) -> [f64;N] { 
    let mut r = [0.0;N];
    for (r,&v) in r.iter_mut().zip(v.iter()) { *r = s * v; }
    r
}

fn vecnormalize<const N : usize>(v : &[f64;N]) -> [f64;N] { 
    vecscale(1.0/norm(v),v)
}

fn vecadd<const N : usize>(lhs : &[f64;N], rhs : &[f64;N]) -> [f64;N] { 
    let mut r = [0.0;N];
    for (res,&l,&r) in izip!(r.iter_mut(),lhs.iter(),rhs.iter()) {
        *res = l+r;
    }
    r
}

#[allow(non_snake_case)]
fn redraw_window(_widget : &DrawingArea, context : &Context, w : i32, h : i32, data : &DrawData) {
    context.set_source_rgb(1.0, 1.0, 1.0);
    _ = context.paint();

    let w : f64 = w.into();
    let h : f64 = h.into();
    let s = w.min(h);
   
    let crop = data.points.iter().fold((0.0,0.0,0.0,0.0), |b,p| ( p[0].min(b.0),p[1].min(b.1),p[0].max(b.0),p[1].max(b.1)));
    println!("box   = {:?}",crop);
    let crop = (crop.0 - (crop.2-crop.0)*0.1,
                crop.1 - (crop.3-crop.1)*0.1,
                crop.2 + (crop.2-crop.0)*0.1,
                crop.3 + (crop.3-crop.1)*0.1);
    println!("  -> box {:?}",crop);
    let boxmax = (crop.2-crop.0).abs().max( (crop.3-crop.1).abs());
    let scale = s / boxmax;

    println!("boxmax = {}",boxmax);
    println!("scale = {}",scale);
    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
    //context.translate(0,-h as f64);
    context.scale(scale, -scale);
    context.translate(-crop.0,-boxmax-crop.1);
    let mx = context.matrix();

    // Forces
    context.set_line_width(3.0);
    context.set_source_rgb(1.0, 0.8, 0.8);
    for forces in data.external_force.iter() {
        for (f,p) in forces.iter().zip(data.points.iter()) {
            if norm(f) > 0.0 {
                context.move_to(p[0],p[1]);
                context.line_to(p[0]+f[0],p[1]+f[1]);

                // arrow head
                let v = [ f[1]-f[0], -f[0]-f[1] ];
                let v = vecscale(0.1 / norm(&v),&v);

                context.move_to(p[0]+f[0]+v[0],p[1]+f[1]+v[1]);
                context.line_to(p[0]+f[0],p[1]+f[1]);
                context.line_to(p[0]+f[0]+v[1],p[1]+f[1]-v[0]);

                context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
                _ = context.stroke();
                context.set_matrix(mx);
            }
        }
    }
    // ARCS
    let numarcs = data.arcs.len();

    context.set_source_rgb(0.0, 0.0, 0.0);    
    if let Some((ref volume,ref stress)) = data.arc_vol_stress {
        for (volume,stress) in volume.chunks(numarcs).zip(stress.chunks(numarcs)) {
            for (&(i,j),&v,&s) in izip!(data.arcs.iter(),volume.iter(),stress.iter()) {
                let pi = data.points[i];
                let pj = data.points[j];
            
                //println!("Arc volume = {:?}",volume);

                if v > 1.0e-4 {
                    let w = v / norm(&[ pj[0]-pi[0], pj[1]-pi[1] ]);
                    if s < 0.0 {
                        context.set_source_rgb(0.7, 0.0, 0.0);    
                    } 
                    else {
                        context.set_source_rgb(0.0, 0.7, 0.0);    
                    }
                    context.set_line_width(w*2.0);
                    context.move_to(pi[0], pi[1]);
                    context.line_to(pj[0], pj[1]);

                    context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
                    _ = context.stroke();
                    context.set_matrix(mx);
                }
            }
        }
    }
    else {
        context.set_source_rgb(0.0, 0.0, 0.0);    
        context.set_line_width(1.0);
        for &(i,j) in data.arcs.iter() {
            let pi = data.points[i];
            let pj = data.points[j];
        
            context.move_to(pi[0], pi[1]);
            context.line_to(pj[0], pj[1]);

            context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
            _ = context.stroke();
            context.set_matrix(mx);
        }
    }

    // NODES
    context.set_source_rgb(0.0, 0.0, 0.5);
    context.set_line_width(1.0);
    for p in data.points.iter() {
        context.arc(p[0],p[1],5.0/scale,0.0,std::f64::consts::PI*2.0);

        context.set_matrix(cairo::Matrix::new(1.0,0.0,0.0,1.0,0.0,0.0));
        _ = context.fill();
        context.set_matrix(mx);
    }

    context.set_source_rgb(0.0, 0.0, 0.0);
    context.set_line_width(1.0);
    for (&f,p) in data.node_type.iter().zip(data.points.iter()) {
        if f {
            context.arc(p[0],p[1],15.0/scale,0.0,std::f64::consts::PI*2.0);
            context.set_matrix(cairo::Matrix::new(3.0,0.0,0.0,3.0,0.0,0.0));
            _ = context.stroke();
            context.set_matrix(mx);
        }
    }
    
}

