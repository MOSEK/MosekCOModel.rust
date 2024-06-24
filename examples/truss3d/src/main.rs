extern crate mosekmodel;

mod truss;

use mosekmodel::expr::*;
use mosekmodel::matrix::SparseMatrix;
use mosekmodel::{hstack, in_rotated_quadratic_cones, unbounded, nonnegative,equal_to,zero, Model, Sense, SolutionType};
use truss::Truss;

#[derive(Clone,Default)]
struct DrawData {
    data           : Truss,
    /// solution
    arc_vol_stress : Option<(Vec<f64>,Vec<f64>)>,
}


const D : usize = 3;

pub fn main() {
    let mut args = std::env::args();

    if let None = args.next() {  
        println!("Syntax: truss3d filename");
    }
    let filename = if let Some (filename) = args.next() { filename } 
    else {
        println!("Syntax: truss3d filename");
        return;
    };

    let dosolve = if let Some(arg) = args.next() { println!("arg = '{}'",arg); arg != "-x" } else { true };

    let mut drawdata = DrawData{ data : Truss::from_file(&filename), arc_vol_stress : None };
    let numnodes = drawdata.data.points.len();
    let numarcs  = drawdata.data.arcs.len();

    // b is a parameter such that 
    // b is a (numarcs x (D*numnodes)) matrix. Rows are indexes by nodes, colunms are indexed by arcs,
    // so each column has an associated (i,j) ∊ A. The element b_{k,(i,j)} means row k, column
    // associated with (i,j). The matrix is build as
    //    b_{j,(ij)} = κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    b_{i,(ij)} = -κ(p_j-p_i)/||p_j-p_i||^2  for (i,j) ∊ A
    //    0 everywhere else.
    if dosolve {
        let sqrtkappa = drawdata.data.kappa.sqrt();
        let b = SparseMatrix::from_iterator(
            numarcs, 
            numnodes*D, 
            drawdata.data.arcs.iter().enumerate().flat_map(|(arci,&(i,j))| {
                let pi = drawdata.data.points[i];
                let pj = drawdata.data.points[j];
                let ti = drawdata.data.node_type[i];
                let tj = drawdata.data.node_type[j];

                
                let d = (pj[0]-pi[0], pj[1]-pi[1]);
                let sqrnormd = d.0.powi(2) + d.1.powi(2);
                
                std::iter::once(           (arci, j*D,   if !tj { sqrtkappa * d.0 / sqrnormd } else { 0.0 }))
                    .chain(std::iter::once((arci, j*D+1, if !tj { sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
                    .chain(std::iter::once((arci, i*D,   if !ti { -sqrtkappa * d.0 / sqrnormd } else { 0.0 })))
                    .chain(std::iter::once((arci, i*D+1, if !ti { -sqrtkappa * d.1 / sqrnormd } else { 0.0 })))
            }));

        let numforceset = drawdata.data.external_force.len();
        let mut m = Model::new(Some("Truss"));
        let tau = m.variable(Some("tau"), unbounded());
        
        let sigma = m.variable(Some("sigma"), unbounded().with_shape(&[numforceset,numarcs]));
        let t = m.variable(Some("t"),unbounded().with_shape(&[numforceset,numarcs]));
        let s = m.variable(Some("s"),unbounded().with_shape(&[numforceset,numarcs]));
        let w = m.variable(Some("w"),equal_to(drawdata.data.total_material_volume));

        // (1)
        m.objective(None, Sense::Minimize, &tau);

        for (fi,forces) in drawdata.data.external_force.iter().enumerate() {
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
    




    println!("Main loop exit!");
}




