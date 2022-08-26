extern crate mosekmodel;

use mosekmodel::{Model};

fn main() -> Result<(),String> {
    let m = Model::new(Some("SuperModel"));

    let x1 = m.variable(Some("x1"), &[4,5]);
    let x2 = m.variable(Some("x2"), &[4]);
    let y1 = m.variable(Some("y1"), greater_than(&[1.0,2.0,3.0,4.0]));
    let y2 = m.variable(Some("y2"), greater_than(&[1.0,2.0,3.0,4.0]).with_shape(&[2,2]));
    let z  = m.variable(Some("z"),  greater_than(&[1.0,4.0]).with_shape_and_sparsity(&[2,2],&[0,3]));
    //let y = m.symmetric_variable(Some("y"), 4);

    let c = m.variable(Some("c"),
                       Expr::from_var(z).into_diag())
}
