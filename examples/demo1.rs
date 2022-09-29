extern crate mosekmodel;

use mosekmodel::*;

fn main() -> Result<(),String> {
    let mut m = Model::new(Some("SuperModel"));

    let x1 = m.variable(Some("x1"), vec![4,5]);
    let x2 = m.variable(Some("x2"), vec![4]);
    let y1 = m.variable(Some("y1"), greater_than(vec![1.0,2.0,3.0,4.0]));
    let y2 = m.variable(Some("y2"), greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(vec![2,2]));
    let z  = m.variable(Some("z"),  greater_than(vec![1.0,4.0]).with_shape_and_sparsity(vec![2,2],vec![0,3]));
    Ok(())
}
