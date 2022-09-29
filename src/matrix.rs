//use itertools::{izip};
use super::expr::{ExprRightMultipliable,ExprTrait,ExprMulLeftDense,ExprMulRightDense};

pub fn dense(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix { DenseMatrix::new(height,width,data) }
pub fn sparse(height : usize, width : usize,
              subi : &[usize],
              subj : &[usize],
              cof  : &[f64]) -> SparseMatrix {
    if subi.len() != subj.len() || subi.len() != cof.len() {
        panic!("Invalid matrix data")
    }

    if subi.iter().max().copied().unwrap_or(0) >= height
        || subj.iter().max().copied().unwrap_or(0) >= width {
            panic!("Invalid matrix data")
        }

    let mut perm : Vec<usize> = (0..subi.len()).collect();
    perm.sort_by_key(|&k| unsafe{(*subi.get_unchecked(k),*subj.get_unchecked(k)) });

    if ! perm.iter().zip(perm[1..].iter()).map(|(&p0,&p1)| unsafe{(*subi.get_unchecked(p0),*subi.get_unchecked(p1),*subj.get_unchecked(p0),*subj.get_unchecked(p1))})
        .all(|(i0,i1,j0,j1)| i0 < i1 || (i0 == i1 && j0 < j1)) {
            panic!("Matrix contains duplicates");
        }

    SparseMatrix{
        dim : (height,width),
        sp  : perm.iter().map(|&p| unsafe{*subi.get_unchecked(p)}*width+unsafe{*subj.get_unchecked(p)}).collect(),
        data : perm.iter().map(|&p| unsafe{*cof.get_unchecked(p)}).collect()
    }
}

pub fn from_triplets(height : usize,
                     width : usize,
                     data : &[(usize,usize,f64)]) -> SparseMatrix {
    if data.iter().max_by_key(|&v| v.0).map(|&v| v.0 >= height).unwrap_or(false)
        || data.iter().max_by_key(|&v| v.1).map(|&v| v.1 >= width).unwrap_or(false) {
            panic!("Invalid matrix data")
        }

    let mut perm : Vec<usize> = (0..data.len()).collect();
    perm.sort_by_key(|&k| { let d = unsafe{*data.get_unchecked(k)}; (d.0,d.1) });

    if ! perm.iter().zip(perm[1..].iter()).map(|(&p0,&p1)| unsafe{(*data.get_unchecked(p0),*data.get_unchecked(p1))})
        .all(|(i0,i1)| i0.0 < i1.0 || (i0.0 == i1.0 && i0.1 < i1.1)) {
            panic!("Matrix contains duplicates");
        }
    SparseMatrix{
        dim : (height,width),
        sp  : perm.iter().map(|&p| { let i = unsafe{data.get_unchecked(p)}; i.0*width+i.1 }).collect(),
        data : perm.iter().map(|&p| unsafe{data.get_unchecked(p)}.2 ).collect()
    }
}

pub fn ones(height : usize, width : usize) -> DenseMatrix {
    DenseMatrix{
        dim : (height,width),
        data : vec![1.0; height*width]
    }
}
pub fn diag(data : &[f64]) -> SparseMatrix {
    SparseMatrix{
        dim : (data.len(),data.len()),
        sp  : (0..data.len()*data.len()).step_by(data.len()+1).collect(),
        data : data.to_vec()
    }
}


#[derive(Clone)]
pub struct DenseMatrix {
    dim  : (usize,usize),
    data : Vec<f64>
}

#[derive(Clone)]
pub struct SparseMatrix {
    dim  : (usize,usize),
    sp   : Vec<usize>,
    data : Vec<f64>,
}

impl DenseMatrix {
    pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        DenseMatrix{
            dim : (height,width),
            data : data
        }
    }
    pub fn dim(&self) -> (usize,usize) { self.dim }
    pub fn height(&self) -> usize { self.dim.0 }
    pub fn width(&self) -> usize { self.dim.1 }
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
}

impl<E:ExprTrait> ExprRightMultipliable<E> for DenseMatrix {
    type Result = ExprMulRightDense<E>;
    fn mul_right(self,other : E) -> Self::Result { other.mul_right_dense(self) }
}

impl DenseMatrix {
    pub fn mul<E:ExprTrait>(self,other : E) -> ExprMulLeftDense<E> { ExprMulLeftDense::new(other,self) }
}
