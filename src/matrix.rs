
use itertools::{izip, iproduct};
use super::expr::{ExprRightMultipliable,ExprTrait,ExprTrait0,ExprTrait1,ExprTrait2,ExprMulLeftDense,ExprMulRightDense};


/// Create a dense matrix from data
///
/// # Arguments
/// - `height` - Height of matrix
/// - `width` - Width if matrix
/// - `data` - Coefficients of data (consumed). This must contain exactly `height * width`
///   elements.
pub fn dense(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix { DenseMatrix::new(height,width,data) }
/// Create a sparse matrix.
///
/// # Arguments
/// - `height`
/// - `width`
/// - `subi` Row indexes of non-zeros
/// - `subj` Column indexes of non-zeros
/// - `cof` Non-zero coefficients
///
/// Note that the lengths if `subi`, `subj` and `cof` must be the same. They must not define
/// duplicate entries, and the number of non-zeros must be at most `height * width`.
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
    if subi.len() < 2 || izip!(subi.iter(),subi[1..].iter(),subj.iter(),subj[1..].iter()).all(|(i0,i1,j0,j1)| i0 < i1 || (i0 == i1 && j0 < j1)) {
        SparseMatrix{
            dim  : [height,width],
            sp   : subi.iter().zip(subj.iter()).map(|(i,j)| i * width + j).collect(),
            data : cof.to_vec()
        }
    }
    else {
        let mut perm : Vec<usize> = (0..subi.len()).collect();
        perm.sort_by_key(|&k| unsafe{(*subi.get_unchecked(k),*subj.get_unchecked(k)) });

        if ! perm.iter().zip(perm[1..].iter()).map(|(&p0,&p1)| unsafe{(*subi.get_unchecked(p0),*subi.get_unchecked(p1),*subj.get_unchecked(p0),*subj.get_unchecked(p1))})
            .all(|(i0,i1,j0,j1)| i0 < i1 || (i0 == i1 && j0 < j1)) {
                panic!("Matrix contains duplicates");
            }

        SparseMatrix{
            dim : [height,width],
            sp  : perm.iter().map(|&p| unsafe{*subi.get_unchecked(p)}*width+unsafe{*subj.get_unchecked(p)}).collect(),
            data : perm.iter().map(|&p| unsafe{*cof.get_unchecked(p)}).collect()
        }
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
        dim : [height,width],
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
        dim : [data.len(),data.len()],
        sp  : (0..data.len()*data.len()).step_by(data.len()+1).collect(),
        data : data.to_vec()
    }
}

/// Represents a dense matrix.
#[derive(Clone)]
pub struct DenseMatrix {
    dim  : (usize,usize),
    data : Vec<f64>
}

/// Represents a sparse matrix.
#[derive(Clone)]
pub struct SparseMatrix {
    dim  : [usize; 2],
    sp   : Vec<usize>,
    data : Vec<f64>,
}

/// Represents a Dense n-dimensional array
#[derive(Clone)]
pub struct DenseNDArray<const N : usize> {
    dim : [usize; N],
    data : Vec<f64>,
}

impl<const N : usize> DenseNDArray<N> {
    pub fn shape(&self) -> [usize; N] { self.dim }
    pub fn extract(self) -> ([usize;N],Vec<f64>) { (self.dim,self.data) }
}


/// represents a Sparse n-dimensional array
#[derive(Clone)]
pub struct SparseNDArray<const N : usize> {
    dim : [usize; N],
    sp  : Vec<usize>,
    data : Vec<f64>,
}


impl SparseMatrix {
    pub fn shape(&self) -> [usize; 2] { self.dim }
    pub fn height(&self) -> usize { self.dim[0] }
    pub fn width(&self) -> usize { self.dim[1] }
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
    pub fn sparsity(&self) -> &[usize] { self.sp.as_slice() }

    pub fn get_flat_data(self) -> (Vec<usize>,Vec<f64>) {
        (self.sp,self.data)
    }
}


impl DenseMatrix {
    pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        DenseMatrix{
            dim : (height,width),
            data
        }
    }
    pub fn shape(&self) -> [usize; 2] { let (d0,d1) = self.dim; [d0,d1] }
    pub fn dim(&self) -> (usize,usize) { self.dim }
    pub fn height(&self) -> usize { self.dim.0 }
    pub fn width(&self) -> usize { self.dim.1 }
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
    pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}


//impl<E:ExprTrait2> ExprRightMultipliable<2,E> for DenseMatrix {
//    type Result = ExprMulRightDense<E>;
//    fn mul_right(self,other : E) -> Self::Result { 
//        other.mul_right_dense(self)
//    }
//}
//
//impl<E:ExprTrait1> ExprRightMultipliable<1,E> for DenseMatrix where E : ExprTrait1 {
//    type Result = ExprReshapeOneRow<2,1,ExprMulRightDense<ExprReshapeOneRow<1,2,E>>>;
//    fn mul_right(self,other : E) -> Self::Result { 
//        other.mul_right_dense(self) 
//    }
//}
//
//impl<E:ExprTrait2> ExprLeftMultipliable<2,E> for DenseMatrix where E : ExprTrait2 {
//    type Result = ExprMulLeftDense<E>;
//    fn mul(self,other : E) -> Self::Result { other.mul_left_dense(self) }
//}
//
//impl<E:ExprTrait1> ExprLeftMultipliable<1,E> for DenseMatrix where E : ExprTrait1 {
//    type Result = ExprReshapeOneRow<2,1,ExprMulLeftDense<ExprReshapeOneRow<1,2,E>>>;
//    fn mul(self,other : E) -> Self::Result { other.mul_left_dense(self) }
//}
//
//// Trait defining the behaviour of multiplying different shapes of expressions on a dense matrix
//pub trait DenseMatrixMulLeftExpr {
//    type Output;
//    fn rev_mul(self,m : DenseMatrix) -> Self::Output;
//}
//// Defines the behaviour when multiplying a 2D expression on a dense matrix
//impl<E> DenseMatrixMulLeftExpr for E where E : ExprTrait<2> {
//    type Output = ExprMulLeftDense<E>;
//    fn rev_mul(self,m : DenseMatrix) -> Self::Output {
//        self.mul_left_dense(m)
//    }
//}
//
//
//// Defines the behaviour when multiplying a 1D expression on a dense matrix
////impl<E> DenseMatrixMulLeftExpr for E where E : ExprTrait<0> {
////    type Output = ExprReshapeOneRow<2,1,ExprMulLeftDense<ExprReshapeOneRow<1,2,E>>>;
////    fn rev_mul(self,m : DenseMatrix) -> Self::Output {
////        self.mul_left_dense(m)
////    }
////}
//
//impl DenseMatrix {
//    pub fn mul<E>(self, other : E) -> E::Output where E : DenseMatrixMulLeftExpr {
//        other.rev_mul(self)
//    }
//}
//
//


impl std::ops::Mul<f64> for DenseMatrix {
    type Output = DenseMatrix;
    fn mul(self,rhs : f64) -> DenseMatrix {
        self.data.iter_mut().for_each(|v| *v *= rhs);
        self
    }
}

impl std::ops::Mul<DenseMatrix> for f64 {
    type Output = DenseMatrix;
    fn mul(self,rhs : DenseMatrix) -> DenseMatrix {
        rhs.data.iter_mut().for_each(|v| *v *= self );
        rhs
    }
}

impl std::ops::MulAssign<f64> for DenseMatrix {
    fn mul_assign(&mut self, rhs: f64) {
        self.data.iter_mut().for_each(|v| *v *= rhs);
    } 
}

impl std::ops::Mul<DenseMatrix> for DenseMatrix {
    type Output = DenseMatrix;

    fn mul(self,rhs : DenseMatrix) -> DenseMatrix {
        let lhsshape = self.shape();
        let rhsshape = rhs.shape();
        if lhsshape[1] != rhsshape[0] { panic!("Mismatching operand dimensions"); }
        // naive implementation:
        
        let shape = [lhsshape[0],rhsshape[1]];

        let data = iproduct!(0..lhsshape[0],0..rhsshape[1]).map(|(i,j)| self.data[i*lhsshape[1]..].iter().zip(rhs.data[j..].iter().step_by(rhsshape[1])).map(|(&v0,&v1)| v0*v1).sum() ).collect();

        DenseMatrix{
            dim : shape.into(),
            data
        }
    }
}
