
use itertools::{izip, iproduct};


pub trait Matrix {
    fn shape(&self) -> [usize; 2];
    fn width(&self) -> usize { self.shape()[1] }
    fn height(&self) -> usize { self.shape()[0] } 
    fn nnz(&self) -> usize;
    fn data(&self) -> &[f64];
    fn sparsity<'a>(& 'a self) -> Option<& 'a [usize]>;


    fn transpose(&self) -> Self;
    fn mul_scalar(self, s : f64) -> Self;
    
    fn extract(self) -> ([usize; 2],Vec<f64>,Option<Vec<usize>>);
    fn extract_full(self) -> ([usize; 2],Vec<f64>);
}


/// Represents a dense matrix.
#[derive(Clone)]
pub struct DenseMatrix {
    shape  : [usize;2],
    data : Vec<f64>
}

impl Matrix for DenseMatrix {
    fn shape(&self) -> [usize; 2] { self.shape }
    fn nnz(&self) -> usize { self.data.len() }
    fn data(&self) -> &[f64] { self.data.as_slice() }
    fn sparsity<'a>(&'a self) -> Option<& 'a [usize]> { None }


    fn transpose(&self) -> DenseMatrix {
        let shape = [self.shape[1],self.shape[0]];
        let data : Vec<f64> = (0..self.shape[1]).map(|j| self.data[j..].iter().step_by(self.shape[1])).flat_map(|it| it.clone()).map(|&i| i).collect();
        DenseMatrix { shape, data }
    }
    fn mul_scalar(mut self, s : f64) -> DenseMatrix {
       self.data.iter_mut().for_each(|v| *v *= s);
       self
    }
    
    fn extract(self) -> ([usize;2],Vec<f64>,Option<Vec<usize>>) { (self.shape,self.data,None) }

    fn extract_full(self) -> ([usize; 2],Vec<f64>) { (self.shape,self.data) }
}

impl DenseMatrix {
    pub fn new(height : usize, width : usize, data : Vec<f64>) -> DenseMatrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        DenseMatrix{
            shape : [height,width],
            data
        }
    }
    //pub fn data(&self) -> &[f64] { self.data.as_slice() }
    //pub fn to_vec(&self) -> Vec<f64> { self.data.clone() }
}

/// Represents a sparse matrix.
#[derive(Clone)]
pub struct SparseMatrix {
    shape  : [usize; 2],
    sp   : Vec<usize>,
    data : Vec<f64>,
}

impl Matrix for SparseMatrix {
    fn shape(&self) -> [usize; 2] { self.shape }
    fn nnz(&self) -> usize { self.data.len() }
    fn data(&self) -> &[f64] { self.data.as_slice() }
    fn sparsity<'a>(&'a self) -> Option<& 'a [usize]> { Some(self.sp.as_slice()) }


    fn transpose(&self) -> SparseMatrix {
        let shape = [self.shape[1],self.shape[0]];
        let nnz = self.data.len();
        let mut data = vec![0.0; nnz];
        let mut sp   = vec![0usize; nnz];
        let mut ptr  = vec![0usize; self.shape[1]+1];

        self.sp.iter().for_each(|&i| unsafe { *ptr.get_unchecked_mut(i % shape[1] + 1) += 1; } );
        _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });

        for (&k,&v) in self.sp.iter().zip(self.data.iter()) {
            let (i,j) = (k / self.shape[1], k % self.shape[1]);
            unsafe {
                let p = *ptr.get_unchecked(k);
                *ptr.get_unchecked_mut(j) += 1;
                *sp.get_unchecked_mut(p) = j*shape[0]+i;
                *data.get_unchecked_mut(p) = v;
            }
        }

        SparseMatrix { shape, data, sp }
    }
    fn mul_scalar(mut self, s : f64) -> SparseMatrix {
       self.data.iter_mut().for_each(|v| *v *= s);
       self
    }
    
    fn extract(self) -> ([usize;2],Vec<f64>,Option<Vec<usize>>) { (self.shape,self.data,Some(self.sp)) }

    fn extract_full(self) -> ([usize; 2],Vec<f64>) {
        let mut data = vec![0.0; self.shape[0]*self.shape[1]];
        for (&i,&c) in self.sp.iter().zip(self.data.iter()) {
            unsafe{ *data.get_unchecked_mut(i) = c; }
        }

        (self.shape,data) 
    }
}




impl SparseMatrix {
    pub fn new(height : usize, width : usize, sparsity : &[[usize;2]], coefficients : &[f64]) -> SparseMatrix {
        if sparsity.len() != coefficients.len() { panic!("Mismatching data dimensions"); }
        if sparsity.iter().any(|&i| i[0] >= height || i[1] >= width) {
            panic!("Sparsity pattern out of bounds");
        }
        let nnz = coefficients.len();

        if sparsity.iter().zip(sparsity[1..].iter()).all(|(i0,i1)| i0[0] < i1[0] || (i0[0] == i1[0] && i0[1] < i1[1])) {
            //sorted
            SparseMatrix{ shape : [height,width],
                          sp    : sparsity.iter().map(|&i| i[0]*width+i[1]).collect(),
                          data  : coefficients.to_vec() }
        }
        else {
            let mut sp   = vec![0usize; nnz];
            let mut data = vec![0.0; nnz];
            let mut perm = vec![0usize;nnz];
            let mut ptr  = vec![0usize; usize::max(height,width)+1];

            sparsity.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i[1]+1) += 1; } );
            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
            sparsity.iter().enumerate().for_each(|(i,&si)| unsafe{ *perm.get_unchecked_mut(*ptr.get_unchecked(si[1])) = i; *ptr.get_unchecked_mut(si[1]) += 1; });

            ptr.iter_mut().for_each(|p| *p = 0);
            sparsity.iter().for_each(|i| unsafe { *ptr.get_unchecked_mut(i[0]+1) += 1; } );
            _ = ptr.iter_mut().fold(0,|c,v| { *v += c; *v });
            for &p in perm.iter() {
                let i = unsafe{ *sparsity.get_unchecked(p) };
                let ti = unsafe{ *ptr.get_unchecked(i[0]) };
                unsafe { 
                    *sp.get_unchecked_mut(ti) = i[0] * width + i[1];
                    *data.get_unchecked_mut(ti) = *coefficients.get_unchecked(p);
                    *ptr.get_unchecked_mut(i[0]) += 1;
                }
            }

            SparseMatrix{ shape : [height,width], sp,data}
        }
    }
    pub fn shape(&self) -> [usize; 2] { self.shape }
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
    pub fn sparsity(&self) -> &[usize] { self.sp.as_slice() }

    pub fn get_flat_data(self) -> (Vec<usize>,Vec<f64>) {
        (self.sp,self.data)
    }
}


//impl Matrix for DenseMatrix {
//    fn shape(&self) -> [usize; 2] { self.shape }
//    fn nnz(&self) -> usize { self.data.len() }
//    fn data(&self) -> Vec<f64> { self.data.clone() }
//    fn sparsity(&self) -> Vec<usize> { (0..self.data.len()).collect() }
//
//
//    fn transpose(&self) -> DenseMatrix {
//        let shape = [self.shape[1],self.shape[0]];
//        let data : Vec<f64> = (0..self.shape[1]).map(|j| self.data[j..].iter().step_by(self.shape[1])).flat_map(|it| it.clone()).collect();
//        DenseMatrix { shape, data }
//    }
//    fn mul_scalar(self, s : f64) -> DenseMatrix {
//       self.data.iter_mut().for_each(|v| *v *= s);
//       self
//    }
//}








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
            shape  : [height,width],
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
            shape : [height,width],
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
        shape : [height,width],
        sp  : perm.iter().map(|&p| { let i = unsafe{data.get_unchecked(p)}; i.0*width+i.1 }).collect(),
        data : perm.iter().map(|&p| unsafe{data.get_unchecked(p)}.2 ).collect()
    }
}

pub fn ones(height : usize, width : usize) -> DenseMatrix {
    DenseMatrix{
        shape : [height,width],
        data : vec![1.0; height*width]
    }
}
pub fn diag(data : &[f64]) -> SparseMatrix {
    SparseMatrix{
        shape : [data.len(),data.len()],
        sp  : (0..data.len()*data.len()).step_by(data.len()+1).collect(),
        data : data.to_vec()
    }
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
    fn mul(mut self,rhs : f64) -> DenseMatrix {
        self.data.iter_mut().for_each(|v| *v *= rhs);
        self
    }
}

impl std::ops::Mul<DenseMatrix> for f64 {
    type Output = DenseMatrix;
    fn mul(self,mut rhs : DenseMatrix) -> DenseMatrix {
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
            shape,
            data
        }
    }
}
