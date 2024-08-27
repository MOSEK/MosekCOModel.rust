//! This module provides basic array functionality.
//!
use itertools::{izip, EitherOrBoth};
use crate::expr::{Expr, IntoExpr};
use utils::*;


/// This trait represents an 2-dimensional array, with a few functions specialized for matrixes on
/// top of functionality provided by n-dimensional arrays
pub trait Matrix  {
    /// Matrix width, number of columns
    fn width(&self) -> usize;
    /// Matrix height, number of rows
    fn height(&self) -> usize;
    /// Transpose matrix and return a new object of the same type as self.
    fn transpose(&self) -> Self;
    /// Get the shape of the matrix
    fn shape(&self) -> [usize; 2];
    /// Reshape the array - the result must have the same total number of elements as this.
    fn reshape(self,shape : [usize; 2]) -> Result<Self,()> where Self:Sized;
    /// Return number of non-zeros
    fn nnz(&self) -> usize;
    /// Return a reference to the non-zero coefficients.
    fn data(&self) -> &[f64];
    /// Return the sparsity pattern if defined. The sparsity pattern is a slice of linear indexes
    /// (rather than n-dimensional indexes) of the elements. 
    fn sparsity(&self) -> Option<&[usize]>; 
    /// Multiply all non-zeros by a scalar
    fn inplace_mul_scalar(&mut self, s : f64);
    /// Return the elements of the object, thereby destroying it.
    fn dissolve(self) -> ([usize;2],Option<Vec<usize>>,Vec<f64>);
    /// Turns a sparse matrix into a dense matrix by adding the missing zeros.
    fn to_dense(&self) -> Self;
}

///////////////////////////////////////////////////////////////////////////////

/// General n-dimensional dense or sparse array structure.
///
/// One important limitation is that the product of the dimensions of the array cannot exceed
/// `usize::MAX`.
#[derive(Clone)]
pub struct NDArray<const N : usize> {
    shape : [usize; N],
    sp    : Option<Vec<usize>>,
    data  : Vec<f64>,
}

impl Matrix for NDArray<2> {
    fn width(&self) -> usize { self.shape()[1] }
    fn height(&self) -> usize { self.shape()[0] }
    fn transpose(&self) -> NDArray<2> {
        let shape = [self.shape[1],self.shape[0]];
        if let Some(ref sp) = self.sp {
            let n = sp.len();

            let mut ptr = vec![0; self.shape[1]+1];
            sp.iter().for_each(|&i| unsafe{ *ptr.get_unchecked_mut(1 + i % self.shape[1]) += 1 });
            _ = ptr.iter_mut().fold(0,|c,p| {*p += c; *p });

            let mut rsp = vec![0usize; n];
            let mut rdata = vec![0.0; n];

            for (&k,&d) in sp.iter().zip(self.data.iter()) {
                let (i,j) = (k / self.shape[1], k % self.shape[1]); 
                let p = unsafe{ *ptr.get_unchecked(j) };
                unsafe {
                    *rsp.get_unchecked_mut(p) = j*self.shape[0] + i;
                    *rdata.get_unchecked_mut(p) = d;
                    *ptr.get_unchecked_mut(j) += 1;
                }
            }

            NDArray{
                shape,
                sp : Some(rsp),
                data:rdata
            }
        }
        else {
            let data : Vec<f64> = (0..self.shape[1]).map(|j| self.data[j..].iter().step_by(self.shape[1])).flat_map(|it| it.clone()).map(|&i| i).collect();
            NDArray { shape, sp : None, data }
        }
    }
    fn shape(&self) -> [usize; 2] { self.shape() } 
    fn reshape(self,shape : [usize; 2]) -> Result<NDArray<2>,()> { self.reshape(shape) }
    fn nnz(&self) -> usize { self.nnz() }
    fn data(&self) -> &[f64] { self.data() }
    fn sparsity(&self) -> Option<&[usize]> { self.sparsity() } 
    fn inplace_mul_scalar(&mut self, s : f64) { self.inplace_mul_scalar(s) }
    fn dissolve(self) -> ([usize;2],Option<Vec<usize>>,Vec<f64>) { self.dissolve() }
    fn to_dense(&self) -> Self { self.to_dense() }
}

impl<const N : usize> NDArray<N> {
    /// Create a new [NDArray] from data, checking that the data is valid.
    ///
    /// # Arguments
    /// - `shape` Shape of the array.
    /// - `sp` Sparsity pattern, if the array is sparse, otherwise `None`. If given, sparsity is
    ///   provided as a vector of linear indexes (rather than as n-dimensional indexes).
    /// - `data` Non-zero coefficients
    pub fn new(shape : [usize;N], sp : Option<Vec<usize>>, data : Vec<f64>) -> Result<NDArray<N>,String> { 
        // validate data
        if let Some(sp) = sp {
            if sp.len() > 1 && sp.iter().zip(sp[1..].iter()).any(|(&i0,&i1)| i1 <= i0) {
                Err("Sparsity is unsorted or contains duplicates".to_string())
            }
            else if sp.len() != data.len() {
                Err("Mismatching sparsity and data lengths".to_string())
            }
            else if sp.len() > 0 && shape.iter().product::<usize>() <= *sp.last().unwrap() {
                Err("Mismatching sparsity and shape".to_string())
            }
            else {
                Ok(NDArray{ shape,sp : Some(sp),data })
            }
        }
        else {
            let nnz : usize = shape.iter().product();
            if nnz != data.len() {
                Err("Mismatching data and shape".to_string())
            }
            else {
                Ok(NDArray{shape,sp:None,data})
            }
        }
    }

    /// Create a new sparse [NDArray] from shape and an iterator.
    ///
    /// #Arguments
    /// - `shape` an N-dimensional shape.
    /// - `it` An iterator where each item `([usize;N],f64)`. The iterator must generate at most
    ///   `shape.iter().product()` elements. The generated items must not contain duplicates, but
    ///   they need not be ordered.
    pub fn from_iter<I>(shape : [usize; N], it : I) -> Result<NDArray<N>,String> where I : Iterator<Item = ([usize;N],f64)>{
        let mut strides = [0usize;N];
        _ = strides.iter_mut().zip(shape.iter()).rev().fold(1usize, |c,(s,d)| { *s = c; c*d });

        let mut sp = Vec::new();
        let mut data = Vec::new();
        let totalsize = shape.iter().product();
        for (i,v) in it.take(totalsize) {
            if i.iter().zip(shape.iter()).any(|(j,d)| j >= d) {
                return Err("Index out of bounds".to_string());
            }
            sp.push( i.iter().zip(strides.iter()).map(|(a,b)| a*b).sum());
            data.push(v);
        }

        NDArray::from_flat_tuples_internal(shape, sp.as_slice(), data.as_slice())
    }

    /// Create a new dense [NDArray] from an iterator. 
    ///
    /// # Arguments
    /// - `shape` the shape of the array
    /// - `it` iterator generating the coefficients. It must provide at least values enough to fill
    ///   the shape. The remaining elements are not used, so it need not have finite length.
    pub fn dense_from_iter<I>(shape : [usize; N], it : I) -> Result<NDArray<N>,String> where I : Iterator<Item = f64> {
        let totalsize = shape.iter().product();
        let data : Vec<f64> = it.take(totalsize).collect();
        if data.len() < totalsize {
            Err("Insufficient data".to_string())
        }
        else {
            Self::new(shape,None,data)
        }
    }

    /// Create a new sparse array from indexes and coefficient data.
    pub fn from_tuples(shape : [usize; N], index : &[ [usize; N] ], data : &[f64]) -> Result<NDArray<N>,String>{
        if data.len() != index.len() {
            Err("Mismatching data and index lengths".to_string())
        }
        else if index.len() > 0 && index.iter().any(|i| i.iter().zip(shape.iter()).any(|(&j,&d)| j >= d)) {
            Err("Index out of bounds".to_string())
        }
        else if index.len() == 0 {
            Ok(NDArray{shape, sp : Some(Vec::new()), data : data.to_vec()})
        }
        else {
            let mut strides = [1usize; N]; _ = strides.iter_mut().zip(shape.iter()).rev().fold(1usize, |c,(s,d)| {*s = c; c*d} );

            let sp_unordered : Vec<usize> = index.iter().map(|i| i.iter().zip(strides.iter()).map(|(j,s)| j*s).sum() ).collect();

            NDArray::from_flat_tuples_internal(shape, sp_unordered.as_slice(), data)
        }
    }

    fn from_flat_tuples_internal(shape : [usize; N], sp_unordered : &[usize], data : &[f64]) -> Result<NDArray<N>,String>{
        if sp_unordered.iter().zip(sp_unordered[1..].iter()).any(|(a,b)| a >= b) {
            // sp is unordered
            let mut perm : Vec<usize> = (0..sp_unordered.len()).collect();
            perm.sort_by_key(|i| unsafe { *sp_unordered.get_unchecked(*i) });
            if perm.iter().zip(perm[1..].iter()).any(|(&i0,&i1)| unsafe{ *sp_unordered.get_unchecked(i0) == *sp_unordered.get_unchecked(i1) } ) {
                // eliminate duplicates
                let nunique = perm.len() - perm.iter().zip(perm[1..].iter()).filter(|(&i0,&i1)| unsafe{ *sp_unordered.get_unchecked(i0) == *sp_unordered.get_unchecked(i1) } ).count();
                let mut rsp = vec![0usize; nunique];
                let mut rdata = vec![0.0f64; nunique];

                rsp[0] = sp_unordered[perm[0]];
                rdata[0] = data[perm[0]];

                let mut i = 0usize;
                for (&p0,&p1) in izip!(perm.iter(),perm[1..].iter()) {
                    let i0 = unsafe { *sp_unordered.get_unchecked(p0) };
                    let i1 = unsafe { *sp_unordered.get_unchecked(p1) };
                    if i0 != i1 {
                        i += 1;
                        unsafe{ *rsp.get_unchecked_mut(i) = i1 };
                    }
                    unsafe { *rdata.get_unchecked_mut(i) += *data.get_unchecked(p1) };
                }
                Ok(NDArray{ shape, sp:Some(rsp), data: data.to_vec()})
            }
            else {
                let sp = perm.iter().map(|&i| unsafe{ *sp_unordered.get_unchecked(i)} ).collect();
                let data = perm.iter().map(|&i| unsafe{ *data.get_unchecked(i)} ).collect();

                Ok(NDArray{ shape, sp : Some(sp), data })
            }
        }
        else {
            Ok(NDArray{ shape, sp : Some(sp_unordered.to_vec()), data : data.to_vec() })
        }
    }

    /// Return the shape
    pub fn shape(&self) -> [usize; N] { self.shape }
    /// Reshape the array. The total number of elements in the result must be the same as in this.
    pub fn reshape<const M : usize>(self,shape : [usize; M]) -> Result<NDArray<M>,()> {
        if shape.iter().product::<usize>() != self.shape.iter().product() {
            Err(())
        }
        else {
            Ok(NDArray{ shape,sp : self.sp, data : self.data })
        }
    }
    /// Return number of non-zeros.
    pub fn nnz(&self) -> usize { self.data.len() }
    /// Return the array coefficients as a slice.
    pub fn data(&self) -> &[f64] { self.data.as_slice() }
    /// Return the sparsity pattern, of present.
    pub fn sparsity(&self) -> Option<&[usize]> { if let Some(ref sp) = self.sp { Some(sp.as_slice()) } else { None } }
    /// Multiply all coefficients by a scalar, inplace.
    pub fn inplace_mul_scalar(&mut self, s : f64) { self.data.iter_mut().for_each(|v| *v *= s); }
    /// Return the array items. This consumes the array.
    pub fn dissolve(self) -> ([usize;N],Option<Vec<usize>>,Vec<f64>) { (self.shape,self.sp,self.data) }
    /// Turns a sparse array into a dense array.
    pub fn to_dense(&self) -> NDArray<N> {
        if let Some(ref sp) = self.sp {
            let mut data = vec![0.0; self.shape.iter().product()];
            assert!(sp.iter().max().map(|&v| v < data.len()).unwrap_or(true));
            for (&i,&f) in izip!(sp.iter(),self.data.iter()) {
                unsafe { *data.get_unchecked_mut(i) = f };
            }
            NDArray{
                shape : self.shape,
                sp : None,
                data
            }
        }
        else {
            self.clone()
        }
    }
    /// Return an expression that represents the array.
    pub fn to_expr(&self) -> super::expr::Expr<N> {
        if let Some(ref sp) = self.sp {
            Expr::new(
                &self.shape,
                Some(sp.clone()),
                (0..sp.len()+1).collect(),
                vec![0; sp.len()],
                self.data.clone())
        }
        else {            
            Expr::new(
                &self.shape,
                None,
                (0..self.nnz()+1).collect(),
                vec![0; self.nnz()],
                self.data.clone())
        }
    }


    pub fn add(self, rhs: Self) -> Self {
        assert!(self.shape == rhs.shape);
        let mut lhs = self;
        let mut rhs = rhs;
        NDArray{
            shape : lhs.shape,
            sp : 
                match (&lhs.sp,&rhs.sp) {
                    (Some(ref lsp),Some(ref rsp)) => 
                        Some(itertools::merge_join_by(lsp.iter().zip(rhs.data.iter()), 
                                                 rsp.iter().zip(rhs.data.iter()),
                                                 |a,b| a.0.cmp(b.0))
                            .map(|v| 
                                 match v {
                                     EitherOrBoth::Left((&i,_)) => i,
                                     EitherOrBoth::Right((&i,_)) => i,
                                     EitherOrBoth::Both((&il,_c),(&_ir,_)) => il
                                 })
                            .collect::<Vec<usize>>()),
                        _ => None
                },
            data : 
                match (&lhs.sp,&rhs.sp) {
                    (None,None)           => { lhs.data.iter_mut().zip(rhs.data.iter()).for_each(|(t,&s)| *t += s); lhs.data },
                    (Some(ref lsp),None)      => { lsp.iter().zip(lhs.data().iter()).for_each(|(&i,c)| rhs.data[i] += c); rhs.data },
                    (None,Some(ref rsp))      => { rsp.iter().zip(rhs.data().iter()).for_each(|(&i,c)| lhs.data[i] += c); lhs.data },
                    (Some(ref lsp),Some(ref rsp)) =>
                        itertools::merge_join_by(lsp.iter().zip(rhs.data.iter()), 
                                                 rsp.iter().zip(rhs.data.iter()),
                                                 |a,b| a.0.cmp(b.0))
                            .map(|v| 
                                 match v {
                                     EitherOrBoth::Left((_,&c)) => c,
                                     EitherOrBoth::Right((_,&c)) => c,
                                     EitherOrBoth::Both((_,&cl),(_,&cr)) => cl+cr
                                 })
                            .collect::<Vec<f64>>(),
                }
        }
    }

    pub fn mul_scalar(mut self, v : f64) -> Self {
        self.data.iter_mut().for_each(|c| *c += v);
        self
    }
}


impl<const N : usize> std::ops::Add for NDArray<N> {
    type Output = NDArray<N>;
    fn add(self, rhs: Self) -> Self::Output {
        (self as NDArray<N>).add(rhs)
    }
}

impl<const N : usize> std::ops::Sub for NDArray<N> {
    type Output = NDArray<N>;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut rhs = rhs;
        rhs.inplace_mul_scalar(-1.0);
        self.add(rhs)
    }
}


impl From<&[f64]> for NDArray<1> {
    fn from(v : &[f64]) -> NDArray<1> {
        NDArray{ shape : [ v.len() ], sp : None, data : v.to_vec() }
    }
}

impl From<Vec<f64>> for NDArray<1> {
    fn from(v : Vec<f64>) -> NDArray<1> {
        NDArray{ shape : [ v.len() ], sp : None, data : v }
    }
}

// Implement conversion

impl<const N : usize> Into<Expr<N>> for &NDArray<N> {
    fn into(self) -> Expr<N> {
        Expr::new(
            &self.shape,
            self.sparsity().map(|s| s.to_vec()),
            (0..self.nnz()+1).collect(), // ptr
            vec![0; self.nnz()], // subj
            self.data().to_vec())
    }
}

impl<const N : usize> IntoExpr<N> for NDArray<N> {
    type Result = Expr<N>;
    fn into(self) -> Expr<N> { 
        let nnz = self.nnz();
        let (shape,sp,data) = (self.shape,self.sp,self.data);
        Expr::new(
            &shape,
            sp,
            (0..nnz+1).collect(), // ptr
            vec![0; nnz], // subj
            data)
    }
}
impl<const N : usize> IntoExpr<N> for &NDArray<N> {
    type Result = Expr<N>;
    fn into(self) -> Expr<N> { 
        Expr::new(
            &self.shape,
            self.sparsity().map(|s| s.to_vec()),
            (0..self.nnz()+1).collect(), // ptr
            vec![0; self.nnz()], // subj
            self.data().to_vec())
    }
}


impl<const N : usize> std::ops::Mul<f64> for NDArray<N> {
    type Output = NDArray<N>;
    fn mul(mut self,rhs : f64) -> Self::Output {
        self.data.iter_mut().for_each(|v| *v *= rhs);
        self
    }
}

impl<const N : usize> std::ops::Mul<NDArray<N>> for f64 {
    type Output = NDArray<N>;
    fn mul(self,mut rhs : NDArray<N>) -> Self::Output {
        rhs.data.iter_mut().for_each(|v| *v *= self );
        rhs
    }
}

impl<const N : usize> std::ops::MulAssign<f64> for NDArray<N> {
    fn mul_assign(&mut self, rhs: f64) {
        self.data.iter_mut().for_each(|v| *v *= rhs);
    } 
}


// GLOBAL FUNCTIONS

/// Create a dense [NDArray] from data.
pub fn dense<const N : usize,D>(shape : [usize;N], data : D) -> NDArray<N> where D : Into<Vec<f64>> {
    NDArray::new(shape,None,data.into()).unwrap()
}


pub trait IntoIndexes<const N : usize> {
    fn into_indexes(&self, shape : &[usize;N]) -> Vec<usize>;
}

impl<const N : usize> IntoIndexes<N> for [[usize;N]] {
    fn into_indexes(&self, shape : &[usize;N]) -> Vec<usize> {
        if self.iter().any(|idx| idx.iter().zip(shape.iter()).any(|(&i,&d)| i >= d)) {
            panic!("Index out of bounds");
        }
        let strides = shape.to_strides();
        self.iter().map(|index| strides.to_linear(&index)).collect()
    }
}

impl<const N : usize> IntoIndexes<N> for Vec<[usize;N]> {
    fn into_indexes(&self, shape : &[usize;N]) -> Vec<usize> {
        if self.iter().any(|idx| idx.iter().zip(shape.iter()).any(|(&i,&d)| i >= d)) {
            panic!("Index out of bounds");
        }
        let strides = shape.to_strides();
        self.iter().map(|index| strides.to_linear(&index)).collect()
    }
}

impl IntoIndexes<1> for [usize] {
    fn into_indexes(&self, _shape : &[usize;1]) -> Vec<usize> { self.to_vec() }
}

pub fn zeros<const N : usize>(shape : [usize;N]) -> NDArray<N> {
    NDArray::new(shape,Some(Vec::new()),Vec::new()).unwrap()
}

/// Create a sparse [NDArray] from data.
pub fn sparse<const N : usize,I,D>(shape : [usize;N], sp : I, data : D) -> NDArray<N> where D : Into<Vec<f64>>, I : IntoIndexes<N> {
    let sparsity = sp.into_indexes(&shape);
    NDArray::new(shape,Some(sparsity),data.into()).unwrap()
}
//pub fn sparse<const N : usize,I,D>(shape : [usize;N], sp : I, data : D) -> NDArray<N> where D : Into<Vec<f64>>, I : Into<Vec<usize>> {
//    NDArray::new(shape,Some(sp.into()),data.into()).unwrap()
//}

/// Create a sparse 2-dimensional diagonal matrix.
pub fn diag<V>(data : V) -> NDArray<2> where V:Into<Vec<f64>> {
    let data = data.into();
    let dim = data.len();
    NDArray::new([dim,dim],Some((0..dim*dim).step_by(dim+1).collect()),data).unwrap()
}

/// Create a sparse 2-dimensional array with ones on the diagonal.
pub fn speye(dim : usize) -> NDArray<2> {
    NDArray::new([dim,dim],Some((0..dim*dim).step_by(dim+1).collect()),vec![1.0; dim]).unwrap()
}

/// Create a dense [NDArray] of ones.
pub fn ones<const N : usize>(shape : [usize; N]) -> NDArray<N> {
    NDArray::new(shape,None,vec![1.0; shape.iter().product()]).unwrap()
}

