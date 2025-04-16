//! Module for Variable object and related implementations

pub mod index;
pub use index::*;
use model::BaseModelTrait;

use std::{fmt::Debug, rc::Rc};

use expr::ExprEvalError;
use utils::*;

use crate::{expr::{ExprRightElmMultipliable, ExprDotRows}, utils::iter::PermuteByEx};

use super::*;
use itertools::{iproduct, izip};
use super::utils;


/// A Variable object is basically a wrapper around a variable index
/// list with a shape and a sparsity pattern. It contains no reference to the [Model] object it
/// belongs to, so in a context of multiple Models, it is not possible to verify that it is used
/// with the originating model.
///
/// A [Variable] object does not directly implement [ExprTrait]. Normally expressions are passed by
/// value, but this is impractical for variables, so instead a [Variable] reference can
/// be converted into [ExprTrait] using [Variable::to_expr]:
///
/// ```
/// use mosekcomodel::*;
///
/// let mut model = Model::new(None);
/// let x = model.variable(None,unbounded());
/// // Explicitly convert x to expression
/// model.constraint(None, x.to_expr() , equal_to(1.0));
/// // Use x reference, indirectly causes it to be converted
/// model.constraint(None, &x , equal_to(1.0));
/// // Using a if it was an expression - this does not work for all operations
/// model.constraint(None, x.add(&x), equal_to(1.0));
///
/// ```
#[derive(Clone)]
pub struct Variable<const N : usize> {
    pub(crate) idxs     : Rc<Vec<usize>>,
    pub(crate) sparsity : Option<Rc<Vec<usize>>>,
    pub(crate) shape    : [usize; N]
}

impl<const N : usize> Debug for Variable<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Variable{idxs:")?;
        self.idxs.fmt(f)?;
        f.write_str(", shape:")?;
        self.shape.fmt(f)?;

        if self.sparsity.is_some() {
            f.write_str(",SPARSE")?;
        }
        f.write_str("}")?;
        Ok(())
    }
}




// TODO: Make it more consistent when we consume self and when we take take a reference


impl Variable<1> {
//    fn square_diag(self) -> ExprSquareDiag<Self> where Self:Sized+ExprTrait<1> { ExprSquareDiag{ item : self }}
}
impl Variable<2> {
    pub fn dot_rows<M>(&self, other : M) -> ExprDotRows<ExprVariable<2>>
        where
            M : Matrix 
    {
        ExprTrait::<2>::dot_rows(IntoExpr::<2>::into(self), other)
    }

    pub fn diag(&self) -> Variable<1>  {
        if self.shape[0] != self.shape[1] {
            panic!("Invalid shape for operation")
        }
        if let Some(ref sp) = self.sparsity {
            let n = self.shape[0];
            let idxs : Vec<usize> = sp.iter().zip(self.idxs.iter()).filter(|(&i,_)| i/n == i%n).map(|v| *v.1).collect();
            let rsp = if idxs.len() < n {
                Some(Rc::new(sp.iter().filter(|&i| i/n == i%n).map(|i| i / n).collect()))
            } 
            else {
                None
            };

             Variable {
                idxs : Rc::new(idxs),
                sparsity : rsp,
                shape : [n]
            }
        }
        else {
            Variable {
                idxs : Rc::new(self.idxs.iter().step_by(self.shape[0]+1).cloned().collect()),
                sparsity : None,
                shape : [self.shape[0]]
            }
        }
    }
    
    pub fn transpose(&self) -> Self {
        let mut shape = [0usize; 2];
        shape[0] = self.shape[1];
        shape[1] = self.shape[0];
        if let Some(ref sp) = self.sparsity {
            let mut xsp : Vec<(usize,usize)> = sp.iter().zip(self.idxs.iter()).map(|(&i,&ni)| (( i % self.shape[1]) * self.shape[0] + i / self.shape[1], ni) ).collect();
            xsp.sort();
            let rsp = xsp.iter().map(|v| v.0).collect();
            let rnidxs = xsp.iter().map(|v| v.1).collect();

            Variable{
                idxs : Rc::new(rnidxs),
                sparsity : Some(Rc::new(rsp)),
                shape
            }
        } else {
            let mut idxs = vec![0usize; self.idxs.len()];

            for (t,&s) in izip!(idxs.iter_mut(),
                                (0..self.shape[1])
                                    .flat_map(|i| self.idxs[i..].iter().step_by(self.shape[1]))) {
                *t = s;
            }

            Variable{
                idxs : Rc::new(idxs),
                sparsity : None,
                shape
            }
        }
    }

    

    /// From a 2-dimensional variable, create a sparse variable of the same shape, but with only
    /// non-zeros in the lower triangular part.
    ///
    /// # Arguments
    /// - `with_diag` Indicates if the diagonal should be included.
    pub fn tril(&self,with_diag:bool) -> Variable<2> { 
        let (_d0,d1) = (self.shape[0],self.shape[1]);
        if with_diag {
            self.filter(|i| i/d1 >= i%d1)
        }
        else {
            self.filter(|i| i/d1 > i%d1)
        }
    }
    /// From a 2-dimensional variable, create a sparse variable of the same shape, but with only
    /// non-zeros in the upper triangular part.
    ///
    /// # Arguments
    /// - `with_diag` Indicates if the diagonal should be included.
    pub fn triu(&self,with_diag:bool) -> Variable<2> {
        let (_d0,d1) = (self.shape[0],self.shape[1]);
        if with_diag {
            self.filter(|i| i/d1 <= i%d1)
        }
        else {
            self.filter(|i| i/d1 < i%d1)
        }
    }

    /// From a 2-dimensional variable, create a 1-dimensional variable with the elements from the
    /// lower triangular par in row-major format. 
    ///
    /// # Arguments
    /// - `with_diag` Indicates if the diagonal should be included.

    pub fn trilvec(&self,with_diag:bool) -> Variable<1> { 
        let (d0,d1) = (self.shape[0],self.shape[1]);
        let v = self.tril(with_diag);
        
        let rshape = 
            if d0 == 0 && d1 == 0 {
                [0]
            }
            else if d0 > d1 { 
                if with_diag { [d1 * (d1+1)/2 + d1 * (d0-d1)] }
                else         { [d1 * (d1-1)/2 + d1 * (d0-d1)] }
            }
            else {
                if with_diag { [d0 * (d0 + 1)/2] }
                else         { [d0 * (d0 - 1)/2] }
            };

        if let Some(ref sp) = v.sparsity {
            if sp.len() < rshape[0] {
                Variable{
                    shape : rshape,
                    idxs : v.idxs,
                    sparsity : None
                }
            } else {
                let sp : Vec<usize> = match (d0 > d1,with_diag) {
                    (false,true)  => sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); i0 * (i0+1) / 2 + i1}).collect(),
                    (false,false) => sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); i0 * (i0-1) / 2 + i1}).collect(),
                    (true,true)   => sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); let (i00,i01) = (i0.min(d1), i0.max(d1)-d1) ; i00 * (i00+1) / 2 + i01 * d1 + i1 }).collect(),
                    (true,false)  => sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); let (i00,i01) = (i0.min(d1), i0.max(d1)-d1) ; i00 * (i00-1) / 2 + i01 * d1 + i1 }).collect(),
                };
                Variable{
                    shape : rshape,
                    idxs : v.idxs,
                    sparsity : Some(Rc::new(sp))
                }
            }
        }
        else {
            Variable{
                shape : rshape,
                idxs : v.idxs,
                sparsity : None
            }
        }
    }
    pub fn triuvec(&self,with_diag:bool) -> Variable<1> { 
        let (d0,d1) = (self.shape[0],self.shape[1]);
        let v = self.triu(with_diag);
        
        let rshape = 
            if d0 == 0 && d1 == 0 {
                [0]
            }
            else if d0 < d1 { 
                if with_diag { [d0 * (d0+1)/2 + d0 * (d1-d0)] }
                else         { [d0 * (d0-1)/2 + d0 * (d1-d0)] }
            }
            else {
                if with_diag { [d1 * (d1 + 1)/2] }
                else         { [d1 * (d1 - 1)/2] }
            };

        if let Some(ref sp) = v.sparsity {
            if sp.len() < rshape[0] {
                Variable{
                    shape : rshape,
                    idxs : v.idxs,
                    sparsity : None
                }
            } else {
                let sp : Vec<usize> = 
                    // with    diag: i0*d1+i1 - i0*(i0-1)/2 = i0(2*d1 - i0 + 1)/2 + i1
                    // without diag: i0*d1+i1 - i0*(i0+1)/2 = i0(2*d1 - i0 - 1)/2 + i1
                    if d1 > d0 { sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); i0*(2 * d1 - i0 + 1)/2 + i1}).collect() } 
                    else       { sp.iter().map(|&i| { let (i0,i1) = (i/d1,i%d1); i0*(2 * d1 - i0 - 1)/2 + i1}).collect() };
                Variable{
                    shape : rshape,
                    idxs : v.idxs,
                    sparsity : Some(Rc::new(sp))
                }
            }
        }
        else {
            Variable{
                shape : rshape,
                idxs : v.idxs,
                sparsity : None
            }
        }
    }
}


impl<const N : usize> Variable<N> {
    pub fn new(idxs : Vec<usize>, sparsity : Option<Vec<usize>>, shape : &[usize; N]) -> Variable<N> {
        Variable{
            idxs : Rc::new(idxs),
            sparsity : sparsity.map(|v| Rc::new(v)),
            shape:*shape }
    }

    pub fn len(&self) -> usize { return self.shape.iter().product(); }
    pub fn to_expr(&self) -> ExprVariable<N> {
        ExprVariable{ item : self.clone() }
    }

    /// Get the raw variable indexes
    pub fn idxs(&self) -> &[usize] { self.idxs.as_slice() }
    /// Get the variable sparsity pattern if defined.
    pub fn sparsity(&self) -> Option<&[usize]> { if let Some(ref sp) = self.sparsity { Some(sp.as_slice()) } else { None }}
    /// Get the variable shape
    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }

    /// Perform axis-permutation. In two dimensions this is a `transpose`.
    pub fn axispermute(&self, perm : &[usize; N]) -> Variable<N> { 
        let mut newshape = [usize::MAX; N]; newshape.iter_mut().zip(self.shape.permute_by(perm)).for_each(|(t,&s)| *t = s);
        let st    = utils::Strides::from_shape(&self.shape);
        let newst = utils::Strides::from_shape(&newshape);

        if let Some(ref idx) = self.sparsity {
            let mut p : Vec<usize> = (0..idx.len()).collect();
            p.sort_by_key(|&i| unsafe{ *idx.get_unchecked(i) });

            let idxs : Vec<usize> = self.idxs.permute_by(p.as_slice()).cloned().collect();
            let sparsity : Vec<usize> = idx.permute_by(p.as_slice()).cloned().collect();

            Variable{
                shape : newshape,
                idxs : Rc::new(idxs),
                sparsity : Some(Rc::new(sparsity))
            }
        }
        else {
            let idxs = (0..self.shape.iter().product())
                .map(|i| {
                    let mut idx = [0usize; N];
                    st.to_index(i).permute_by(perm).zip(idx.iter_mut()).for_each(|(&s,t)| *t = s);
                    newst.to_linear(&idx) })
                .collect();

            Variable{
                shape : newshape,
                idxs : Rc::new(idxs),
                sparsity : None
            }
        }
    }
   
    /// Reverse order of elements in a subset of dimensions
    pub fn flip(&self, dims : &[bool;N]) -> Self {
        let st = self.shape.to_strides();
        if let Some(ref sp) = self.sparsity {
            let idxs = sp.iter().map(|&i| {
                let mut idx = st.to_index(i);
                izip!(idx.iter_mut(),dims,self.shape.iter())
                    .for_each(|(i,&f,&d)| if f { *i = d-*i; });
                st.to_linear(&idx)
                }).collect::<Vec<usize>>();
            let mut perm = (0..idxs.len()).collect::<Vec<usize>>();
            perm.sort_unstable_by_key(|&i| unsafe{ *idxs.get_unchecked(i) });

            let rsp = idxs.as_slice().permute_by(perm.as_slice()).cloned().collect();
            let jj  = self.idxs.as_slice().permute_by(perm.as_slice()).cloned().collect();
            Variable {
                shape : self.shape,
                sparsity : Some(Rc::new(rsp)),
                idxs : Rc::new(jj)
            }
        }
        else {
            let idxs = (0..self.idxs.len())
                .map(|i| {
                    let mut idx = st.to_index(i);
                    izip!(idx.iter_mut(),dims,self.shape.iter())
                        .for_each(|(i,&f,&d)| if f { *i = d-*i; });
                    unsafe { *self.idxs.get_unchecked(st.to_linear(&idx)) }
                })
                .collect();
            Variable{
                idxs : Rc::new(idxs),
                sparsity : None,
                shape : self.shape
            }
        }
    }

    /// Maps to [ExprTrait::sum]. 
    pub fn sum(&self) -> impl ExprTrait<0> { ExprTrait::sum(IntoExpr::into(self)) }
    /// Maps to [ExprTrait::neg]. 
    pub fn neg(&self) -> impl ExprTrait<N> { ExprTrait::<N>::neg(IntoExpr::into(self)) }
    /// Maps to [ExprTrait::sum_on]. 
    pub fn sum_on<const K : usize>(&self, axes : &[usize; K]) -> impl ExprTrait<K> { ExprTrait::<N>::sum_on::<K>(IntoExpr::into(self),axes) }
    /// Maps to [ExprTrait::add]. 
    pub fn add<RHS>(&self, rhs : RHS) -> impl ExprTrait<N> where RHS : IntoExpr<N> { ExprTrait::<N>::add(IntoExpr::into(self),rhs) }
    /// Maps to [ExprTrait::sub]. 
    pub fn sub<RHS>(&self, rhs : RHS) -> impl ExprTrait<N> where RHS : IntoExpr<N> { ExprTrait::<N>::sub(IntoExpr::into(self),rhs) }
    /// Maps to [ExprTrait::mul_elem]. 
    pub fn mul_elem<RHS>(&self, other : RHS) -> RHS::Result where RHS : ExprRightElmMultipliable<N,ExprVariable<N>> { other.mul_elem(IntoExpr::into(self)) }

    /// Method mapping [ExprTrait] behavior.
    ///
    /// See [ExprTrait::dynamic].
    pub fn dynamic<'a>(&self) -> super::expr::ExprDynamic<'a,N> where Self : Sized+'a { ExprVariable{ item : self.clone() }.dynamic() }

    pub fn flatten(&self) -> Variable<1> {
        Variable {
            idxs : self.idxs.clone(),
            sparsity : self.sparsity.clone(),
            shape : [self.shape.iter().product()]
        }
    }


    pub fn gather(&self) -> Variable<1> {
        Variable {
            shape : [self.idxs.len()],
            idxs : self.idxs.clone(),
            sparsity : None,
        }
    }

    /// Create a variable as an index or a slice of this variable. 
    ///
    /// Generally, this does what one expects but there are a few irregularities. 
    ///
    /// # Arguments
    /// - `idx` this is the index(es) or the range(s). By default following are accepted
    ///     - Single element index:
    ///         - `usize` for `Variable<1>`: produces a scalar variable
    ///         - `[usize; N]` `variable<N>`: produces a scalar variable
    ///     - Ranges of elements. Either a single range or an array of ranges resulting in a
    ///       [Variable]`<N>`:
    ///         - One of [std::ops::Range]`<usize>`, [std::ops::RangeFrom]`<usize>`,
    ///           [std::ops::RangeTo]`<usize>`, [std::ops::RangeFull] for [Variable]`<1>`: produces a
    ///           [Variable]`<1>`
    ///         - An array `[T; N]` where `T` is one of the range types above.
    ///     - Tuples of length `N` for `N` betweem 2 and 5, where each element is [usize] or one of the range types
    ///       above. 
    ///
    /// # Example
    /// ```rust
    /// use mosekcomodel::*;
    /// 
    /// let mut m = Model::new(None);
    /// let x1 = m.variable(None, 10);
    /// let x2 = m.variable(None, &[10,10]);
    ///
    /// let y1_1 : Variable<0> = x1.index(5);
    /// let y1_2 : Variable<0> = x1.index([5]);
    /// let y1_3 : Variable<1> = x1.index(1..3);
    /// let y1_4 : Variable<1> = x1.index([1..3]);
    /// let y1_5 : Variable<1> = x1.index(2..);
    /// let y1_6 : Variable<1> = x1.index(..3);
    /// let y1_7 : Variable<1> = x1.index(..); // effectively .clone()
    /// // Index a scalar
    /// let y2_1 : Variable<0> = x2.index([2,2]);
    /// // Index an array of ranges
    /// let y2_2 : Variable<2> = x2.index([1..3,1..3]);
    /// // Indexing with tuples
    /// let y2_3 : Variable<2> = x2.index((2,1..3));
    /// let y2_4 : Variable<2> = x2.index((2,3));
    /// let y2_5 : Variable<2> = x2.index((..2,1..3));
    /// let y2_6 : Variable<2> = x2.index((1..,..3));
    /// ```
    pub fn index<I>(&self, idx : I) -> I::Output where I : ModelItemIndex<Self> {
        idx.index(self)
    }
    pub fn into_column(&self) -> Variable<2> {
        Variable {
            shape : [self.shape.iter().product(),1],
            idxs : self.idxs.clone(),
            sparsity : None,
        }
    }

    pub fn dot<RHS>(&self,rhs: RHS) -> RHS::Result where RHS: RightDottable<N,ExprVariable<N>> { rhs.dot(IntoExpr::<N>::into(self)) }
    pub fn mul<RHS>(&self,other : RHS) -> RHS::Result where RHS : ExprRightMultipliable<N,ExprVariable<N>> { other.mul_right(IntoExpr::into(self)) }
    pub fn rev_mul<LHS>(&self, lhs: LHS) -> LHS::Result where LHS : ExprLeftMultipliable<N,ExprVariable<N>> { lhs.mul(IntoExpr::<N>::into(self)) }
    pub fn reshape<const M : usize>(&self,shape : &[usize; M]) -> Variable<M> {
        if shape.iter().product::<usize>() != self.shape.iter().product::<usize>() {
            panic!("Mismatching shapes: {:?} cannot be reshaped into {:?}",self.shape,shape);
        }
        Variable{
            idxs     : self.idxs.clone(),
            sparsity : self.sparsity.clone(),
            shape    : *shape
        }
    }

    pub fn repeat(&self, dim : usize, num : usize) -> impl ExprTrait<N> { self.to_expr().repeat(dim,num)}
    pub fn stack(dim : usize, xs : &[&Variable<N>]) -> Variable<N> {
        if xs.iter().zip(xs[1..].iter())
            .any(|(v0,v1)| v0.shape.iter().zip(v1.shape.iter()).enumerate().all(|(i,(a,b))| i != dim && *a != *b)) {
                panic!("Operands have mismatching shapes");
            }

        let ddim : usize = xs.iter().map(|v| v.shape[dim]).sum();
        //let n      = xs.len();
        let rnelm  = xs.iter().map(|v| v.idxs.len()).sum();
        let mut rshape = xs[0].shape.clone(); rshape[dim] = ddim;
        let nd = rshape.len();

        if dim == 0 {
            let mut ridxs : Vec<usize> = Vec::with_capacity(rnelm);
            for v in xs {
                ridxs.extend(v.idxs.iter());
            }
            let rsp = if rnelm < rshape.iter().product() {
                let mut ofs : usize = 0;
                let mut rsp : Vec<usize> = Vec::with_capacity(rnelm);
                for v in xs {
                    if let Some(ref sp) = v.sparsity {
                        rsp.extend(sp.iter().map(|&i| i+ofs));
                    }
                    else {
                        rsp.extend(ofs..ofs+v.idxs.len());
                    }
                    ofs += v.shape.iter().product::<usize>();
                }

                Some(rsp)
            }
            else {
                None
            };

            Variable{
                idxs     : Rc::new(ridxs),
                sparsity : rsp.map(|v| Rc::new(v)),
                shape    : rshape }
        }
        else if rshape.iter().product::<usize>() == rnelm {
            let _d0 : usize = rshape[0..dim].iter().product();
            let d1 : usize = rshape[dim];
            let d2 : usize = if dim < nd - 1 { rshape[dim+1..].iter().product() } else { 1 };

            let mut ridxs : Vec<usize> = vec![0; rnelm];

            let stride = d1*d2;
            let mut ofs = 0;

            for v in xs {
                let vd1 = v.shape[dim];
                let chunksize = vd1*d2;
                for (src,dst) in v.idxs.chunks(chunksize).zip(ridxs.chunks_mut(stride)) {
                    dst[ofs..ofs+chunksize].clone_from_slice(src);
                }

                ofs += v.shape[dim];
            }
            Variable{idxs     : Rc::new(ridxs),
                     sparsity : None,
                     shape    : rshape}
        }
        else {
            let d0 : usize = rshape[0..dim].iter().product();
            let d1 : usize = rshape[dim];
            let d2 : usize = if dim < nd - 1 { rshape[dim+1..].iter().product() } else { 1 };

            let mut dofs : usize = 0;
            let mut ridxs = Vec::with_capacity(rnelm);
            let mut rsp   = Vec::with_capacity(rnelm);
            for v in xs {
                let vd1 = v.shape[dim];
                ridxs.extend(v.idxs.iter());
                if let Some(ref sp) = v.sparsity {
                    rsp.extend(sp.iter().map(|&i| { let (i0,i1,i2) = (i/(vd1*d2),(i/d2)%vd1,i%d2); (i0*d1+i1+dofs)*d2+i2 }))
                }
                else {
                    rsp.extend(iproduct!(0..d0,dofs..dofs+vd1,0..d2).map(|(i0,i1,i2)| (i0*d1+i1)*d2+i2))
                }
                dofs += v.shape[dim];
            }

            let mut perm : Vec<usize> = (0..rnelm).collect();
            perm.sort_by_key(|&p| *unsafe{rsp.get_unchecked(p)});
            Variable{
                idxs     : Rc::new(perm.iter().map(|&p| unsafe{*ridxs.get_unchecked(p)}).collect()),
                sparsity : Some(Rc::new(perm.iter().map(|&p| unsafe{*rsp.get_unchecked(p)}).collect())),
                shape    : rshape }
        }
    }
    pub fn vstack(xs : &[&Variable<N>]) -> Variable<N> { Self::stack(0,xs) }
    pub fn hstack(xs : &[&Variable<N>]) -> Variable<N> { Self::stack(1,xs) }


    pub fn with_shape<const M : usize>(self, shape : &[usize; M]) -> Variable<M> {
        match self.sparsity {
            None =>
                if self.idxs.len() != shape.iter().product::<usize>() {
                    panic!("Shape does not match the size");
                },
            Some(ref sp) =>
                if ! sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
                    panic!("Shape does not match the sparsity pattern");
                }
        }

        Variable{
            idxs : self.idxs,
            sparsity : self.sparsity,
            shape:*shape
        }
    }

    pub fn with_shape_and_sparsity<const M : usize>(self,shape : &[usize; M], sp : Vec<usize>) -> Variable<M> {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 1  && ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
            panic!("Sparsity pattern is not sorted or contains duplicates");
        }
        if sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }
        Variable {
            idxs     : self.idxs,
            sparsity : Some(Rc::new(sp)),
            shape    : *shape
        }
    }

    pub fn with_sparsity(self, sp : Vec<usize>) -> Variable<N> {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 1 && ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
            panic!("Sparsity pattern is not sorted or contains duplicates");
        }
        if ! sp.last().map_or_else(|| true, |&v| v < self.shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }
       
        Variable {
            idxs : self.idxs,
            sparsity : Some(Rc::new(sp)),
            shape : self.shape
        }
    }



    pub fn numnonzeros(&self) -> usize {
        if let Some(ref sp) = self.sparsity {
            sp.len()
        }
        else {
            self.len()
        }
    }

    pub fn sparse_primal_into<M:BaseModelTrait>(&self,m : &ModelAPI<M>,solid : SolutionType, res : & mut [f64], idx : & mut [[usize;N]]) -> Result<usize,String> 
    {
        let sz = self.numnonzeros();
        if res.len() < sz || idx.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            let mut strides = [0; N];
            _ = strides.iter_mut().zip(self.shape.iter()).rev().fold(1,|c,(s,&d)| { *s = c; *s * d} );
            if let Some(ref sp) = self.sparsity {
                for (&i,ix) in sp.iter().zip(idx.iter_mut()) {
                    let _ = strides.iter().zip(ix.iter_mut()).fold(i, |i,(&s,ix)| { *ix = i / s; i % s } );
                }
            }
            else {
                for (i,ix) in idx.iter_mut().enumerate() {
                    let _ = strides.iter().zip(ix.iter_mut()).fold(i, |i,(&s,ix)| { *ix = i / s; i % s } );
                }
            }
            Ok(sz)
        }
    }
    
    fn filter<F>(&self, mut f : F) -> Variable<N> where F : FnMut(usize) -> bool {
        if let Some(ref sp) = self.sparsity {
            let mut rsp = vec![0usize; sp.len()];
            let mut idxs = vec![0usize; sp.len()];

            let n = izip!(sp.iter().zip(self.idxs.iter()).filter(|(i,_)| f(**i)),
                          rsp.iter_mut(),
                          idxs.iter_mut())
                .fold(0,|n,((&i,&ix),ri,ridx)| { *ri = i; *ridx = ix; n+1 });
            rsp.resize(n, 0);
            idxs.resize(n,0);
            Variable{ idxs : Rc::new(idxs), sparsity : Some(Rc::new(rsp)), shape : self.shape }
        }
        else {
            let mut rsp = vec![0usize; self.idxs.len()];
            let mut idxs = vec![0usize; self.idxs.len()];

            let n = izip!(self.idxs.iter().enumerate().filter(|(i,_)| f(*i)),
                          rsp.iter_mut(),
                          idxs.iter_mut())
                .fold(0,|n,((i,&ix),ri,ridx)| { *ri = i; *ridx = ix; n+1 });
            rsp.resize(n, 0);
            idxs.resize(n,0);

            if n < self.idxs.len() {
                Variable{ idxs : Rc::new(idxs), sparsity : Some(Rc::new(rsp)), shape : self.shape }
            }
            else {
                Variable{ idxs : Rc::new(idxs), sparsity : None, shape : self.shape }
            }
        }
    }
}



pub struct ExprVariable<const N : usize> {
    item : Variable<N>
}

impl<const N : usize> IntoExpr<N> for Variable<N> {
    type Result = ExprVariable<N>;
    fn into(self) -> Self::Result { ExprVariable{ item : self } }
}

impl<const N : usize> IntoExpr<N> for &Variable<N> {
    type Result = ExprVariable<N>;
    fn into(self) -> Self::Result { self.to_expr() }
}

impl<const N : usize> ExprTrait<N> for ExprVariable<N> {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) -> Result<(),ExprEvalError>{
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&self.item.shape,
                                                  self.item.idxs.len(),
                                                  self.item.idxs.len());
        rptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        rsubj.clone_from_slice(self.item.idxs.as_slice());
        rcof.fill(1.0);
        if let (Some(rsp),Some(sp)) = (rsp,&self.item.sparsity) {
            rsp.clone_from_slice(sp.as_slice())
        }
        Ok(())
    }
}

