//! Module for Variable object and related implementations

use std::{fmt::Debug, rc::Rc};

use expr::ExprEvalError;
use iter::IndexIteratorExt;
use utils::*;

use crate::expr::{ExprRightElmMultipliable, ExprDotRows};

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
/// model.constraint(None, &x.to_expr() , equal_to(1.0));
/// // Use x reference, indirectly causes it to be converted
/// model.constraint(None, &x , equal_to(1.0));
/// // Using a if it was an expression - this does not work for all operations
/// model.constraint(None, &x.add(&x), equal_to(1.0));
///
/// ```
#[derive(Clone)]
pub struct Variable<const N : usize> {
    idxs     : Rc<Vec<usize>>,
    sparsity : Option<Rc<Vec<usize>>>,
    shape    : [usize; N]
}

impl<const N : usize> Debug for Variable<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Variable{idxs:")?;
        self.idxs.fmt(f);
        f.write_str(", shape:")?;
        self.shape.fmt(f)?;

        if self.sparsity.is_some() {
            f.write_str(",SPARSE")?;
        }
        f.write_str("}")?;
        Ok(())
    }
}


impl<const N : usize> ModelItem<N> for Variable<N> {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn shape(&self) -> [usize; N] { self.shape }
    
    fn sparse_primal(&self,m : &Model,solid : SolutionType) -> Result<(Vec<f64>,Vec<[usize;N]>),String> {
        let mut nnz = vec![0.0; self.numnonzeros()];
        let dflt = [0usize; N];
        let mut idx : Vec<[usize;N]> = vec![dflt;self.numnonzeros()];
        self.sparse_primal_into(m,solid,nnz.as_mut_slice(),idx.as_mut_slice())?;
        Ok((nnz,idx))
    }

    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; });
            }
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; })
            }
            Ok(sz)
        }
    }
}

impl ModelItemIndex<Variable<1>> for usize {
    type Output = Variable<0>;
    fn index(self, v : &Variable<1>) -> Variable<0> {
        if v.shape.len() != 1 { panic!("Cannot index into multi-dimensional variable"); }
        if let Some(ref sp) = v.sparsity {
            if let Ok(i) = sp.binary_search(&self) {
                Variable{
                    idxs: Rc::new(vec![v.idxs[i]]),
                    sparsity: None,
                    shape : []
                }
            }
            else {
                Variable{
                    idxs: Rc::new(vec![]),
                    sparsity: Some(Rc::new(vec![])),
                    shape : []
                }
            }
        }
        else {
            Variable{
                idxs : Rc::new(vec![v.idxs[self]]),
                sparsity : None,
                shape : []
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<Variable<N>> for [usize; N] {
    type Output = Variable<0>;
    fn index(self, v : &Variable<N>) -> Variable<0> {
        let index = v.shape.iter().zip(self.iter()).fold(0,|v,(&d,&i)| v*d+i);
        if let Some(ref sp) = v.sparsity {
            if let Ok(i) = sp.binary_search(&index) {
                Variable{
                    idxs : Rc::new(vec![v.idxs[i]]),
                    sparsity : None,
                    shape : []
                }
            }
            else {
                Variable{
                    idxs : Rc::new(vec![]),
                    sparsity : Some(Rc::new(vec![])),
                    shape : []
                }
            }
        }
        else {
            Variable{
                idxs : Rc::new(vec![v.idxs[index]]),
                sparsity : None,
                shape : []
            }
        }
    }
}

impl ModelItemIndex<Variable<1>> for std::ops::Range<usize> {
    type Output = Variable<1>;
    fn index(self, v : &Variable<1>) -> Variable<1> {
        let n = self.len();
        if let Some(ref sp) = v.sparsity {
            let first = match sp.binary_search(&self.start) {
                Ok(i)  => i,
                Err(i) => i
            };
            let last = match sp.binary_search(&self.start) {
                Ok(i) => i+1,
                Err(i) => i
            };

            Variable{
                idxs     : Rc::new(v.idxs[first..last].to_vec()), 
                sparsity : Some(Rc::new(sp[first..last].iter().map(|&i| i - self.start).collect())),
                shape    : [n]
            }
        }
        else {
            Variable{
                idxs     : Rc::new(v.idxs[self].to_vec()),
                sparsity : None,
                shape    : [n]
            }
        }
    }
}

impl<const N : usize> ModelItemIndex<Variable<N>> for [std::ops::Range<usize>; N] {
    type Output = Variable<N>;
    fn index(self, v : &Variable<N>) -> Variable<N> {
        if !self.iter().zip(v.shape.iter()).any(|(r,&d)| r.start > r.end || r.end <= d ) { panic!("The range is out of bounds in the the shape: {:?} in {:?}",self,v.shape) }

        let mut rshape = [0usize;N]; rshape.iter_mut().zip(self.iter()).for_each(|(rs,ra)| *rs = ra.end-ra.start);
        let rstrides = rshape.to_strides();
        let strides = v.shape.to_strides();


        if let Some(ref sp) = v.sparsity {
            let mut rsp   = Vec::with_capacity(sp.len());
            let mut ridxs = Vec::with_capacity(v.idxs.len());

            sp.iter().zip(v.idxs.iter())
                .for_each(|(&s,&ix)|
                          if izip!(rshape.iter(),strides.iter(),self.iter()).all(|(&sh,&st,ra)| { let i = (s / st) % sh; i <= ra.start && i < ra.end }) {
                              rsp.push(izip!(rshape.iter(),
                                             strides.iter(),
                                             self.iter(),
                                             rstrides.iter()).map(|(&sh,&st,ra,&rst)| ((s / st) % sh - ra.start) * rst).sum());
                              ridxs.push(ix);
                          });
            Variable{
                idxs     : Rc::new(ridxs), 
                sparsity : Some(Rc::new(rsp)),
                shape : rshape }
        }
        else {
            fn addvec<const N : usize>(lhs : &[usize;N], rhs : &[usize;N]) -> [usize;N] {
                let mut r = [0usize;N]; 
                r.iter_mut().zip(lhs.iter().zip(rhs.iter())).for_each(|(d,(&s0,&s1))| *d = s0+s1 );
                r
            }

            let mut offset = [0usize; N]; offset.iter_mut().zip(self.iter()).for_each(|(o,i)| *o = i.start);
            let ridxs : Vec<usize> = 
                rshape.index_iterator()
                    .map(|index| strides.to_linear(&addvec(&index,&offset)))
                    .map(|i| v.idxs[i] /*TODO: unsafe get*/)
                    .collect();

            Variable{
                idxs     : Rc::new(ridxs),
                sparsity : None,
                shape    : rshape }
        }
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
    
    pub fn transpose(self) -> Self {
        let mut shape = [0usize; 2];
        shape[0] = self.shape[1];
        shape[1] = self.shape[0];
        if let Some(sp) = self.sparsity {
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

//    fn tril(self,with_diag:bool) -> Self { ExprTriangularPart{item:self,upper:false,with_diag} }
//    fn triu(self,with_diag:bool) -> Self { ExprTriangularPart{item:self,upper:true,with_diag} }
//    fn trilvec(self,with_diag:bool) -> ExprGatherToVec<2,ExprTriangularPart<Self>> where Self:Sized+ExprTrait<2> { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:false,with_diag} } } 
//    fn triuvec(self,with_diag:bool) -> ExprGatherToVec<2,ExprTriangularPart<Self>> where Self:Sized+ExprTrait<2> { ExprGatherToVec{ item:ExprTriangularPart{item:self,upper:true,with_diag} } }
//    fn diag(self) -> ExprDiag<Self> where Self:Sized+ExprTrait<2> { ExprDiag{ item : self, anti : false, index : 0 } }
}


impl<const N : usize> Variable<N> {
    pub fn new(idxs : Vec<usize>, sparsity : Option<Vec<usize>>, shape : &[usize; N]) -> Variable<N> {
        Variable{
            idxs : Rc::new(idxs),
            sparsity : sparsity.map(|v| Rc::new(v)),
            shape:*shape }
    }

    pub fn to_expr(&self) -> ExprVariable<N> {
        ExprVariable{ item : self.clone() }
    }

    /// Get the raw variable indexes
    pub fn idxs(&self) -> &[usize] { self.idxs.as_slice() }
    /// Get the variable sparsity pattern if defined.
    pub fn sparsity(&self) -> Option<&[usize]> { if let Some(ref sp) = self.sparsity { Some(sp.as_slice()) } else { None }}
    /// Get the variable shape
    pub fn shape(&self) -> &[usize] { self.shape.as_slice() }

    pub fn axispermute(self,_perm : &[usize; N]) -> Variable<N> {
        unimplemented!("Not implemented: axispermute");
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
    /// Notice that this can be confused with `ExprTrait::index` since `Variable<N>` implements
    /// `ExprTrait<N>`. To ensure that `Variable::index` is called, it may be necessary to call it
    /// as `(&x).index(...)` or `ExprTrait::index` will take preceedence.
    ///
    /// # Arguments
    /// - `idx` this is the index(es) or the range(s). By default following are accepted
    ///   - `usize` for `Variable<1>`: produces a scalar variable
    ///   - `[usize; N]` `variable<N>`: produces a scalar variable
    ///   - `Range<usize>` for `Variable<1>`: produces a `Variable<1>`
    ///   - `[Range<usize>;N]` for `Variable<N>`: produces a `Variable<N>`
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



    fn numnonzeros(&self) -> usize {
        if let Some(ref sp) = self.sparsity {
            sp.len()
        }
        else {
            self.len()
        }
    }

    fn sparse_primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64], idx : & mut [[usize;N]]) -> Result<usize,String> {
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


//    fn map<const M : usize,F>(self, shape : &[usize;M], f : F) -> ExprMap<N,M,F,Self> 
//        where 
//            F : Clone+FnMut(&[usize;N]) -> Option<[usize;M]>,
//            Self : Sized
//    {
//        ExprMap{ item : self, shape : *shape, f}
//    }

}



pub struct ExprVariable<const N : usize> {
    item : Variable<N>
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


