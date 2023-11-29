use super::{ExprTrait, ExprReshapeOneRow};
use super::workstack::WorkStack;
use super::matrix::Matrix;
use crate::utils::*;
use std::slice::Chunks;
use itertools::izip;

pub struct ExprMulScalar<const N : usize, E:ExprTrait<N>> {
    item : E,
    lhs  : f64
}

pub struct ExprMulLeft<E:ExprTrait<2>> {
    item : E,
    
    shape : [usize;2],
    data  : Vec<f64>,
    sp    : Option<Vec<usize>>
}

pub struct ExprMulRight<E:ExprTrait<2>> {
    item : E,
    shape : [usize;2],
    data  : Vec<f64>,
    sp    : Option<Vec<usize>>
}

pub struct ExprMulElm<const N : usize,E> where E : ExprTrait<N>+Sized {
    pub(super) expr : E,
    pub(super)datashape : [usize; N],
    pub(super)datasparsity : Option<Vec<usize>>,
    pub(super)data : Vec<f64>
}

pub struct ExprDotRows<E> where E : ExprTrait<2> {
    expr : E,
    shape : [usize; 2],
    sparsity : Option<Vec<usize>>,
    data : Vec<f64>
}


///////////////////////////////////////////////////////////////////////////////
// Left multiplication
//
// SOMETHING.mul(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

/// Trait defining something that can be left-multiplied on an
/// expression.
pub trait ExprLeftMultipliable<const N : usize,E> 
    where E:ExprTrait<N>
{
    type Result;
    fn mul(self,other : E) -> Self::Result;
}

impl<E, M> ExprLeftMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulLeft<E>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulLeft{
            item : rhs,
            shape,
            data,
            sp}
    }
}

impl<E, M> ExprLeftMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<ExprReshapeOneRow<1,2,E>>>;
    fn mul(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : ExprReshapeOneRow{ item: rhs, dim : 0 },
                shape,
                data,
                sp},
            dim : 0
        }
    }
}

impl<E> ExprLeftMultipliable<2,E> for Vec<f64>
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<E>>;
    fn mul(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : rhs,
                shape,
                data,
                sp : None},
            dim : 0 }
    }
}

impl<E> ExprLeftMultipliable<2,E> for &[f64]
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulLeft<E>>;
    fn mul(self,rhs : E) -> Self::Result {
        ExprReshapeOneRow{
            item : ExprMulLeft{
                item : rhs,
                shape : [1,self.len()],
                data : self.to_vec(),
                sp : None},
            dim : 0 }
    }
}

impl<const N : usize, E> ExprLeftMultipliable<N,E> for f64
    where E : ExprTrait<N>
{
    type Result = ExprMulScalar<N,E>;
    fn mul(self, rhs : E) -> Self::Result {
        ExprMulScalar{
            item : rhs,
            lhs : self
        }
    }
}



///////////////////////////////////////////////////////////////////////////////
// Right multiplication
//
// E.mul(SOMETHING) where E :ExprTrait
//
// It is used like this: ExprTrait<N> implements
// ``` 
// fn mul(self,rhs:ExprRightMultipliable<N,Self>) -> rhs::Result { rhs.mul_right(self) }
// ```
///////////////////////////////////////////////////////////////////////////////

/// Trait defining something that can be right-multiplied on an
/// expression of dimension N, producing an expression of .
pub trait ExprRightMultipliable<const N : usize,E> 
    where E:ExprTrait<N>
{
    type Result;
    fn mul_right(self,other : E) -> Self::Result;
}

impl<E, M> ExprRightMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulRight<E>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulRight{
            item : rhs,
            shape,
            data,
            sp}
    }
}

impl<E, M> ExprRightMultipliable<1,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<1>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<ExprReshapeOneRow<1,2,E>>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : ExprReshapeOneRow{ item: rhs, dim : 0 },
                shape,
                data,
                sp},
            dim : 0
        }
    }
}

impl<E> ExprRightMultipliable<2,E> for Vec<f64>
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        let shape = [1,self.len()];
        let data = self;
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : rhs,
                shape,
                data,
                sp : None},
            dim : 0 }
    }
}

impl<E> ExprRightMultipliable<2,E> for &[f64]
    where 
        E : ExprTrait<2>
{
    type Result = ExprReshapeOneRow<2,1,ExprMulRight<E>>;
    fn mul_right(self,rhs : E) -> Self::Result {
        ExprReshapeOneRow{
            item : ExprMulRight{
                item : rhs,
                shape : [1,self.len()],
                data : self.to_vec(),
                sp : None},
            dim : 0 }
    }
}

impl<const N : usize, E> ExprRightMultipliable<N,E> for f64
    where E : ExprTrait<N>
{
    type Result = ExprMulScalar<N,E>;
    fn mul_right(self, rhs : E) -> Self::Result {
        ExprMulScalar{
            item : rhs,
            lhs : self
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Left element-wise multiplication
//
// SOMETHING.mul_elem(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

pub trait ExprLeftElmMultipliable<const N: usize, E> 
    where E : ExprTrait<N>
{
    type Result;
    fn mul_elem(self, other : E) -> Self::Result;
}

impl<E,M> ExprLeftElmMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulElm<2,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulElm{
            expr : rhs,
            datashape : shape,
            datasparsity : sp,
            data
        }
    }
}

impl<E> ExprLeftElmMultipliable<1,E> for Vec<f64>
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self
        }
    }
}

impl<E> ExprLeftElmMultipliable<1,E> for &[f64] 
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result { 
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self.to_vec()
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Right element-wise multiplication
//
// SOMETHING.mul_elem(E:ExprTrait)
///////////////////////////////////////////////////////////////////////////////

pub trait ExprRightElmMultipliable<const N: usize, E> 
    where E : ExprTrait<N>
{
    type Result;
    fn mul_elem(self, other : E) -> Self::Result;
}

impl<E,M> ExprRightElmMultipliable<2,E> for M 
    where 
        M : Matrix,
        E : ExprTrait<2>
{
    type Result = ExprMulElm<2,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        let (shape,data,sp) = self.extract();
        ExprMulElm{
            expr : rhs,
            datashape : shape,
            datasparsity : sp,
            data
        }
    }
}

impl<E> ExprRightElmMultipliable<1,E> for Vec<f64>
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result {
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self
        }
    }
}

impl<E> ExprRightElmMultipliable<1,E> for &[f64] 
    where 
        E : ExprTrait<1>
{
    type Result = ExprMulElm<1,E>;

    fn mul_elem(self,rhs : E) -> Self::Result { 
        ExprMulElm{
            expr : rhs,
            datashape : [self.len()],
            datasparsity : None,
            data : self.to_vec()
        }
    }
}



///////////////////////////////////////////////////////////////////////////////
// Trait ExprTrait<N> implementations for
//
// ExprMulLeft
// ExprMulRight
// ExprMulScalar
// ExprMulElm
///////////////////////////////////////////////////////////////////////////////


impl<E> ExprTrait<2> for ExprMulLeft<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        if let Some(ref sp) = self.sp {
            super::eval::mul_left_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs);
        } else {
            super::eval::mul_left_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs);
        }
    }
}

impl<E> ExprTrait<2> for ExprMulRight<E> where E:ExprTrait<2> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        if let Some(ref sp) = self.sp {
            super::eval::mul_right_sparse(self.shape[0],self.shape[1],sp.as_slice(),self.data.as_slice(),rs,ws,xs);
        } else {
            super::eval::mul_right_dense(self.data.as_slice(), self.shape[0],self.shape[1],rs,ws,xs);
        }
    }
}


impl<const N : usize, E:ExprTrait<N>> ExprTrait<N> for ExprMulScalar<N,E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        let (_shape,_ptr,_sp,_subj,cof) = rs.peek_expr_mut();
        cof.iter_mut().for_each(|c| *c *= self.lhs)
    }
}

impl<const N : usize, E : ExprTrait<N>> ExprTrait<N> for ExprMulElm<N,E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        let &nnz = ptr.last().unwrap();
        let nelm = ptr.len()-1;

        if shape.iter().zip(self.datashape.iter()).any(|(&s0,&s1)| s0 != s1) { panic!("Mismatching operand shapes in mul_elm"); }

        if let Some(msp) = &self.datasparsity {
            if let Some(esp) = sp {
                panic!("Unimplemented: Sparse Expr .* Sparse Matrix");
            }
            else {
                // count result size
                let rnelm = msp.len();
                let rnnz = msp.iter().map(|&i| ptr[i+1]-ptr[i]).sum();
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);
                rptr[0] = 0;
                let mut nzi = 0usize;

                if let Some(rsp) = rsp {
                    for (ri,rp,&i,&mc) in izip!(rsp.iter_mut(),
                                                rptr[1..].iter_mut(),
                                                msp.iter(),
                                                self.data.iter()) {
                        let p0 = ptr[i];
                        let p1 = ptr[i+1];

                        *ri = i;
                        *rp = p1-p0; 
                        rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                        rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * mc);
                        nzi += p1-p0;
                    }
                }
            }
        }
        else if let Some(esp) = sp {
            let rnnz = nnz;
            let rnelm = nelm;

            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);

            if let Some(rsp) = rsp {
                rsp.clone_from_slice(esp);
            }
            rsubj.clone_from_slice(subj);
            rptr.clone_from_slice(ptr);
            rcof.clone_from_slice(cof);
            for (&p0,&p1,&i) in izip!(ptr.iter(),ptr[1..].iter(),esp.iter()) {
                let mc = self.data[i];
                rcof[p0..p1].iter_mut().for_each(|c| *c *= mc);
            }
        }
        else {
            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(shape, nnz, nelm);
            rptr.clone_from_slice(ptr);
            rsubj.clone_from_slice(subj);
            rcof.clone_from_slice(cof);
            for (&p0,&p1,&c) in izip!(ptr.iter(),ptr[1..].iter(),self.data.iter()) {
                rcof[p0..p1].iter_mut().for_each(|t| *t *= c );
            }
        }

    }
}

///////////////////////////////////////////////////////////////////////////////
// MulDiag
//
///////////////////////////////////////////////////////////////////////////////

pub trait ExprDiagMultipliable<E> where E : ExprTrait<2> {
    type Result : ExprTrait<1>;
    fn mul_internal(self, other : E) -> Self::Result;
}



//
//impl<E> ExprTrait<1> for ExprDotRows<E> where E : ExprTrait<2> {
//    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
//        self.expr.eval(ws,rs,xs);
//        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
//        let &nnz = ptr.last().unwrap();
//        let nelm = ptr.len()-1;
//
//        let mheight = self.shape[0];
//        let mwidth  = self.shape[1];
//    
//        if shape[0] != self.shape[0] || shape[1] != self.shape[1] {
//            panic!("Mismatching shpes");
//        }
//
//        if let (None,None) = (& self.sparsity,sp) {
//            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[shape[0]], nnz, shape[0]);
//            
//            rsubj.clone_from_slice(subj);
//            rcof.clone_from_slice(cof);
//            rcof.iter_mut().zip(self.data.iter()).for_each(|(c,&v)| *c *= v);
//            rptr.iter_mut().zip(ptr.iter().step_by(self.shape[1])).for_each(|(d,&s)| *d = s);
//        }
//        else {
//            let (tmp,_) = xs.alloc(self.data.len()+nelm,0);
//            let (xmsp,xesp) = tmp.split_at_mut(self.data.len());
//            if let Some(msp) = &self.sparsity { xmsp.clone_from_slice(msp.as_slice()) }
//            else { xmsp.iter_mut().enumerate().for_each(|(i,s)| *s = i); }
//            if let Some(esp) = &self.sparsity { xesp.clone_from_slice(&esp); }
//            else { xesp.iter_mut().enumerate().for_each(|(i,s)| *s = i); }
//        
//            //=========== count nonzeros and element 
//            let mut rnnz = 0;
//            let mut rnelem = 0;
//            {
//                let mut prev_row = usize::MAX;
//                let mut mspit = xmsp.iter().peekable();
//                let mut espit = izip!(xesp.iter(),ptr.iter(),ptr[1..].iter()).peekable();
//                while let (Some(&mi),Some((&ei,&p0,&p1))) = (mspit.peek(),espit.peek()) {
//                    match ei.cmp(mi) {
//                        std::cmp::Ordering::Greater => { _ = mspit.next(); },
//                        std::cmp::Ordering::Less => { _ = espit.next() },
//                        std::cmp::Ordering::Equal => {
//                            rnnz += p1-p0;
//                            if prev_row != mi / mwidth {
//                                prev_row = mi / mwidth;
//                                rnelem += 1;
//                            }
//                        }
//                    }
//                }
//            }
//            //=========== allocate result
//            let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&[mheight], rnnz, rnelem);
//            rptr[0] = 0;
//            let mut nzi = 0;
//            let mut elmi = 0;
//            {
//                let mut prev_row = usize::MAX;
//                let mut mspit = (xmsp.iter().zip(self.data.iter())).peekable();
//                let mut espit = izip!(xesp.iter(),ptr.iter(),ptr[1..].iter()).peekable();
//
//                if let Some(rsp) = rsp {
//                    while let (Some((&mi,&mc)),Some((&ei,&p0,&p1))) = (mspit.peek(),espit.peek()) {
//                        match ei.cmp(&mi) {
//                            std::cmp::Ordering::Greater => {_ = mspit.next(); },
//                            std::cmp::Ordering::Less => { _ = espit.next() },
//                            std::cmp::Ordering::Equal => {
//                                if prev_row != mi / mwidth {
//                                    prev_row = mi / mwidth;
//                                    rsp[elmi] = prev_row; 
//                                    elmi += 1;
//                                }
//                            
//                                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
//                                rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
//                                rcof[nzi..nzi+p1-p0].iter_mut().for_each(|v| *v *= mc);
//
//                                nzi += p1-p0;
//                                rptr[elmi] = nzi;
//                            }
//                        }
//                    }
//                } else {
//                    while let (Some((&mi,&mc)),Some((&ei,&p0,&p1))) = (mspit.peek(),espit.peek()) {
//                        match ei.cmp(&mi) {
//                            std::cmp::Ordering::Greater => { _ = mspit.next(); },
//                            std::cmp::Ordering::Less => { _ = espit.next() }
//                            std::cmp::Ordering::Equal => {
//                                if prev_row != mi / mwidth {
//                                    prev_row = mi / mwidth;
//                                    elmi += 1;
//                                }
//                                
//                                rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
//                                rcof[nzi..nzi+p1-p0].clone_from_slice(&cof[p0..p1]);
//                                rcof[nzi..nzi+p1-p0].iter_mut().for_each(|v| *v *= mc);
//
//                                nzi += p1-p0;
//                                rptr[elmi] = nzi;
//                            }
//                        }
//                    }
//                }
//            }
//        }
//    } 
//}
