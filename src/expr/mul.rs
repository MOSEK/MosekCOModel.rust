use super::{ExprTrait, ExprReshapeOneRow};
use super::workstack::WorkStack;
use super::matrix::Matrix;
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

        match (&self.datasparsity,sp) {
            (Some(msp),Some(esp)) => {
                let (upart,xcof) = xs.alloc(esp.len()*2+1+subj.len(),subj.len());
                let (xsp,upart) = upart.split_at_mut(esp.len());
                let (xptr,xsubj) = upart.split_at_mut(esp.len()+1);

                xptr[0] = 0;
                let mut mit   = msp.iter().zip(self.data.iter()).peekable();
                let mut eit   = izip!(esp.iter(),ptr.iter(),ptr[1..].iter()).peekable();
                let mut rnelm = 0usize;
                let mut rnnz  = 0usize;

                while let (Some((&mi,&mc)),Some((ei,&p0,&p1))) = (mit.peek(),eit.peek()) {
                    match mi.cmp(ei) {
                        std::cmp::Ordering::Less => _ = mit.next(),
                        std::cmp::Ordering::Greater => _ = eit.next(),
                        std::cmp::Ordering::Equal => {
                            xsp[rnelm] = mi;                            
                            xsubj[rnnz..rnnz+p1-p0].clone_from_slice(&subj[p0..p1]);                            
                            xcof[rnnz..rnnz+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(tc,&sc)| *tc = sc * mc);

                            rnelm += 1;
                            rnnz += p1-p0;
                            xptr[rnelm] = rnnz;

                            _ = mit.next();
                            _ = eit.next();
                        }
                    }
                }

                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);
                rptr.clone_from_slice(&xptr[..rnelm+1]);
                assert!(rptr[0] == 0);
                if let Some(rsp) = rsp { rsp.clone_from_slice(&xsp[..rnelm]) };
                rsubj.clone_from_slice(&xsubj[..rnnz]);
                rcof.clone_from_slice(&xcof[..rnnz]);
                xs.clear();
            }
            (Some(msp),None) =>  {
                // count result size
                let rnelm = msp.len();
                let rnnz = msp.iter().map(|&i| ptr[i+1]-ptr[i]).sum();
                //println!("ExprMulElm::eval(): nelm = {}, nnz = {}, rnelm = {}, rnnz = {}",nelm,nnz, rnelm,rnnz);
                //println!("ExprMulElm::eval():\nSource\n\tptr = {:?}\n\tsubj = {:?}",ptr,subj);
                let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(shape, rnnz, rnelm);
                rptr[0] = 0;
                let mut nzi = 0usize;

                let rsp = rsp.unwrap();
                for (ri,rp,&i,&mc) in izip!(rsp.iter_mut(),
                                            rptr[1..].iter_mut(),
                                            msp.iter(),
                                            self.data.iter()) {
                    let p0 = ptr[i];
                    let p1 = ptr[i+1];

                    //println!("  p0 = {}, p1 = {}",p0,p1);
                    *ri = i;
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rc,&c)| *rc = c * mc);
                    nzi += p1-p0;
                    *rp = nzi; 
                }
                //println!("ExprMulElm::eval(): rptr = {:?}",rptr);
            }
            (None,Some(esp)) => {
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
            (None,None) => {
                let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(shape, nnz, nelm);
                rptr.clone_from_slice(ptr);
                rsubj.clone_from_slice(subj);
                rcof.clone_from_slice(cof);
                for (&p0,&p1,&c) in izip!(ptr.iter(),ptr[1..].iter(),self.data.iter()) {
                    rcof[p0..p1].iter_mut().for_each(|t| *t *= c );
                }
            }
        } // match (self.sparsity,sp)
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

#[cfg(test)]
mod test {
    use crate::matrix;

    #[test]
    fn mul() {
        let m = matrix::dense(2, 5, vec![1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0, 1.0,1.0]);
        let e = super::super::Expr{ shape: [2,2],
                                    aptr : vec![0,1,2,3,4],
                                    asubj: vec![5,6,7,8],
                                    acof: vec![1.0,1.0,1.0,1.0],
                                    sparsity : None};
        m.rev_mul(e).eval()
    }
}

