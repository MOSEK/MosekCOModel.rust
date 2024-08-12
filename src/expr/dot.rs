use super::ExprTrait;
use super::matrix::{Matrix,NDArray};
use super::workstack::WorkStack;
use itertools::izip;



/// Implements support for dot-operator (inner product).
///
/// Implementations *should* adhere to the rules:
/// - The operands must have the same shape, and
/// - The result is a value representing the sum of elements in the element-wise multiplication of
///   the operands.
pub trait Dot<RHS> {
    type Result;
    fn dot(self, rhs : RHS) -> Self::Result;
}

pub struct ExprDot<const N : usize,E> where E : ExprTrait<N> {
    expr  : E,
    shape : [usize; N],
    cof   : Vec<f64>,
    sp    : Option<Vec<usize>>
}

/// Implements for `NDArray<N<>.dot(ExprTrait<N>)`
impl<const N : usize, E> Dot<E> for NDArray<N>
    where 
        E : ExprTrait<N>
{
    type Result = ExprDot<N,E>;
    fn dot(self, rhs : E) -> Self::Result {
        let (shape,sp,data) = self.dissolve();

        ExprDot{
            expr : rhs,
            shape,
            cof: data,
            sp
        }
    }
}

/// Implements for `NDArray<N<>.dot(ExprTrait<N>)`
impl<const N : usize, E> Dot<NDArray<N>> for E
    where 
        E : ExprTrait<N>
{
    type Result = ExprDot<N,E>;
    fn dot(self, rhs : NDArray<N>) -> Self::Result { rhs.dot(self) }
}

// Implements ExprTrait<1>.dot(&[f64])
impl<E> Dot<&[f64]> for E where E : ExprTrait<1> {
    type Result = ExprDot<1,E>;
    fn dot(self,rhs : &[f64]) -> Self::Result {
        ExprDot{
            expr : self,
            shape : [rhs.len()],
            cof: rhs.to_vec(),
            sp: None
        }
    }
}

// Support &[f64] . dot(ExprTrait<1>)
impl<E> Dot<E> for &[f64] where E : ExprTrait<1> {
    type Result = ExprDot<1,E>;
    fn dot(self,rhs : E) -> Self::Result { rhs.dot(self) }
}


// Implements ExprTrait<1>.dot(Vec<f64>)
impl<E> Dot<Vec<f64>> for E where E : ExprTrait<1> {
    type Result = ExprDot<1,E>;
    fn dot(self,rhs : Vec<f64>) -> Self::Result {
        ExprDot{
            expr : self,
            shape : [rhs.len()],
            cof: rhs,
            sp: None
        }
    }
}
// Support Vec<f64> . dot(ExprTrait<1>)
impl<E> Dot<E> for Vec<f64> where E : ExprTrait<1> {
    type Result = ExprDot<1,E>;
    fn dot(self,rhs : E) -> Self::Result { rhs.dot(self) }
}





impl<const N : usize, E> ExprTrait<0> for ExprDot<N,E> where E : ExprTrait<N> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        if shape.iter().zip(self.shape.iter()).any(|(&a,&b)| a != b) {
            panic!("Mismatching operand shapes for dot");
        }
        let &nnz = ptr.last().unwrap();

        let rnnz : usize =
            if let Some(ref msp) = self.sp {  
                if let Some(esp) = sp {
                    let mut i0 = msp.iter();
                    let mut i1 = esp.iter().zip(ptr.iter().zip(ptr[1..].iter()));

                    let mut v0 = i0.next();
                    let mut v1 = i1.next();
                    let mut r : usize = 0;
                    while let (Some(&mspi),Some((&espi,(&p0,&p1)))) = (v0,v1) {
                        match mspi.cmp(&espi) { 
                            std::cmp::Ordering::Less => { v0 = i0.next(); },
                            std::cmp::Ordering::Greater => { v1 = i1.next(); }
                            std::cmp::Ordering::Equal => {
                                v0 = i0.next();
                                v1 = i1.next();
                                r += p1-p0;
                            }
                        }
                    }
                    r
                } else {
                    msp.iter().map(|&i| unsafe{ *ptr.get_unchecked(i+1) - *ptr.get_unchecked(i)}).sum()
                }
            } else {  
                nnz
            };

        let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],rnnz,1);
        rptr[0] = 0;
        rptr[1] = rnnz;
        //rsubj.clone_from_slice(subj);
        if let Some(ref msp) = self.sp {
            if let Some(esp) = sp {
                let mut nzi = 0usize;
    
                let mut i0 = msp.iter().zip(self.cof.iter());
                let mut i1 = esp.iter().zip(ptr.iter().zip(ptr[1..].iter()));

                let mut v0 = i0.next();
                let mut v1 = i1.next();
                while let (Some((&mspi,&mc)),Some((&espi,(&p0,&p1)))) = (v0,v1) {
                    match mspi.cmp(&espi) { 
                        std::cmp::Ordering::Less => { v0 = i0.next(); },
                        std::cmp::Ordering::Greater => { v1 = i1.next(); }
                        std::cmp::Ordering::Equal => {
                            rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(tc,&sc)| *tc = sc * mc );
                            rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);

                            v0 = i0.next();
                            v1 = i1.next();
                            nzi += p1-p0;
                        }
                    }
                }
            } else {
                let mut nzi = 0usize;
                for (&mspi,&mc) in msp.iter().zip(self.cof.iter()) {
                    let p0 = ptr[mspi];
                    let p1 = ptr[mspi+1];

                    rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(tc,&sc)| *tc = sc * mc );
                    rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                    nzi += p1-p0;
                }
            }
        } 
        else if let Some(esp) = sp {
            rcof.clone_from_slice(cof);
            rsubj.clone_from_slice(subj);
            for (&espi,&p0,&p1) in izip!(esp.iter(),ptr.iter(),ptr[1..].iter()) {
                let v = unsafe{ *self.cof.get_unchecked(espi) };
                rcof[p0..p1].iter_mut().for_each(|c| *c *= v );
            }
        }
        else {
            rsubj.clone_from_slice(subj);
            rcof.clone_from_slice(cof);
            for (&p0,&p1,c) in izip!(ptr.iter(),ptr[1..].iter(),self.cof.iter()) {
                rcof[p0..p1].iter_mut().for_each(|rc| *rc *= c );
            }
        }
    }
}

