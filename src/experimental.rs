use expr::ExprEvalError;
use utils::iter::*;

use crate::*;

pub enum EitherExpr<const N : usize,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    Left(A),
    Right(B)
}

impl<const N : usize,A,B> ExprTrait<N> for EitherExpr<N,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr::Left(e) => e.eval(rs,ws,xs),
            EitherExpr::Right(e) => e.eval(rs,ws,xs)
        }
    }
}

pub struct GeneratorExpr<const N : usize,F,R> 
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : ExprTrait<0>
{
    shape : [usize; N],
    sp    : Option<Vec<usize>>,
    f : F
}

pub fn genexpr<const N : usize,F,R>(shape : [usize; N], sp : Option<Vec<usize>>, f : F) -> GeneratorExpr<N,F,R>
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : ExprTrait<0>
{
    GeneratorExpr{ shape, sp, f }
}


impl<const N : usize,F,R> ExprTrait<N> for GeneratorExpr<N,F,R>
    where 
        F : Fn(&[usize; N]) -> Option<R>,
        R : ExprTrait<0>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),ExprEvalError> 
    {
        let maxnelm = 
            if let Some(ref sp) = self.sp {
                sp.len()
            } else {
                self.shape.iter().product()
            };
        let mut spx = Vec::new();
        //let (spx,_) = xs.alloc(maxnelm,0);
        let mut nelm : usize = 0;

        if let Some(sp) = &self.sp {
            for i in sp.iter() {
                let mut ii = [0usize;N];
                _ = ii.iter_mut().zip(self.shape.iter()).rev().fold(*i,|i,(k,&d)| { *k = i%d; i/d });
                if let Some(e) = (self.f)(&ii) {
                    spx[nelm] = *i;
                    nelm += 1;

                    e.eval(ws,rs,xs);
                }
            }
        }
        else {
            for (i,ii) in self.shape.index_iterator().enumerate() {
                if let Some(e) = (self.f)(&ii) {
                    spx[nelm] = i;
                    nelm += 1;

                    e.eval(ws,rs,xs);
                }
            }
        }

        let exprs = ws.pop_exprs(nelm);
        
        let nnz = exprs.iter().map(|(_,_,sp,subj,_)| subj.len()).sum::<usize>();
        
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&self.shape, nnz, nelm);

        rptr[0] = 0;
        rptr[1..].iter_mut().zip(exprs.iter()).fold(0,|p,(rp,(_,_,subj,_))| { *rp = p + subj.len(); *rp });
        rsubj.iter_mut().zip(exprs.iter().flat_map(|(_,_,subj,_)| subj.iter());
        rcofiter_mut().zip(exprs.iter().flat_map(|(_,_,subj,_)| subj.iter());

        Ok(())
    }
}


