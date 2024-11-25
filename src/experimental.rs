use expr::ExprEvalError;
use utils::iter::*;
use crate::*;

pub enum EitherExpr<const N : usize,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    Left(A),
    Right(B)
}

pub enum EitherExpr3<const N : usize,E1,E2,E3> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N> 
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3)
}

pub enum EitherExpr4<const N : usize,E1,E2,E3,E4> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N> 
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3),
    Opt4(E4)
}

pub enum EitherExpr5<const N : usize,E1,E2,E3,E4,E5> 
    where 
        E1 : Sized+ExprTrait<N>,
        E2 : Sized+ExprTrait<N>,
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>,
        E5 : Sized+ExprTrait<N>
{
    Opt1(E1),
    Opt2(E2),
    Opt3(E3),
    Opt4(E4),
    Opt5(E5)
}

impl<const N : usize,A,B> ExprTrait<N> for EitherExpr<N,A,B> where A : Sized+ExprTrait<N>, B : Sized+ExprTrait<N> {
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr::Left(e) => e.eval(rs,ws,xs),
            EitherExpr::Right(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3> ExprTrait<N> for EitherExpr3<N,E1,E2,E3> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N> 
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr3::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr3::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr3::Opt3(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3,E4> ExprTrait<N> for EitherExpr4<N,E1,E2,E3,E4> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr4::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt3(e) => e.eval(rs,ws,xs),
            EitherExpr4::Opt4(e) => e.eval(rs,ws,xs)
        }
    }
}
impl<const N : usize,E1,E2,E3,E4,E5> ExprTrait<N> for EitherExpr5<N,E1,E2,E3,E4,E5> 
    where 
        E1 : Sized+ExprTrait<N>, 
        E2 : Sized+ExprTrait<N>, 
        E3 : Sized+ExprTrait<N>,
        E4 : Sized+ExprTrait<N>,
        E5 : Sized+ExprTrait<N>
{
    fn eval(&self, rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) -> Result<(),expr::ExprEvalError> {
        match self {
            EitherExpr5::Opt1(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt2(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt3(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt4(e) => e.eval(rs,ws,xs),
            EitherExpr5::Opt5(e) => e.eval(rs,ws,xs)
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


/// Generative expression. Generate expression as one scalar expression per index. Note that by
/// construction all generated expressions must have the same type, which means that conditional
/// expressions must be expressed either as dynamic expressions or as enumerated expressions. 
///
/// Generating expressions as a set of scalar expressions is generally less efficient than writing
/// them on vectorized form, but there may be situations where it is simpler, for example if the
/// expression uses complicated index arithmetic.
///
/// # Arguments
/// 
/// - `shape` Shape of the expression.
/// - `sp` Optional sparsity pattern. If given, the function will only be called for indexes in the
///   sparsity.
/// - `f` The generative function. This is a function that generates a scalar expression for each
///   index.
///
/// # Example
/// ```
/// use mosekmodel::*;
/// use mosekmodel::experimental::*;
///
/// let mut m = Model::new(None);
/// let x = m.variable(None, &[5,5]);
/// // Generate expression as E + E'
/// _ = m.constraint(None, 
///                  & genexpr([5,5], None, |i| Some(x.clone().index(*i).add(x.clone().index([i[1],i[0]])))),
///                  greater_than(0.0).with_shape(&[5,5]));
/// // Generate expression as E + E', except diagonal elements that are just E 
/// _ = m.constraint(None,
///                  & genexpr([5,5], None, 
///                            |i| Some(if i[0] != i[1] {
///                                       EitherExpr::Left(x.clone().index(*i).add(x.clone().index([i[1],i[0]]))) 
///                                     }
///                                     else {
///                                       EitherExpr::Right(x.clone().index(*i))
///                                     })),
///                  greater_than(0.0).with_shape(&[5,5]));
/// // Generate expression as the lower triangular part if E+E'
/// _ = m.constraint(None,
///                  & genexpr([5,5], None,
///                            |i| if i[0] == i[1] {
///                                  Some(EitherExpr::Right(x.clone().index(*i)))
///                                }
///                                else if i[0] > i[1] {
///                                  Some(EitherExpr::Left(x.clone().index(*i).add(x.clone().index([i[1],i[0]]))))
///                                } 
///                                else {
///                                  None
///                                }),
///                  greater_than(0.0).with_shape(&[5,5]));
/// ```
///
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
        let mut spx = Vec::with_capacity(maxnelm);
        //let (spx,_) = xs.alloc(maxnelm,0);
        let mut nelm : usize = 0;

        if let Some(sp) = &self.sp {
            for i in sp.iter() {
                let mut ii = [0usize;N];
                _ = ii.iter_mut().zip(self.shape.iter()).rev().fold(*i,|i,(k,&d)| { *k = i%d; i/d });
                if let Some(e) = (self.f)(&ii) {
                    spx[nelm] = *i;
                    nelm += 1;

                    e.eval(ws,rs,xs)?;
                }
            }
        }
        else {
            for (i,ii) in self.shape.index_iterator().enumerate() {
                if let Some(e) = (self.f)(&ii) {
                    spx[nelm] = i;
                    nelm += 1;

                    e.eval(ws,rs,xs)?;
                }
            }
        }

        let exprs = ws.pop_exprs(nelm);
        
        let nnz = exprs.iter().map(|(_,_,_,subj,_)| subj.len()).sum::<usize>();
        
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(&self.shape, nnz, nelm);

        rptr[0] = 0;
        rptr[1..].iter_mut().zip(exprs.iter()).fold(0,|p,(rp,(_,_,_,subj,_))| { *rp = p + subj.len(); *rp });
        rsubj.iter_mut().zip(exprs.iter().flat_map(|(_,_,_,subj,_)| subj.iter())).for_each(|(rj,&j)| *rj = j);
        rcof.iter_mut().zip(exprs.iter().flat_map(|(_,_,_,_,cof)| cof.iter())).for_each(|(rc,&c)| *rc = c);

        if let Some(rsp) = rsp {
            rsp.copy_from_slice(spx.as_slice());
        }

        Ok(())
    }
}

//#[cfg(test)]
//mod test {
//    use super::*;
//    use crate::*;
//    #[test]
//    fn test_gen1() {
//        let mut model = Model::new(None);
//        let x = 
//    }
//}
