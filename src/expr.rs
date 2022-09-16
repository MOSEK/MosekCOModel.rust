extern crate itertools;

//use itertools::{iproduct,izip};
//use super::utils;
use super::Variable;
use itertools::{izip};
use super::utils;

// impl<E : ExprTrait> ExprTrait for ExprPrepare<E> {
//     fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
//         self.expr.eval(ws,rs,xs);
//         let (shape,ptr,sp,subj,cof) = ws.pop_expr();
//         let mut perm : Vec<usize> = Vec::with_capacity(subj.len());
//         let mut rptr  = Vec::with_capacity(ptr.len());
//         let mut rsubj = Vec::with_capacity(subj.len());
//         let mut rcof  = Vec::with_capacity(cof.len());

//         rptr.push(0);
//         ptr[0..ptr.len()-1].iter().join(ptr[1..].iter()).for_each(|(&p0,&p1)| {
//             if p0 + 1 == p1 {
//                 rsubj.push(subj[p0]);
//                 rcof.push(cof[p0]);
//                 rptr.push(subj.len());
//             }
//             else if p0 + 1 < p1 && p1 < subj.len() {
//                 perm.clear();
//                 for i in 0..(p1-p0) { perm.push(i); }
//                 perm.sort_by(|&i| *unsafe { subj.get_unchecked(i) } );

//                 rsubj.push(*unsafe { subj.get_unchecked(perm[p0]) } );
//                 perm[p0..p1-1].iter()
//                     .zip(perm[p0+1..p1].iter())
//                     .for_each(|(j0,j1)| if j0 == j1 { rcof.push(cof) } else { rsubj.push(subj.get_unchecked()); });
//             }
//             else {
//                 panic!("invalid ptr construction");
//             }
//         });
//     }
// }

/// Structure of a computed expression on the workstack:
/// stack top <---> bottom
/// susize: [ ndim, nnz, nelm, shape[ndim], ptr[nnz+1], { nzidx[nnz] if nnz < shape.product() else []}], asubj[nnz]
/// sf64:   [ acof[nnz] ]
pub struct WorkStack {
    susize : Vec<usize>,
    sf64   : Vec<f64>,

    utop : usize,
    ftop : usize
}

impl WorkStack {
    pub fn new(cap : usize) -> WorkStack {
        WorkStack{
            susize : Vec::with_capacity(cap),
            sf64   : Vec::with_capacity(cap),
            utop : 0,
            ftop : 0  }
    }

    /// Allocate a new expression on the stack.
    ///
    /// Arguments:
    /// - shape Shape of the expression
    /// - nsp None if the expression is dense, otherwise the number of nonzeros. This must ne
    ///   strictly smaller than the product of the dimensions.
    /// Returns (ptr,sp,subj,cof)
    ///
    pub fn alloc_expr(& mut self, shape : &[usize], nnz : usize, nelm : usize) -> (& mut [usize], Option<& mut [usize]>,& mut [usize], & mut [f64]) {
        let nd = shape.len();
        let ubase = self.utop;
        let fbase = self.ftop;

        let fullsize = shape.iter().product();
        if fullsize < nelm { panic!("Invalid number of elements"); }

        let unnz  = 3+nd+(nelm+1)+nnz+(if nelm < fullsize { nelm } else { 0 } );

        self.utop += unnz;
        self.ftop += nnz;
        self.susize.resize(self.utop,0);
        self.sf64.resize(self.ftop,0.0);

        let (_,upart) = self.susize.split_at_mut(ubase);
        let (_,fpart) = self.sf64.split_at_mut(fbase);


        let (subj,upart) = upart.split_at_mut(nnz);
        let (sp,upart)   =
            if nelm < fullsize {
                let (sp,upart) = upart.split_at_mut(nelm);
                (Some(sp),upart)
            } else {
                (None,upart)
            };
        let (ptr,upart)    = upart.split_at_mut(nelm+1);
        let (shape_,head)  = upart.split_at_mut(nd);
        shape_.clone_from_slice(shape);

        head[nd]   = nelm;
        head[nd+1] = nnz;
        head[nd+2] = nd;

        let cof = fpart;

        (ptr,sp,subj,cof)
    }

    pub fn alloc(&mut self, nint : usize, nfloat : usize) -> (& mut [usize], & mut [f64]) {
        self.susize.resize(nint,0);
        self.sf64.resize(nfloat,0.0);
        (self.susize.as_mut_slice(),self.sf64.as_mut_slice())
    }

    /// Returns a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    pub fn pop_exprs(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
        let mut res = Vec::with_capacity(n);
        for _ in 0..n {
            let nd   = self.susize[self.utop-1];
            let nnz  = self.susize[self.utop-2];
            let nelm = self.susize[self.utop-3];
            let shape = &self.susize[self.utop-3-nd..self.utop-3];
            let fullsize : usize = shape.iter().product();

            let utop = self.utop-3-nd;
            let (ptr,utop) = (&self.susize[utop-nelm-1..utop],utop - nelm - 1);
            let (sp,utop) =
                if fullsize > nnz {
                    (Some(&self.susize[utop-nelm..utop]),utop-nelm)
                }
                else {
                    (None,utop)
                };

            let subj = &self.susize[utop-nnz..utop];
            let cof  = &self.sf64[self.ftop-nnz..self.ftop];

            self.utop = utop - nnz;
            self.ftop -= nnz;

            res.push((shape,ptr,sp,subj,cof));
        }

        res
    }
    pub fn pop_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        let nd   = self.susize[self.utop-1];
        let nnz  = self.susize[self.utop-2];
        let nelm = self.susize[self.utop-3];
        let shape = &self.susize[self.utop-3-nd..self.utop-3];
        let fullsize : usize = shape.iter().product();

        let utop = self.utop-3-nd;
        let (ptr,utop) = (&self.susize[utop-nelm-1..utop],utop - nelm - 1);
        let (sp,utop) =
            if fullsize > nnz {
                (Some(&self.susize[utop-nelm..utop]),utop-nelm)
            }
        else {
            (None,utop)
        };

        let subj = &self.susize[utop-nnz..utop];
        let cof  = &self.sf64[self.ftop-nnz..self.ftop];

        self.utop = utop - nnz;
        self.ftop -= nnz;

        (shape,ptr,sp,subj,cof)
    }
}

pub trait ExprTrait {
    /// eval_chil`d() will evaluate the expression and put the result on the `rs` stack.
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack);
    /// eval() will evaluate the expression, then cleanit up and put
    /// it on the `rs` stack. I will guarantee that:
    /// - all rows are sorted
    /// - expression contains no zeros or duplicate elements.
    /// - the expression is dense
    fn eval_finalize(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.eval(ws,rs,xs);

        let (shape,ptr,_sp,subj,cof) = ws.pop_expr();
        let nnz  = subj.len();
        let nelm = shape.iter().product();
        let (rptr,_,rsubj,rcof) = rs.alloc_expr(shape,nnz,nelm);

        let maxj = rsubj.iter().max().unwrap_or(&0);
        let (jj,ff) = xs.alloc(maxj*2+2,maxj+1);
        let (jj,jjind) = jj.split_at_mut(maxj+1);


        let mut ii  = 0;
        let mut nzi = 0;
        rptr[0] = 0;
        ptr[0..ptr.len()-1].iter().zip(ptr[1..].iter()).enumerate().for_each(|(i,(&p0,&p1))| {
            rptr[ii..i].iter_mut().for_each(|v| *v = nzi); ii = i;

            let mut rownzi : usize = 0;
            subj[p0..p1].iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
                if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
                    unsafe{
                        *jjind.get_unchecked_mut(j)   = 1;
                        *jj.get_unchecked_mut(rownzi) = j;
                        *ff.get_unchecked_mut(j)      = c;
                    }
                    rownzi += 1;
                }
                else {
                    unsafe{
                        *ff.get_unchecked_mut(j) += c;
                    }
                }
            });

            izip!(jj[0..rownzi].iter(),
                  rsubj[nzi..nzi+rownzi].iter_mut(),
                  rcof[nzi..nzi+rownzi].iter_mut()).for_each(|(&j,rj,rc)| {
                      *rc = unsafe{ *ff.get_unchecked(j) };
                      unsafe{ *jjind.get_unchecked_mut(j) = 0; };
                      *rj = j;
                  });

            nzi += rownzi;
            rptr[i] = nzi;
            ii += 1;
        });
    }

    //fn into_diag(self) -> ExprIntoDiag<Self> { ExprIntoDiag{ item : self } }
    //fn reshape(self, shape : &[usize]) -> ExprReshape<Self>  { ExprReshape{  item : self, shape : shape.to_vec() } }
    //fn mul_scalar(self, c : f64) -> ExprMulScalar<Self> { ExprMulScalar{ item:self, c : c } }
    //fn mul_vec_left(self, v : Vec<f64>) -> ExprMulVec<Self>
    //fn mul_matrix_left(self, matrix : Matrix) -> ExprMulMatrixLeft<Self>
    //fn mul_matrix_right(self, matrix : Matrix) -> ExprMulMatrixRight<Self>
    //fn transpose(self) -> ExprPermuteAxes<Self>
    //fn axispermute(self) -> ExprPermuteAxes<Self>
}



////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// Expression objects

#[derive(Clone)]
pub struct Expr {
    aptr  : Vec<usize>,
    asubj : Vec<usize>,
    acof  : Vec<f64>,
    shape : Vec<usize>,
    sparsity : Option<Vec<usize>>
}

impl Expr {
    pub fn new(aptr  : Vec<usize>,
               asubj : Vec<usize>,
               acof  : Vec<f64>) -> Expr {
        if aptr.len() == 0 { panic!("Invalid aptr"); }
        if ! aptr[0..aptr.len()-1].iter().zip(aptr[1..].iter()).all(|(a,b)| a <= b) {
            panic!("Invalid aptr: Not sorted");
        }
        let & sz = aptr.last().unwrap();
        if sz != asubj.len() || asubj.len() != acof.len() {
            panic!("Mismatching aptr, asubj and acof");
        }

        Expr{
            aptr,
            asubj,
            acof,
            shape : (0..sz).collect(),
            sparsity : None
        }
    }

    pub fn from_variable(variable : &Variable) -> Expr {
        let sz = variable.shape.iter().product();

        match variable.sparsity {
            None =>
                Expr{
                    aptr  : (0..sz+1).collect(),
                    asubj : variable.idxs.clone(),
                    acof  : vec![1.0; sz],
                    shape : variable.shape.clone(),
                    sparsity : None
                },
            Some(ref sp) => {
                Expr{
                    aptr  : (0..sp.len()+1).collect(),
                    asubj : variable.idxs.clone(),
                    acof  : vec![1.0; sp.len()],
                    shape : variable.shape.clone(),
                    sparsity : Some(sp.clone())
                }
            }
        }
    }

    pub fn into_diag(self) -> Expr {
        if self.shape.len() != 1 {
            panic!("Diagonals can only be made from vector expressions");
        }

        let d = self.shape[0];
        Expr{
            aptr : self.aptr,
            asubj : self.asubj,
            acof : self.acof,
            shape : vec![d,d],
            sparsity : Some((0..d*d).step_by(d+1).collect())
        }
    }

    pub fn reshape(self,shape:&[usize]) -> Expr {
        if self.shape.iter().product() != shape.iter().product() {
            panic!("Invalid shape for this expression");
        }

        Expr{
            aptr : self.aptr,
            asubj : self.asubj,
            acof : self.acof,
            shape : shape.to_vec(),
            sparsity : self.sparsity
        }
    }
}

impl ExprTrait for Expr {
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let nnz  = self.asubj.len();
        let nelm = self.aptr.len()-1;

        let (aptr,sp,asubj,acof) = rs.alloc_expr(self.shape.as_slice(),nnz,nelm);

        match (&self.sparsity,sp) {
            (Some(ref ssp),Some(dsp)) => dsp.clone_from_slice(ssp.as_slice()),
            _ => {}
        }

        aptr.clone_from_slice(self.aptr.as_slice());
        asubj.clone_from_slice(self.asubj.as_slice());
        acof.clone_from_slice(self.acof.as_slice());
    }
}

////////////////////////////////////////////////////////////
// Multiply
struct Matrix {
    dim  : (usize,usize),
    rows : bool,
    data : Vec<f64>
}

struct ExprMulLeft<E:ExprTrait> {
    item : E,
    lhs  : Matrix
}

struct ExprMulRight<E:ExprTrait> {
    item : E,
    rhs  : Matrix
}

struct ExprMulScalar<E:ExprTrait> {
    item : E,
    lhs  : f64
}

impl<E:ExprTrait> ExprTrait for ExprMulLeft<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        let nd = shape.ldn();
        let nnz = subj.len();
        let nelm = ptr.len()-1;
        let (mdimi,mdimj) = self.lhs.dim;

        if nd != 2 { panic!("Invalid shape for multiplication") }
        if mdimj != shape[0] { panic!("Mismatching shapes for multiplication") }

        let mrowdata = {
            if self.lhs.rows { self.lhs.data }
            else {
                (0..mdimi).zip(0..mdimj)
                    .map(|(i,j)| unsafe { * self.lhs.data.get_unchecked(j*mdimi+i) })
                    .collect()
            }
        };

        let rdimi = dimi;
        let rdimj = shape[1];
        let edimi = shape[0];
        let rshape = [ rdimi, rdimj ];
        let rnnz = nnz * mdimi;
        let rnelm = mdimi * rdimj;
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(rnnz,rnelm);

        rptr[0] = 0;
        let mut elmi = i;
        let mut nzi  = 0;
        if let Some(sp) = sp {
            let (perm_spptr,_) = xs.alloc(sp.len()+rdimj+1,0);
            let (perm,spptr) = perm_spptr.split_at_mut(sp.len());
            spptr.iter_mut().for_each(|v| *v = 0);
            perm.iter().enumerate().for_each(|(i,pi)| *pi = i );
            perm.sort_by_key(|&k| {
                let spi = unsafe{ sp.get_unchecked(k) };
                let ii = spi / rdimj; let jj = spi - ii*dimj;
                unsafe { *spptr[jj+1] += 1 };
                jj * edimi + ii
            });

            { let mut cum = 0; spptr.iter_mut().for_each(|v| let tv = *v; *v = cum; cum = tv; ); }

            // loop over matrix rows
            iproduct!((0..mdimi).zip(mrowdata.chunks(mdimj)),(0..shape[1])) {
                spptr[..rdimj].iter().zip(spptr[1..].iter()).for_each(|(sp0,sp1)| {
                    izip!(sp[sp0..sp1].iter(),ptr[sp0..sp1].iter(),ptr[sp0+1..sp1+1].iter()).for_each(|(spi,p0,p1)| {
                        izip!(subj[p0..p1].iter(),
                              cof[p0..p1].iter(),
                              rsubj[nzi..nzi+p1-p0].iter_mut(),
                              rcof[nzi..nzi+p1-p0].iter_mut())
                            .for_each(|&j,&c,rj,rc| { &rj = j; &rc = c; });
                        nzi += p1-p0;
                        nelm += 1;
                        rptr[nelm] = nzi;
                    });
                });
            }
        }
        else {
            iproduct!((0..mdimi).zip(mrowdata.chunks(mdimj)),(0..shape[1]))
                .for_each(|((i,mrowi),j)| {
                    izip!(mrowi.iter(), ptr.iter().step_by(rdimj), ptr[1..].iter().step_by(rdimi)).for_each(|(&c,&p0,&p1)| {
                        rsubj[nzi..nzi+p1-p0].iter().zip(subj[p0..p1]).for_each(|(rj,&j) *rj = j );
                        rcof[nzi..nzi+p1-p0].iter().zip(cof[p0..p1]).for_each(|(rv,&v) *rv = c*v );
                        nzi += p1-p0;
                    });
                    nelmi += 1;
                    rptr[elmi] = nzi;
                });
        }
    }
}


impl<E:ExprTrait> ExprTrait for ExprMulScalar<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();

        
        ...
    }
}
