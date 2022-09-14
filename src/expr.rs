extern crate itertools;

//use itertools::{iproduct,izip};
use super::utils;
use super::Variable;

#[derive(Clone)]
pub struct Expr {
    aptr  : Vec<usize>,
    asubj : Vec<usize>,
    acof  : Vec<f64>,
    shape : Vec<usize>,
    sparsity : Option<Vec<usize>>
}

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
    fn alloc_expr(& mut self, shape : &[usize], nnz : usize, nelm : usize) -> (& mut [usize], Option<& mut [usize]>,& mut [usize], & mut [f64]) {
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

    fn alloc(&mut self, nint : usize, nfloat : usize) -> (& mut [usize], & mut [f64]) {
        self.susize.resize(nint,0);
        self.sf64.resize(nfloat,0.0);
        (self.susize.as_mut_slice(),self.sf64.as_mut_slice())
    }

    /// Returns a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    fn pop_exprs(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
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
    fn pop_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
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
    /// eval_child() will evaluate the expression and put the result on the `rs` stack.
    fn eval_child(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack);
    /// eval() will evaluate the expression, then cleanit up and put
    /// it on the `rs` stack. I will guarantee that:
    /// - all rows are sorted
    /// - expression contains no zeros or duplicate elements.
    /// - the expression is dense
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.eval_child(ws,rs,xs);

        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        let nnz  = subj.size();
        let nelm = shape.iter().product();
        let (rptr,_,rsubj,rcof) = rs.alloc_expr(shape,nnz,nelm);

        let maxj = rsubj.iter().max().unwrap_or(0);
        let (jj,ff) = xs.alloc(maxj*2+2,maxj+1);
        let (jj,jjind) = jj.split_at_mut(maxj+1);


        let mut ii  = 0;
        let mut nzi = 0;
        rptr[0] = 0;
        ptr[0..ptr.len()-1].iter().zip(ptr[1..].iter()).enumerate().for_each(|(i,(&p0,&p1))| {
            rptr[ii..i].iter().for_each(|v| *v = nzi); ii = i;

            let mut rownzi : usize = 0;
            p.apply_slice(p0,p1-p0,subj).for_each(|&j| unsafe{ *jjind.get_unchecked(j) = 0; });
            p.apply_slice(p0,p1-p0,subj).zip(p.apply_slice(p0,p1-p0,cof)).for_each(|(&j,&c)| {
                if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
                    unsafe{
                        *jjind.get_unchecked(j)   = 1;
                        *jj.get_unchecked(rownzi) = j;
                        *ff.get_unchecked(j)      = c;
                    }
                    rownzi += 1;
                }
                else {
                    unsafe{
                        *ff.get_unchecked(j) += c;
                    }
                }
            });

            izip!(jj[0..rownzi].iter(),
                  rsubj[nzi..nzi+rownzi].iter_mut(),
                  rcof[nzi..nzi+rownzi].iter_mut()).for_each(|(&j,rj,rc)| {
                      rc = unsafe{ *ff.get_unchecked(j) };
                      unsafe{ *jjind.get_unchecked(j) = 0; };
                      rj = j;
                  });

            nzi += rownzi;
            rptr[i] = nzi;
            ii += 1;
        });
    }
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
}

impl ExprTrait for Expr {
    fn eval_child(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        let nnz  = self.asubj.len();
        let nelm = self.aptr.len()-1;

        let (aptr,sp,asubj,acof) = rs.alloc_expr(self.shape,nnz,nelm);

        match (self.sparsity,sp) {
            (Some(ref ssp),Some(dsp)) => dsp.clone_from_slice(ssp.as_slice()),
            _ => {}
        }

        aptr.clone_from_slice(self.aptr.as_slice());
        asubj.clone_from_slice(self.asubj.as_slice());
        acof.clone_from_slice(self.acof.as_slice());
    }
}
