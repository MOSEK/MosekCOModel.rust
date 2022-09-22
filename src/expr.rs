extern crate itertools;

use itertools::{iproduct,izip};
//use super::utils;
use super::Variable;
//use super::utils;

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

        head[0] = nelm;
        head[1] = nnz;
        head[2] = nd;

        let cof = fpart;

        (ptr,sp,subj,cof)
    }

    pub fn alloc(&mut self, nint : usize, nfloat : usize) -> (& mut [usize], & mut [f64]) {
        self.susize.resize(nint,0);
        self.sf64.resize(nfloat,0.0);
        (self.susize.as_mut_slice(),self.sf64.as_mut_slice())
    }

    /// Returns and validatas a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    pub fn pop_exprs(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
        let mut res = Vec::with_capacity(n);

        let mut sutop = self.utop;
            let mut sftop = self.ftop;

        for _ in 0..n {
            let nd   = self.susize[self.utop-1];
            let nnz  = self.susize[self.utop-2];
            let nelm = self.susize[self.utop-3];
            let shape = &self.susize[self.utop-3-nd..self.utop-3];
            let fullsize : usize = shape.iter().product();

            let utop = sutop-3-nd;
            let (ptr,utop) = (&self.susize[utop-nelm-1..utop],utop - nelm - 1);
            let (sp,utop) =
                if fullsize > nnz {
                    (Some(&self.susize[utop-nelm..utop]),utop-nelm)
                }
                else {
                    (None,utop)
                };

            let subj = &self.susize[utop-nnz..utop];
            let cof  = &self.sf64[sftop-nnz..sftop];

            if let Some(sp) = sp {
                if ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(&a,&b)| a < b) { panic!("Stack does not contain a valid expression: invalid Sparsity"); }
                if let Some(&n) = sp.last() { if n > fullsize { panic!("Stack does not contain a valid expression: invalid Sparsity"); } }
            }

            if ! ptr[..ptr.len()-1].iter().zip(ptr[1..].iter()).all(|(&a,&b)| a < b) {  panic!("Stack does not contain a valid expression: invalid ptr"); }
            if let Some(&p) = ptr.last() { if p > nnz { panic!("Stack does not contain a valid expression: invalid ptr"); } }

            sutop = utop - nnz;
            sftop -= nnz;

            res.push((shape,ptr,sp,subj,cof));
        }

        self.utop = sutop;
        self.ftop = sftop;

        res
    }
    /// Returns and validatas a view of the top-most expression on the stack.
    pub fn pop_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        // |subj[nnz],sp[nelm],ptr[nelm+1],shape[nd],nelm,nnz,nd|
        let nd   : usize = self.susize[self.utop-1];
        let nnz  : usize = self.susize[self.utop-2];
        let nelm : usize = self.susize[self.utop-3];
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

        if let Some(sp) = sp {
            if ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(&a,&b)| a < b) { panic!("Stack does not contain a valid expression: invalid Sparsity"); }
            if let Some(&n) = sp.last() { if n > fullsize { panic!("Stack does not contain a valid expression: invalid Sparsity"); } }
        }


        if ! ptr[..ptr.len()-1].iter().zip(ptr[1..].iter()).all(|(&a,&b)| a <= b) { // println!("ptr = {:?}",ptr);
                                                                                    panic!("Stack does not contain a valid expression: invalid ptr"); }
        if let Some(&p) = ptr.last() { if p > nnz { // println!("p = {}, nnz = {}",p,nnz);
                                                    panic!("Stack does not contain a valid expression: invalid ptr"); } }

        let nnz : usize = if let Some(&p) = ptr.last() { p } else { 0 };
        //println!("shape = {:?}\n\tptr = {:?}\n\tsubj = {:?}\n\tcof = {:?}\n\tsp = {:?}",shape,ptr,&subj[..nnz],&cof[..nnz],sp);

        self.utop = utop - nnz;
        self.ftop -= nnz;

        println!("pop_expr: nd = {}, nnz = {}, nelm = {}, ptr = {:?}, subj = {:?}",nd,nnz,nelm,ptr,subj);
        (shape,ptr,sp,&subj[..nnz],&cof[..nnz])
    }
    /// Returns without validation a mutable view of the top-most
    /// expression on the stack, but does not remove it from the
    /// stack.  Note that this returns the full subj and cof, not just
    /// the part indexes by ptr.
    pub fn peek_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        let nd   = self.susize[self.utop-1];
        let nnz  = self.susize[self.utop-2];
        let nelm = self.susize[self.utop-3];
        let totalsize : usize = self.susize[self.utop-3-nd..self.utop-3].iter().product();

        let totalusize = nd+nelm+1+nnz + (if totalsize < nelm { nelm } else { 0 });
        let totalfsize = nnz;

        let utop = self.utop-3;
        let ftop = self.utop-3;

        let uslice : &[usize] = & self.susize[utop-totalusize..utop];
        let cof    : &[f64]   = & self.sf64[ftop-totalfsize..ftop];
        let (subj,uslice) = uslice.split_at(nnz);
        let (sp,uslice) = if totalsize < nelm { let (a,b) = uslice.split_at(nelm); (Some(a),b) } else { (None,uslice) };
        let (ptr,shape) = uslice.split_at(nelm+1);

        (shape,ptr,sp,subj,cof)
    }
    /// Returns without validation a mutable view of the top-most
    /// expression on the stack, but does not remove it from the stack
    pub fn peek_expr_mut(&mut self) -> (&mut [usize],&mut [usize],Option<&mut [usize]>,&mut [usize],&mut [f64]) {
        let nd   = self.susize[self.utop-1];
        let nnz  = self.susize[self.utop-2];
        let nelm = self.susize[self.utop-3];
        let totalsize : usize = self.susize[self.utop-3-nd..self.utop-3].iter().product();

        let totalusize = nd+nelm+1+nnz + (if totalsize < nelm { nelm } else { 0 });
        let totalfsize = nnz;

        let utop = self.utop-3;
        let ftop = self.utop-3;

        let uslice : &mut[usize] = & mut self.susize[utop-totalusize..utop];
        let cof    : &mut[f64]   = & mut self.sf64[ftop-totalfsize..ftop];
        let (subj,uslice) = uslice.split_at_mut(nnz);
        let (sp,uslice) = if totalsize < nelm { let (a,b) = uslice.split_at_mut(nelm); (Some(a),b) } else { (None,uslice) };
        let (ptr,shape) = uslice.split_at_mut(nelm+1);

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

        //println!("ExprTrait::eval_finalize");
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        let nnz  = subj.len();
        let nelm = shape.iter().product();
        let (rptr,_,rsubj,rcof) = rs.alloc_expr(shape,nnz,nelm);

        let maxj = rsubj.iter().max().unwrap_or(&0);
        let (jj,ff) = xs.alloc(maxj*2+2,maxj+1);
        let (jj,jjind) = jj.split_at_mut(maxj+1);


        let mut ii  = 0;
        let mut nzi = 0;
        rptr[0] = 0;

        if let Some(sp) = sp {
            for (i,(&p0,&p1)) in ptr[0..ptr.len()-1].iter().zip(ptr[1..].iter()).enumerate() {
                rptr[ii+1..i+1].fill(nzi); ii = i;

                let mut rownzi : usize = 0;
                subj[p0..p1].iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
                subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
                    if c == 0.0 {}
                    else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
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
                println!("ExprTrait::eval_finalize: nzi = {}",nzi);
                rptr[i+1] = nzi;
                ii += 1;
            }
            ....
        }
        else {
            for (&p0,&p1,rp1) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),rptr[1..].iter()) {
                //rptr[ii+1..i+1].fill(nzi); ii = i;
                //let mut rownzi : usize = 0;

                izip!(subj[p0..p1].iter().map(|&v| v)
                      cof[p0..p1].iter().map(|&v| v))
                    .partial_fold_map(|(j0,v0),(j,v)| if j0 == j { Some(v0+v) } else { None })
                    .for_each(|(j0,v0)| {
                        .....
                    });


                    

                subj[p0..p1].iter().for_each(|&j| unsafe{ *jjind.get_unchecked_mut(j) = 0; });
                subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&j,&c)| {
                    if c == 0.0 {}
                    else if (unsafe{ *jjind.get_unchecked(j) } == 0 ) {
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
                println!("ExprTrait::eval_finalize: nzi = {}",nzi);
                rptr[i+1] = nzi;
                ii += 1;
            }
        }
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
        if self.shape.iter().product::<usize>() != shape.iter().product::<usize>() {
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

    pub fn mul_left<E : ExprTrait>(m : Matrix, e : E) -> ExprMulLeft<E> { ExprMulLeft{item : e, lhs : m} }

    pub fn dot<E : ExprTrait>(expr : E, data:Vec<f64>) -> ExprDot<E> {
        ExprDot{ data, expr }
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
#[derive(Clone)]
pub struct Matrix {
    dim  : (usize,usize),
    rows : bool,
    data : Vec<f64>
}

impl Matrix {
    pub fn new(height : usize, width : usize, data : Vec<f64>) -> Matrix {
        if height*width != data.len() { panic!("Invalid data size for matrix")  }
        Matrix{
            dim : (height,width),
            rows : true,
            data : data
        }
    }
    pub fn ones(height : usize, width : usize) -> Matrix {
        Matrix{
            dim : (height,width),
            rows : true,
            data : vec![1.0; height*width]
        }
    }
    pub fn diag(data : &[f64]) -> Matrix {
        Matrix{
            dim : (data.len(),data.len()),
            rows : true,
            data : iproduct!((0..data.len()).zip(data.iter()),0..data.len()).map(|((i,&c),j)| if i == j {c} else { 0.0 }).collect()
        }
    }
}


pub struct ExprMulLeft<E:ExprTrait> {
    item : E,
    lhs  : Matrix
}

// struct ExprMulRight<E:ExprTrait> {
//     item : E,
//     rhs  : Matrix
// }

pub struct ExprMulScalar<E:ExprTrait> {
    item : E,
    lhs  : f64
}

impl<E:ExprTrait> ExprTrait for ExprMulLeft<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(ws,rs,xs);
        // println!("ExprMulLeft eval");
        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        
        let nd   = shape.len();
        let nnz  = subj.len();
        let nelm = ptr.len()-1;
        let (mdimi,mdimj) = self.lhs.dim;


        if nd != 2 && nd != 1{ panic!("Invalid shape for multiplication") }
        if mdimj != shape[0] { panic!("Mismatching shapes for multiplication") }

        let rdimi = mdimi;
        let rdimj = if nd == 1 { 1 } else { shape[1] };
        let edimi = shape[0];
        let rnnz = nnz * mdimi;
        let rnelm = mdimi * rdimj;

        let (perm_spptr,mrowdata) = xs.alloc(nelm+rdimj+1,self.lhs.data.len());
        if self.lhs.rows { mrowdata.clone_from_slice(self.lhs.data.as_slice()); }
        else {
            iproduct!((0..mdimi),(0..mdimj)).zip(mrowdata.iter_mut())
                .for_each(|((i,j),dst)| *dst = unsafe { * self.lhs.data.get_unchecked(j*mdimi+i) });
        }

        let (rptr,_rsp,rsubj,rcof) = if nd == 2 {
            rs.alloc_expr(&[rdimi,rdimj],rnnz,rnelm)
        }
        else {
            rs.alloc_expr(&[rdimi],rnnz,rnelm)
        };

        rptr[0] = 0;
        let mut elmi = 0;
        let mut nzi  = 0;

        rptr[0] = 0;

        if let Some(sp) = sp {
            let (perm,spptr) = perm_spptr.split_at_mut(sp.len());
            spptr.iter_mut().for_each(|v| *v = 0);
            perm.iter_mut().enumerate().for_each(|(i,pi)| *pi = i );
            perm.sort_by_key(|&k| {
                let spi = unsafe{ sp.get_unchecked(k) };
                let ii = spi / rdimj; let jj = spi - ii*rdimj;
                unsafe { *spptr.get_unchecked_mut(jj+1) += 1 };
                jj * edimi + ii
            });

            { let mut cum = 0; spptr.iter_mut().for_each(|v| { let tv = *v; *v = cum; cum = tv; } ); }

            // loop over matrix rows x expr columns
            iproduct!(mrowdata.chunks(mdimj),
                      spptr[..rdimj].iter().zip(spptr[1..].iter()))
                .for_each(|(mrow,(&sp0,&sp1))| {
                    izip!(sp[sp0..sp1].iter(),ptr[sp0..sp1].iter(),ptr[sp0+1..sp1+1].iter()).for_each(|(&spi,&p0,&p1)| {
                        let spi_i = spi / rdimj;
                        //let spi_j = spi % rdimj;
                        let v = unsafe { *mrow.get_unchecked(spi_i) };
                        izip!(subj[p0..p1].iter(),
                              cof[p0..p1].iter(),
                              rsubj[nzi..nzi+p1-p0].iter_mut(),
                              rcof[nzi..nzi+p1-p0].iter_mut())
                            .for_each(|(&j,&c,rj,rc)| { *rj = j; *rc = v*c; });
                        nzi += p1-p0;
                        elmi += 1;
                        rptr[nelm] = nzi;
                    });
                });
        }
        else {
            mrowdata.chunks(mdimj).for_each(|mrowi| { // for each matrix row
                (0..rdimj).for_each(|j| { // for each expression column
                    izip!(mrowi.iter(), ptr[j..].iter().step_by(rdimj), ptr[j+1..].iter().step_by(rdimj)).for_each(|(&c,&p0,&p1)| {
                        rsubj[nzi..nzi+p1-p0].clone_from_slice(&subj[p0..p1]);
                        rcof[nzi..nzi+p1-p0].iter_mut().zip(cof[p0..p1].iter()).for_each(|(rv,&v)| *rv = c*v );
                        nzi += p1-p0;
                    });
                    elmi += 1;
                    rptr[elmi] = nzi;
                    // println!("rptr[{}] = {}",elmi,nzi);
                });
            });
        }
        // println!("rptr = {:?}",rptr);
    }
}


impl<E:ExprTrait> ExprTrait for ExprMulScalar<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.item.eval(rs,ws,xs);
        let (_shape,_ptr,_sp,_subj,cof) = rs.peek_expr_mut();
        cof.iter_mut().for_each(|c| *c *= self.lhs)
    }
}

pub struct ExprDot<E:ExprTrait> {
    data : Vec<f64>,
    expr : E
}

impl<E:ExprTrait> ExprTrait for ExprDot<E> {
    fn eval(&self,rs : & mut WorkStack, ws : & mut WorkStack, xs : & mut WorkStack) {
        self.expr.eval(ws,rs,xs);

        let (shape,ptr,sp,subj,cof) = ws.pop_expr();
        println!("ExprDot::eval: subj = {:?}, cof = {:?}",subj,cof);
        let nd   = shape.len();
        let nnz  = subj.len();

        if nd != 1 || shape[0] != self.data.len() {
            // println!("nd = {}, shape = {:?}, data = {:?}",nd,shape,self.data);
            panic!("Mismatching operands");
        }

        if let Some(sp) = sp {
            let rnnz = nnz;
            let rnelm = 1;
            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],rnnz,rnelm);

            rsubj.clone_from_slice(subj);
            rptr[0] = 0;
            rptr[1] = rnnz;
            for (&i,&p0,&p1) in izip!(sp.iter(),ptr[0..ptr.len()-1].iter(),ptr[1..].iter()) {
                let v = self.data[i];
                for (&c,rc) in cof[p0..p1].iter().zip(rcof.iter_mut()) {
                    *rc = c*v;
                }
            }
        }
        else {
            let rnnz = nnz;
            let rnelm = 1;
            let (rptr,_rsp,rsubj,rcof) = rs.alloc_expr(&[],rnnz,rnelm);

            rsubj.clone_from_slice(subj);
            rptr[0] = 0;
            rptr[1] = rnnz;
            println!("ExprDot::eval: result nnz = {}, nelm = {}, ptr = {:?}, subj = {:?}",rnnz,rnelm,rptr,rsubj);
            for (&p0,&p1,v) in izip!(ptr[0..ptr.len()-1].iter(),ptr[1..].iter(),self.data.iter()) {
                for (&c,rc) in cof[p0..p1].iter().zip(rcof.iter_mut()) {
                    *rc = c*v;
                }
            }
        }
    }
}
