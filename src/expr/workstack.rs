

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

    pub fn clear(& mut self) {
        self.utop = 0;
        self.ftop = 0;
    }
    pub fn is_empty(& mut self) -> bool {
        self.utop == 0 && self.ftop == 0
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

        let fullsize : usize = shape.iter().product();
        if fullsize < nelm { panic!("Invalid number of elements"); }

        let unnz  = 3+nd+(nelm+1)+nnz+(if nelm < fullsize { nelm } else { 0 } );

        self.utop += unnz;
        self.ftop += nnz;
        self.susize.resize(self.utop,0);
        self.sf64.resize(self.ftop,0.0);

        let (_,upart) = self.susize.split_at_mut(ubase);
        let (_,fpart) = self.sf64.split_at_mut(fbase);

        let (subj,upart) = upart.split_at_mut(nnz);
        let (sp,upart)   = if nelm < fullsize {
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

        // println!("pop_expr: nd = {}, nnz = {}, nelm = {}, ptr = {:?}, subj = {:?}",nd,nnz,nelm,ptr,subj);
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


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn workstack() {
        let mut ws = WorkStack::new(512);

        {
            let (ptr,_sp,subj,cof) = ws.alloc_expr(&[3,3],9,9);
            ptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
            subj.iter_mut().enumerate().for_each(|(i,p)| *p = i);
            cof.iter_mut().enumerate().for_each(|(i,p)| *p = (i as f64)*1.1);
        }

        {
            let (ptr,_sp,subj,cof) = ws.alloc_expr(&[2,3],6,6);
            ptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
            subj.iter_mut().enumerate().for_each(|(i,p)| *p = i);
            cof.iter_mut().enumerate().for_each(|(i,p)| *p = (i as f64)*1.1);
        }

        {
            let (shape,ptr,_sp,subj,cof) = ws.pop_expr();

            assert!(shape.len() == 2);
            assert!(shape[0] == 2);
            assert!(shape[1] == 3);
            assert!(ptr.iter().enumerate().all(|(i,&p)| i == p));
            assert!(subj.iter().enumerate().all(|(i,&j)| i == j));
            assert!(cof.iter().enumerate().all(|(i,&c)| (i as f64)*1.1 == c));
        }

        {
            let (shape,ptr,_sp,subj,cof) = ws.pop_expr();

            assert!(shape.len() == 2);
            assert!(shape[0] == 3);
            assert!(shape[1] == 3);
            assert!(ptr.iter().enumerate().all(|(i,&p)| i == p));
            assert!(subj.iter().enumerate().all(|(i,&j)| i == j));
            assert!(cof.iter().enumerate().all(|(i,&c)| (i as f64)*1.1 == c));
        }
    }
}
