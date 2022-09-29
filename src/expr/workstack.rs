use itertools::{izip};


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

    fn soft_pop(&self, utop : usize, ftop : usize) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64],usize,usize) {
        let nd   = self.susize[utop-1];
        let nnz  = self.susize[utop-2];
        let nelm = self.susize[utop-3];
        let totalsize : usize = self.susize[utop-3-nd..utop-3].iter().product();

        let totalusize = nd+nelm+1+nnz + (if totalsize < nelm { nelm } else { 0 });
        let totalfsize = nnz;

        let utop = utop-3;
        let ftop = utop-3;

        let ubase = utop - totalusize;
        let fbase = ftop - totalfsize;

        let uslice : &[usize] = & self.susize[ubase..utop];
        let cof    : &[f64]   = & self.sf64[fbase..ftop];

        let subj  = &uslice[ubase..ubase+nnz];
        let sp    = if totalsize > nelm { Some(&uslice[ubase+nnz..ubase+nnz+nelm]) } else { None };
        let ptr   = &uslice[totalusize-nelm-1..totalusize-nd];
        let shape = &uslice[totalusize-nelm-1..totalusize-nd];

        (shape,ptr,sp,subj,cof,ubase,fbase)
    }

    fn soft_pop_validate(&self, utop : usize, ftop : usize) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64],usize,usize) {
        let (shape,ptr,sp,subj,cof,ubase,fbase) = self.soft_pop(utop,ftop);
        let nnz = subj.len();
        let fullsize : usize = shape.iter().product();

        if let Some(ref sp) = sp {
            if izip!(sp[0..sp.len()-1].iter(),
                     sp[1..].iter()).any(|(&a,&b)| a >= b) { panic!("Stack does not contain a valid expression: invalid Sparsity"); }
            if let Some(&n) = sp.last() { if n > fullsize { panic!("Stack does not contain a valid expression: invalid Sparsity"); } }
        }

        if izip!(ptr[..ptr.len()-1].iter(),
                 ptr[1..].iter()).any(|(&a,&b)| a > b) {  panic!("Stack does not contain a valid expression: invalid ptr"); }
        if let Some(&p) = ptr.last() { if p > nnz { panic!("Stack does not contain a valid expression: invalid ptr"); } }

        (shape,ptr,sp,subj,cof,ubase,fbase)
    }

    /// Returns and validatas a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    pub fn pop_exprs(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
        println!("-------------WorkStack::pop_exprs({})",n);
        let mut res = Vec::with_capacity(n);

        let mut selfutop = self.utop;
        let mut selfftop = self.ftop;
        for i in 0..n {
            println!("---ustack @ {} = {:?}",i,&self.susize[..selfutop]);
            let nd   = self.susize[selfutop-1];
            let nnz  = self.susize[selfutop-2];
            let nelm = self.susize[selfutop-3];
            let totalsize : usize = self.susize[selfutop-3-nd..selfutop-3].iter().product();
            println!("nd = {}, nelm = {}, nnz = {}",nd,nelm,nnz);
            println!("shape = {:?}",&self.susize[selfutop-3-nd..selfutop-3]);

            let totalusize = nd+nelm+1+nnz + (if nelm < totalsize { nelm } else { 0 });
            let totalfsize = nnz;

            let utop = selfutop-3;
            let ftop = selfftop;

            let ubase = utop - totalusize;
            let fbase = ftop - totalfsize;

            let uslice : &[usize] = & self.susize[ubase..utop];
            println!("  expr slice = {:?}",uslice);
            
            let cof    : &[f64]   = & self.sf64[fbase..ftop];

            let subj  = &uslice[..nnz];
            let sp    = if totalsize > nelm { Some(&uslice[nnz..nnz+nelm]) } else { None };
            let ptrbase = nnz+sp.map(|v| v.len()).unwrap_or(0);
            let ptr   = &uslice[ptrbase..ptrbase+nelm+1];
            let shape = &uslice[ptrbase+nelm+1..];

            let rnnz = ptr.last().copied().unwrap();

            selfutop = ubase;
            selfftop = fbase;
            res.push((shape,ptr,sp,&subj[..rnnz],&cof[..rnnz]))
        }

        self.utop = selfutop;
        self.ftop = selfftop;

        res
    }
    /// Returns and validatas a view of the top-most expression on the stack.
    pub fn pop_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        println!("ustack = {:?}",&self.susize[..self.utop]);
        let mut selfutop = self.utop;
        let mut selfftop = self.ftop;

        let nd   = self.susize[selfutop-1];
        let nnz  = self.susize[selfutop-2];
        let nelm = self.susize[selfutop-3];

        println!("nd = {}, nelm = {}, nnz = {}",nd,nelm,nnz);
        let totalsize : usize = self.susize[selfutop-3-nd..selfutop-3].iter().product();
        
        let totalusize = nd+nelm+1+nnz + (if nelm < totalsize { nelm } else { 0 });
        println!("totalusize = {}, ustack.len = {}",totalusize,self.susize.len());

        let utop = selfutop-3;
        let ftop = selfftop;

        let ubase = utop - totalusize;
        let fbase = ftop - nnz;

        let uslice : &[usize] = & self.susize[ubase..utop];
        let cof    : &[f64]   = & self.sf64[fbase..ftop];

        let subj  = &uslice[..nnz];
        let sp    = if totalsize > nelm { Some(&uslice[nnz..nnz+nelm]) } else { None };
        let ptrbase = nnz+sp.map(|v| v.len()).unwrap_or(0);
        let ptr   = &uslice[ptrbase..ptrbase+nelm+1];
        let shape = &uslice[ptrbase+nelm+1..ptrbase+nelm+1+nd];

        let rnnz = ptr.last().copied().unwrap();

        self.utop = ubase;
        self.ftop = fbase;

        (shape,ptr,sp,&subj[..rnnz],&cof[..rnnz])
    }
    /// Returns without validation a mutable view of the top-most
    /// expression on the stack, but does not remove it from the
    /// stack.  Note that this returns the full subj and cof, not just
    /// the part indexes by ptr.
    pub fn peek_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        let (shape,ptr,sp,subj,cof,_nextutop,_nextftop) = self.soft_pop_validate(self.utop,self.ftop);

        (shape,ptr,sp,subj,cof)
    }
    /// Returns without validation a mutable view of the top-most
    /// expression on the stack, but does not remove it from the stack
    pub fn peek_expr_mut(&mut self) -> (&mut [usize],&mut [usize],Option<&mut [usize]>,&mut [usize],&mut [f64]) {
        let nd   = self.susize[self.utop-1];
        let nnz  = self.susize[self.utop-2];
        let nelm = self.susize[self.utop-3];
        let totalsize : usize = self.susize[self.utop-3-nd..self.utop-3].iter().product();

        let ubase = self.utop - if totalsize > nelm { 3+2*nelm+1+nnz+nd } else { 3+nelm+1+nnz+nd };
        let fbase = self.ftop-nelm;

        let utop = self.utop-3;
        let ftop = self.ftop;

        let uslice : &mut[usize] = & mut self.susize[ubase..utop];
        let cof    : &mut[f64]   = & mut self.sf64[fbase..ftop];
        let (subj,uslice) = uslice.split_at_mut(nnz);
        if nelm < totalsize {
            let (sp,uslice) = uslice.split_at_mut(nelm);
            let (ptr,shape) = uslice.split_at_mut(nelm+1);
            (shape,ptr,Some(sp),subj,cof)
        }
        else {
            let (ptr,shape) = uslice.split_at_mut(nelm+1);
            (shape,ptr,None,subj,cof)
        }
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
            subj.iter_mut().enumerate().for_each(|(i,p)| *p = i+100);
            cof.iter_mut().enumerate().for_each(|(i,p)| *p = (i as f64)*1.1);
        }

        {
            let (shape,ptr,_sp,subj,cof) = ws.pop_expr();

            println!("shape = {:?},nelm = {},nnz = {}",shape,ptr.len()-1,subj.len());
            assert!(shape.len() == 2);
            assert!(shape[0] == 2);
            assert!(shape[1] == 3);
            assert!(ptr.iter().enumerate().all(|(i,&p)| i == p));
            assert!(subj.iter().enumerate().all(|(i,&j)| i == j-100));
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
