use itertools::izip;

/// Structure of a computed expression on the workstack:
/// stack bottom <---> top 
/// susize: [ asubj[nnz], sp[0 if nelm < fullsize else nelm], ptr[nelm+1], shape[ndim], shape[ndim], nelm, nnz, nd ]
/// sf64:   [ acof[nnz] ]
/// 
/// NOTE: We allocate some extra space for the shape to allow fast reshaping.



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


    pub fn inline_reshape_expr(& mut self, shape: &[usize]) -> Result<(),String> {
        let newnd = shape.len();
        if shape.len() > 8 { 
            return Err("Shape is too large".to_string());
        } 
        let newtotalsize : usize = shape.iter().product();

        let selfutop = self.utop;

        let nd    = self.susize[selfutop-1];
        let nnz   = self.susize[selfutop-2];
        let nelem = self.susize[selfutop-3];
       
        {
            let shape_field = & mut self.susize[(selfutop-3-nd) .. (selfutop-3)];

            if newtotalsize != shape_field.iter().product() {
                return Err("New shape and original shape do not match".to_string());
            }
        }

        if newnd < nd {
            self.utop -= nd-newnd;
        }
        else {
            self.utop += newnd - nd;
        }

        self.susize[self.utop-newnd-3..self.utop-3].clone_from_slice(shape);
        self.susize[self.utop-1] = newnd;
        self.susize[self.utop-2] = nnz;
        self.susize[self.utop-3] = nelem;

        
        Ok(())
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
        let nd      = shape.len();
        let padding = if nd < 8 { 8 - nd } else { 0 };
        let ubase   = self.utop;
        let fbase   = self.ftop;

        let fullsize : usize = shape.iter().product();
        if fullsize < nelm { panic!("Number of elements too large for shape: {} in {:?}",nelm,shape); }

        let unnz  = 3+nd+padding+(nelm+1)+nnz+(if nelm < fullsize { nelm } else { 0 } );

        self.utop += unnz;
        self.ftop += nnz;
        self.susize.resize(self.utop,0);
        self.sf64.resize(self.ftop,0.0);

        let (_,upart) = self.susize.split_at_mut(ubase);
        let (_,fpart) = self.sf64.split_at_mut(fbase);

        #[cfg(debug_assertions)]
        {
            upart.fill(usize::MAX);
            fpart.fill(f64::MAX);
        }

        let (subj,upart) = upart.split_at_mut(nnz);
        let (sp,upart)   = 
            if nelm < fullsize {
                let (sp,upart) = upart.split_at_mut(nelm);
                (Some(sp),upart)
            } else {
                (None,upart)
            };
        let (ptr,upart)    = upart.split_at_mut(nelm+1);
        let (_,upart)      = upart.split_at_mut(padding);
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

        let ubase = utop - totalusize;
        let fbase = ftop - totalfsize;

        let uslice : &[usize] = & self.susize[ubase..utop];
        let cof    : &[f64]   = & self.sf64[fbase..ftop];

        let subj_base = ubase;
        let sp_base = subj_base+nnz;
        let ptr_base = if nelm < totalsize { sp_base + nelm } else { sp_base };
        let shape_base = ptr_base + nelm+1;

        let subj  = &uslice[subj_base..subj_base+nnz];
        let sp    = if totalsize > nelm { Some(&uslice[sp_base..sp_base+nelm]) } else { None };
        let ptr   = &uslice[ptr_base..ptr_base+nelm+1];
        let shape = &uslice[shape_base..shape_base+nd];

        (shape,ptr,sp,subj,cof,ubase,fbase)
    }


    fn validate(shape : &[usize], ptr : &[usize], sp : Option<&[usize]>, subj : &[usize]) -> Result<(),String> {
        let & nnz = ptr.last().unwrap();
        let fullsize : usize = shape.iter().product();

        if let Some(sp) = sp {
            if izip!(sp.iter(),
                     sp[1..].iter()).any(|(&a,&b)| a >= b) { return Err("Popped invalid expression: Sparsity not sorted or contains duplicates".to_string()); }
            if let Some(&n) = sp.last() { if n > fullsize { return Err("Popped invalid expression: Sparsity entry out of bounds".to_string()); } }
        }

        if izip!(ptr.iter(),ptr[1..].iter()).any(|(&a,&b)| a > b) { return Err("Popped invalid expression: Ptr is not ascending".to_string()); }
        if nnz > subj.len() { return Err("Popped invalid expression: Ptr does not match the number of actual nonzeros".to_string()) }
        Ok(())
    }

    fn soft_pop_validate(&self, utop : usize, ftop : usize) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64],usize,usize) {
        let (shape,ptr,sp,subj,cof,ubase,fbase) = self.soft_pop(utop,ftop);
        let nnz = subj.len();
        let fullsize : usize = shape.iter().product();

        if let Some(sp) = sp {
            if izip!(sp[0..sp.len()-1].iter(),
                     sp[1..].iter()).any(|(&a,&b)| a >= b) { panic!("Stack does not contain a valid expression: invalid Sparsity"); }
            if let Some(&n) = sp.last() { if n > fullsize { panic!("Stack does not contain a valid expression: invalid Sparsity"); } }
        }

        if izip!(ptr[..ptr.len()-1].iter(),
                 ptr[1..].iter()).any(|(&a,&b)| a > b) {  panic!("Stack does not contain a valid expression: invalid ptr"); }
        if ptr.last().copied().unwrap() > nnz { panic!("Stack does not contain a valid expression: invalid ptr"); }

        (shape,ptr,sp,subj,cof,ubase,fbase)
    }

    /// Returns and validatas a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    pub fn pop_exprs(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
        // println!("-------------WorkStack::pop_exprs({})",n);
        let mut res = Vec::with_capacity(n);

        let mut selfutop = self.utop;
        let mut selfftop = self.ftop;
        for _i in 0..n {
            // println!("---ustack @ {} = {:?}",i,&self.susize[..selfutop]);
            let nd   = self.susize[selfutop-1];
            let nnz  = self.susize[selfutop-2];
            let nelm = self.susize[selfutop-3];
            let totalsize : usize = self.susize[selfutop-3-nd..selfutop-3].iter().product();
            // println!("nd = {}, nelm = {}, nnz = {}",nd,nelm,nnz);
            // println!("shape = {:?}",&self.susize[selfutop-3-nd..selfutop-3]);

            let totalusize = nd+nelm+1+nnz + (if nelm < totalsize { nelm } else { 0 });
            let totalfsize = nnz;

            let utop = selfutop-3;
            let ftop = selfftop;

            let ubase = utop - totalusize;
            let fbase = ftop - totalfsize;

            let uslice : &[usize] = & self.susize[ubase..utop];
            // println!("  expr slice = {:?}",uslice);

            let cof    : &[f64]   = & self.sf64[fbase..ftop];

            let subj  = &uslice[..nnz];
            let sp    = if totalsize > nelm { Some(&uslice[nnz..nnz+nelm]) } else { None };
            let ptrbase = nnz+sp.map(|v| v.len()).unwrap_or(0);
            let ptr   = &uslice[ptrbase..ptrbase+nelm+1];
            let shape = &uslice[ptrbase+nelm+1..];

            let rnnz = ptr.last().copied().unwrap();

            selfutop = ubase;
            selfftop = fbase;

            Self::validate(shape,ptr,sp,subj).unwrap();
            res.push((shape,ptr,sp,&subj[..rnnz],&cof[..rnnz]))
        }

        self.utop = selfutop;
        self.ftop = selfftop;

        res
    }
    /// Returns and validatas a view of the top-most expression on the stack.
    pub fn pop_expr(&mut self) -> (&[usize],&[usize],Option<&[usize]>,&[usize],&[f64]) {
        // println!("ustack = {:?}",&self.susize[..self.utop]);
        let selfutop = self.utop;
        let selfftop = self.ftop;

        let nd   = self.susize[selfutop-1];
        let nnz  = self.susize[selfutop-2];
        let nelm = self.susize[selfutop-3];

        // println!("nd = {}, nelm = {}, nnz = {}",nd,nelm,nnz);
        let totalsize : usize = self.susize[selfutop-3-nd..selfutop-3].iter().product();

        let totalusize = nd+nelm+1+nnz + (if nelm < totalsize { nelm } else { 0 });
        // println!("totalusize = {}, ustack.len = {}",totalusize,self.susize.len());

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
        
        // println!("{}:{}: workstack::pop_expr:\n\tshape={:?}\n\tptr={:?}\n\tsubj={:?}",file!(),line!(),shape,ptr,subj);
        
        Self::validate(shape,ptr,sp,subj).unwrap();
        let &rnnz = ptr.last().unwrap();

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
impl std::fmt::Display for WorkStack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f,"WorkStack{{ us : {:?}, fs : {:?} }}",&self.susize[..self.utop],&self.sf64[..self.ftop])
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
