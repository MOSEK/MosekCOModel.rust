extern crate mosek;
extern crate itertools;

use itertools::{iproduct};

pub struct Model {
    task : mosek::Task,

    vars      : Vec<i64>,
    barvarelm : Vec<(usize,usize)>,
    cons      : Vec<(usize,usize)>
}

#[derive(Clone)]
pub struct Variable {
    idxs     : Vec<usize>,
    sparsity : Option<Vec<usize>>,
    shape    : Vec<usize>
}

impl Variable {
    fn new(idxs : Vec<usize>) -> Variable {
        let n = idxs.len();
        Variable {
            idxs : idxs,
            sparsity : None,
            shape : vec![n]
        }
    }
    fn with_shape(self, shape : Vec<usize>) -> Variable {
        match self.sparsity {
            None =>
                if self.idxs.len() != shape.iter().product() {
                    panic!("Shape does not match the size");
                },
            Some(ref sp) =>
                if ! sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
                    panic!("Shape does not match the sparsity pattern");
                }
        }

        Variable{
            idxs     : self.idxs,
            sparsity : self.sparsity,
            shape    : shape
        }
    }
    fn with_sparsity(self, sp : Vec<usize>) -> Variable {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 0 {
            if sp.len() > 1 {
                if ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                    panic!("Sparsity pattern is not sorted or contains duplicates");
                }
            }
        }
        if ! sp.last().map_or_else(|| true, |&v| v < self.shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }

        Variable {
            idxs : self.idxs,
            sparsity : Some(sp),
            shape : self.shape
        }
    }

    fn with_shape_and_sparsity(self,shape : Vec<usize>, sp : Vec<usize>) -> Variable {
        if sp.len() != self.idxs.len() {
            panic!("Sparsity does not match the size");
        }
        if sp.len() > 0 {
            if sp.len() > 1 {
                if ! sp[0..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                    panic!("Sparsity pattern is not sorted or contains duplicates");
                }
            }
        }
        if sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
            panic!("Sparsity pattern does not match the shape");
        }
        Variable {
            idxs : self.idxs,
            sparsity : Some(sp),
            shape : shape.to_vec()
        }
    }

    fn flatten(self) -> Variable {
        Variable {
            idxs : self.idxs,
            sparsity : self.sparsity,
            shape : vec![self.shape.iter().product()]
        }
    }
}

#[derive(Clone)]
pub struct Expr {
    aptr  : Vec<usize>,
    asubj : Vec<usize>,
    acof  : Vec<f64>,
    shape : Vec<usize>,
    sparsity : Option<Vec<usize>>
}

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

    /// Returns a list of views of the `n` top-most expressions on the stack, first in the result
    /// list if the top-most.
    fn pop_expr(&mut self, n : usize) -> Vec<(&[usize],&[usize],Option<&[usize]>,&[usize],&[f64])> {
        let mut res = Vec::with_capacity(n);

        // let utop = self.utop;
        // let ftop = self.ftop;

        // {
        //     let mut utop = self.utop;
        //     let mut ftop = self.ftop;

        //     for _ in 0..n {
        //         let nd = self.susize[utop-1];
        //         let nnz = self.susize[utop-2];
        //         let nelm = self.susize[utop-3];
        //         let fullsize = self.susize[utop-3-nd..utop-3].iter().product();

        //         utop -= 3 + nd + nelm + 1 + nnz;
        //         if nelm < fullsize { utop -= nelm }
        //         ftop -= nnz;
        //     }

        //     self.utop = utop;
        //     self.ftop = ftop;
        // }

        // let mut ustack = & self.susize[..utop];
        // let mut fstack = & self.sf64[..ftop];

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
}

pub trait ExprTrait {
    fn eval(&self,ws : & mut WorkStack, rs : & mut WorkStack, xs : & mut WorkStack);
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
    fn eval(&self,rs : & mut WorkStack, _ws : & mut WorkStack, _xs : & mut WorkStack) {
        match self.sparsity {
            Some(ref sp) => rs.susize.extend_from_slice(sp.as_slice()),
            None => {}
        }
        rs.susize.extend_from_slice(self.aptr.as_slice());
        rs.susize.extend_from_slice(self.shape.as_slice());
        rs.susize.push(self.asubj.len());
        rs.susize.push(self.aptr.len()-1);
        rs.susize.push(self.shape.len());

        rs.susize.extend_from_slice(self.asubj.as_slice());
        rs.sf64.extend_from_slice(self.acof.as_slice());
    }
}
////////////////////////////////////////////////////////////
// Domain definitions
pub enum LinearDomainType {
    NonNegative,
    NonPositive,
    Zero,
    Free
}

pub enum ConicDomainType {
    QuadraticCone,
    RotatedQuadraticCone
}
pub enum ParamConicDomainType {
    PrimalPowerCone,
    DualPowerCone
}

pub trait DomainTrait {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable;
}

pub struct LinearDomain {
    dt    : LinearDomainType,
    ofs   : Vec<f64>,
    shape : Vec<usize>,
    sp    : Option<Vec<usize>>
}

pub struct ConicDomain {
    dt  : ConicDomainType,
    ofs : Vec<f64>,
    shape : Vec<usize>,
    conedim : usize
}

impl DomainTrait for ConicDomain {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        let size = self.shape[self.conedim];
        let num = self.shape.iter().product::<usize>() / size;
        let v = m.conic_variable(name,size,num,self.dt,Some(self.ofs));
        if self.conedim < self.shape.len()-1 {
            // permute the indexes
            let d0 = self.shape[..self.conedim].iter().product();
            let d1 = self.shape[self.conedim];
            let d2 = self.shape[self.conedim+1..].iter().product();
            let idxs : Vec<usize> = iproduct!(0..d0,0..d1,0..d2).map(|(i0,i1,i2)| unsafe { *v.idxs.get_unchecked(i0*d1*d2+i2*d1+i1) }).collect();
            Variable{
                idxs : idxs,
                sparsity : None,
                shape : self.shape
            }
        }
        else {
            v
        }
    }
}

impl DomainTrait for &[i32] {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,LinearDomainType::Free,vec![0.0; self.iter().product::<i32>().try_into().unwrap()]).with_shape(self.iter().map(|&v| v.try_into().unwrap()).collect::<Vec<usize>>())
    }
}

impl DomainTrait for &[usize] {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,LinearDomainType::Free,vec![0.0; self.iter().product()]).with_shape(self.to_vec())
    }
}

impl DomainTrait for Vec<usize> {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        let n = self.iter().product();
        m.linear_variable(name,LinearDomainType::Free,vec![0.0; n]).with_shape(self)
    }
}
impl DomainTrait for usize {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,LinearDomainType::Free,vec![0.0])
    }
}

impl LinearDomain {
    pub fn with_shape(self,shape : Vec<usize>) -> LinearDomain {
        match self.sp {
            Some(ref sp) => if ! sp.last().map_or_else(|| true,|&v| v < shape.iter().product()) {
                panic!("Shaped does not match sparsity");
            },
            None => if self.ofs.len() != shape.iter().product()  {
                panic!("Shaped does not fit expression");
            }
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : shape,
            sp    : self.sp
        }
    }

    pub fn with_sparsity(self,sp : Vec<usize>) -> LinearDomain {
        if sp.len() > 1 {
            if ! sp[..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                panic!("Sparsity pattern is not sorted");
            }
        }
        if ! sp.last().map_or_else(|| true, |&v| v < self.shape.iter().product()) {
                panic!("Sparsity pattern does not fit in shape");
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : self.shape,
            sp    : Some(sp)
        }
    }

    pub fn with_shape_and_sparsity(self,shape : Vec<usize>, sp : Vec<usize>) -> LinearDomain {
        if sp.len() > 1 {
            if ! sp[..sp.len()-1].iter().zip(sp[1..].iter()).all(|(a,b)| a < b) {
                panic!("Sparsity pattern is not sorted");
            }
        }
        if ! sp.last().map_or_else(|| true, |&v| v < shape.iter().product()) {
                panic!("Sparsity pattern does not fit in shape");
        }
        LinearDomain{
            dt    : self.dt,
            ofs   : self.ofs,
            shape : shape,
            sp    : Some(sp)
        }
    }
}
impl DomainTrait for LinearDomain {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,self.dt,self.ofs)
    }
}

pub trait OffsetTrait {
    fn greater_than(self) -> LinearDomain;
    fn less_than(self)    -> LinearDomain;
    fn equal_to(self)     -> LinearDomain;
}

impl OffsetTrait for f64 {
    fn greater_than(self) -> LinearDomain { LinearDomain{ dt : LinearDomainType::NonNegative, ofs:vec![self], shape:vec![], sp : None } }
    fn less_than(self)    -> LinearDomain { LinearDomain{ dt : LinearDomainType::NonPositive,  ofs:vec![self], shape:vec![], sp : None } }
    fn equal_to(self)     -> LinearDomain { LinearDomain{ dt : LinearDomainType::Zero,         ofs:vec![self], shape:vec![], sp : None } }
}

impl OffsetTrait for Vec<f64> {
    fn greater_than(self) -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:self, shape:vec![n], sp : None } }
    fn less_than(self)    -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:self, shape:vec![n], sp : None } }
    fn equal_to(self)     -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero, ofs:self, shape:vec![n], sp : None } }
}

impl OffsetTrait for &[f64] {
    fn greater_than(self) -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonNegative, ofs:self.to_vec(), shape:vec![n], sp : None } }
    fn less_than(self)    -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::NonPositive, ofs:self.to_vec(), shape:vec![n], sp : None } }
    fn equal_to(self)     -> LinearDomain { let n = self.len(); LinearDomain{ dt : LinearDomainType::Zero, ofs:self.to_vec(), shape:vec![n], sp : None } }
}

pub fn greater_than<T : OffsetTrait>(v : T) -> LinearDomain { v.greater_than() }
pub fn less_than<T : OffsetTrait>(v : T) -> LinearDomain { v.less_than() }
pub fn equal_to<T : OffsetTrait>(v : T) -> LinearDomain { v.equal_to() }
pub fn in_quadratic_cone(dim : usize) -> ConicDomain { ConicDomain{dt:ConicDomainType::QuadraticCone,ofs:vec![0.0; dim],shape:vec![dim],conedim:0} }
pub fn in_rotated_quadratic_cone(dim : usize) -> ConicDomain { ConicDomain{dt:ConicDomainType::RotatedQuadraticCone,ofs:vec![0.0; dim],shape:vec![dim],conedim:0} }
pub fn in_quadratic_cones(shape : Vec<usize>, conedim : usize) -> ConicDomain {
    if conedim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicDomain{dt:ConicDomainType::QuadraticCone,
                ofs : vec![0.0; shape.iter().product()],
                shape:shape,
                conedim:conedim}
}
pub fn in_rotated_quadratic_cones(shape : Vec<usize>, conedim : usize) -> ConicDomain {
    if conedim >= shape.len() {
        panic!("Invalid cone dimension");
    }
    ConicDomain{dt      : ConicDomainType::RotatedQuadraticCone,
                ofs     : vec![0.0; shape.iter().product()],
                shape   : shape,
                conedim : conedim}
}


impl Model {
    pub fn new(name : Option<&str>) -> Model {
        let mut task = mosek::Task::new().unwrap();
        match name {
            Some(name) => task.put_task_name(name).unwrap(),
            None => {}
        }
        Model{
            task      : task,
            vars      : Vec::new(),
            barvarelm : Vec::new(),
            cons      : Vec::new()
        }
    }

    pub fn variable<D : DomainTrait>(& mut self, name : Option<&str>, dom : D) -> Variable {
        dom.create_variable(self,name)
    }

    fn alloc_linear_var(&mut self, size : usize) -> (usize,i32) {
        let firstidx = self.vars.len();
        let firsttaskidx = self.task.get_num_var().unwrap();
        let nvaridx : i32 = size.try_into().unwrap();
        let lasttaskidx = firsttaskidx+nvaridx;
        self.task.append_vars(size as i32).unwrap();

        self.vars.resize(self.vars.len() + size,0);
        self.vars[firstidx..].iter_mut().zip(firsttaskidx..lasttaskidx).for_each(|(a,b)| *a = b as i64);

        (firstidx,firsttaskidx)
    }

    fn linear_variable(&mut self, _name : Option<&str>,bt : LinearDomainType, bound : Vec<f64>) -> Variable {
        let size = bound.len();
        let (firstidx,firsttaskidx) = self.alloc_linear_var(size);
        let lasttaskidx = firsttaskidx + size as i32;

        match bt {
            LinearDomainType::Free        => self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap(),
            LinearDomainType::Zero        => {
                let bk = vec![mosek::Boundkey::FX; size];
                self.task.put_var_bound_slice(firsttaskidx,lasttaskidx,bk.as_slice(),bound.as_slice(),bound.as_slice()).unwrap();
            },
            LinearDomainType::NonNegative => {
                let bk = vec![mosek::Boundkey::LO; size];
                self.task.put_var_bound_slice(firsttaskidx,lasttaskidx,bk.as_slice(),bound.as_slice(),bound.as_slice()).unwrap();
            },
            LinearDomainType::NonPositive => {
                let bk = vec![mosek::Boundkey::UP; size];
                self.task.put_var_bound_slice(firsttaskidx,lasttaskidx,bk.as_slice(),bound.as_slice(),bound.as_slice()).unwrap()
            }
        }

        Variable::new((firstidx..firstidx+size).collect())
    }

    fn free_variable(&mut self, _name : Option<&str>, size : usize) -> Variable {
        let (firstidx,firsttaskidx) = self.alloc_linear_var(size);
        let lasttaskidx = firsttaskidx + size as i32;
        self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
        Variable::new((firstidx..firstidx+size).collect())
    }

    fn conic_variable(&mut self, _name : Option<&str>, size : usize, num : usize, ct : ConicDomainType, ofs : Option<Vec<f64>>) -> Variable {
        let n = size * num;
        let (firstidx,firsttaskidx) = self.alloc_linear_var(n);
        let lasttaskidx = firsttaskidx + n as i32;

        let firstafeidx = self.task.get_num_afe().unwrap();
        self.task.append_afes(size.try_into().unwrap()).unwrap();
        let lastafeidx = firstafeidx + n as i64;
        let afeidxs : Vec<i64> = (firstafeidx..lastafeidx).collect();
        let varidxs : Vec<i32> = (firsttaskidx..lasttaskidx).collect();
        self.task.put_afe_f_entry_list(afeidxs.as_slice(),
                                  varidxs.as_slice(),
                                       vec![1.0; n].as_slice()).unwrap();
        self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
        let firstaccidx = self.task.get_num_acc().unwrap();
        let dom = match ct {
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(size.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(size.try_into().unwrap()).unwrap(),
        };

        match ofs {
            None => self.task.append_accs_seq(vec![dom; num].as_slice(),
                                              n as i64,
                                              firstafeidx,
                                              vec![0.0; n].as_slice()).unwrap(),
            Some(offset) => self.task.append_accs_seq(vec![dom; num].as_slice(),
                                                      n as i64,
                                                      firstafeidx,
                                                      offset.as_slice()).unwrap()
        }

        Variable::new((firstidx..firstidx+size).collect())
    }


    fn parametrized_conic_variable(&mut self, _name : Option<&str>, size : usize, num : usize, ct : ParamConicDomainType, alpha : &[f64], ofs : Option<&[f64]>) -> Variable {
        let n = size * num;
        let (firstidx,firsttaskidx) = self.alloc_linear_var(n);
        let lasttaskidx = firsttaskidx + n as i32;

        let firstafeidx = self.task.get_num_afe().unwrap();
        self.task.append_afes(size.try_into().unwrap()).unwrap();
        let lastafeidx = firstafeidx + n as i64;
        let afeidxs : Vec<i64> = (firstafeidx..lastafeidx).collect();
        let varidxs : Vec<i32> = (firsttaskidx..lasttaskidx).collect();
        self.task.put_afe_f_entry_list(afeidxs.as_slice(),
                                  varidxs.as_slice(),
                                  vec![1.0; n].as_slice()).unwrap();
        self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
        let firstaccidx = self.task.get_num_acc().unwrap();
        let dom = match ct {
            ParamConicDomainType::PrimalPowerCone => self.task.append_primal_power_cone_domain(size.try_into().unwrap(), alpha).unwrap(),
            ParamConicDomainType::DualPowerCone   => self.task.append_dual_power_cone_domain(size.try_into().unwrap(), alpha).unwrap()
        };

        match ofs {
            None => self.task.append_accs_seq(vec![dom; num].as_slice(),
                                              n as i64,
                                              firstafeidx,
                                              vec![0.0; n].as_slice()).unwrap(),
            Some(offset) => self.task.append_accs_seq(vec![dom; num].as_slice(),
                                                      n as i64,
                                                      firstafeidx,
                                                      offset).unwrap()
        }

        Variable::new((firstidx..firstidx+size).collect())
    }
    // fn constraint(& mut self, name : Option<&str>,
    //               expr : & Expr,
    //               dom  : Domain) -> Constraint;
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn it_works() {
        let mut m = Model::new(Some("SuperModel"));
        let mut v1 = m.variable(None, greater_than(5.0));
        let mut v2 = m.variable(None, 10);
        let mut v3 = m.variable(None, vec![3,3]);
        let mut v4 = m.variable(None, in_quadratic_cone(5));
        let mut v5 = m.variable(None, greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(vec![2,2]));
        let mut v6 = m.variable(None, greater_than(vec![1.0,3.0]).with_shape_and_sparsity(vec![2,2],vec![0,3]));
    }
}
