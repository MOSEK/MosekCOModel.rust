extern crate mosek;
extern crate itertools;

mod utils;
mod expr;
use itertools::{iproduct,izip};

/////////////////////////////////////////////////////////////////////
// Model, constraint and variables

pub struct Model {
    /// The MOSEK task
    task : mosek::Task,

    /// Mapping from Model variable atoms to Mosek variables.
    ///
    /// If `i` is the Model variable index and `(k,l) = vars[i]` then
    /// - if `k > 0` then `k-1` is the index of
    ///   the Task variable, and if `l>0` then then `l-1` is the index
    ///   of a constraint element defining the conic constraint of the
    ///   variable. If `l` == 0 then the variable will have an upper
    ///   bound, a lower bound, be fixed or be free.
    /// - if `k == 0` then it is a const term (interpreted as vars[0] being a variable fixed to 1.0)
    /// - if `k < 0` then i is a PSD variable entry and `-(k+1)` the index into barvarelm
    vars      : Vec<(i64,usize)>,
    /// Mapping from Model PSD variable index to `(barj,offset)`.  If
    /// `(barj,ofs) = barvarelm[i] `, then the `i`th entry corresponds
    /// to linear offset `ofs`, counting in row-major format, into
    /// PSD variable `barj`. A side-effect is that when fetching a PSD
    /// solution from mosek for all barvars, the result entries
    /// correspond directly to the indexes in barvarelm.
    barvarelm : Vec<(usize,usize)>,
    /// Mapping from Model constraint index to `(acci,ofs)`.
    cons      : Vec<(usize,usize)>,

    /// Workstacks for evaluating expressions
    rs : expr::WorkStack,
    ws : expr::WorkStack,
    xs : expr::WorkStack
}

/// A Variable object is basically a wrapper around a variable index
/// list with a shape and a sparsity pattern. 
#[derive(Clone)]
pub struct Variable {
    idxs     : Vec<usize>,
    sparsity : Option<Vec<usize>>,
    shape    : Vec<usize>
}

/// A Constraint object is a wrapper around an array of constraint
/// indexes and a shape. Note that constraint objects are never sparse.
#[derive(Clone)]
pub struct Constraint {
    idxs     : Vec<usize>,
    shape    : Vec<usize>
}
/////////////////////////////////////////////////////////////////////
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
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint;
}

pub struct LinearDomain {
    dt    : LinearDomainType,
    ofs   : Vec<f64>,
    shape : Vec<usize>,
    sp    : Option<Vec<usize>>
}

pub struct ConicDomain {
    dt      : ConicDomainType,
    ofs     : Vec<f64>,
    shape   : Vec<usize>,
    conedim : usize
}

impl DomainTrait for ConicDomain {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        let idxs = m.conic_variable(name,size,num,self.dt,Some(self.ofs));

        let size = self.shape[self.conedim];
        let num  = self.shape.iter().product::<usize>() / size;
        let idxs = m.conic_variable(name,size,num,self.dt,Some(self.ofs));
        if self.conedim < self.shape.len()-1 {
            // permute the indexes
            let d0 = self.shape[..self.conedim].iter().product();
            let d1 = self.shape[self.conedim];
            let d2 = self.shape[self.conedim+1..].iter().product();
            let idxs : Vec<usize> = iproduct!(0..d0,0..d1,0..d2).map(|(i0,i1,i2)| unsafe { *idxs.get_unchecked(i0*d1*d2+i2*d1+i1) }).collect();
            Variable {
                idxs,
                sparsity : None,
                shape : self.shape }
        }
        else {
            Variable {
                idxs,
                None,
                shape : self.shape }
        }
    }

    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.conic_constraint(name,self).collect())
    }
}

impl DomainTrait for &[i32] {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,LinearDomainType::Free,vec![0.0; self.iter().product::<i32>().try_into().unwrap()]).with_shape(self.iter().map(|&v| v.try_into().unwrap()).collect::<Vec<usize>>())
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>, rs : & mut expr::WorkStack) -> Constraint {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        let c = m.linear_constraint(name,ptr,subj,cof);
    }
}

impl DomainTrait for &[usize] {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.linear_variable(name,LinearDomainType::Free,vec![0.0; self.iter().product()]).with_shape(self.to_vec())
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>, rs : & mut expr::WorkStack) -> Constraint {
        let (shape,ptr,sp,subj,cof) = rs.pop_expr();
        let c = m.linear_constraint(name,ptr,subj,cof);
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
        Variable{
            idxs     : m.linear_variable(name,self.dt,self.ofs),
            sparsity : self.sp,
            shape    : self.shape
        }
    }

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

    ////////////////////////////////////////////////////////////
    // Variable interface

    pub fn variable<D : DomainTrait>(& mut self, name : Option<&str>, dom : D) -> Variable {
        dom.create_variable(self,name)
    }

    // fn alloc_linear_var(&mut self, size : usize) -> (usize,i32) {
    //     let firstidx = self.vars.len();
    //     let firsttaskidx = self.task.get_num_var().unwrap();
    //     let nvaridx : i32 = size.try_into().unwrap();
    //     let lasttaskidx = firsttaskidx+nvaridx;
    //     self.task.append_vars(size as i32).unwrap();

    //     self.vars.resize(self.vars.len() + size,0);
    //     self.vars[firstidx..].iter_mut().zip(firsttaskidx..lasttaskidx).for_each(|(a,b)| *a = b as i64);

    //     (firstidx,firsttaskidx)
    // }

    fn linear_variable(&mut self, _name : Option<&str>,dom : LinearDomain) -> Vec<usize> {
        let n = dom.ofs.len();
        let vari = self.task.get_num_var().unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.vars.reserve(n);
        (vari..vari+n as i32).for_each(|j| self.vars.push((1+vari as i64,0)));

        match bt {
            LinearDomainType::Free        => self.task.put_var_bound_slice_const(vari,vari+n as i32,mosek::Boundkey::FR,0.0,0.0).unwrap(),
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

    fn free_variable(&mut self, _name : Option<&str>, size : usize) -> Vec<usize> {
        let (firstidx,firsttaskidx) = self.alloc_linear_var(size);
        let lasttaskidx = firsttaskidx + size as i32;
        self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
        (firstidx..firstidx+size).collect()
    }

    fn conic_variable(&mut self, _name : Option<&str>, dom : ConicDomain, ofs : Option<Vec<f64>>) -> Vec<usize> {
        let n    = dom.shape.iter().product();
        let acci = self.task.get_num_acc()?;
        let afei = self.task.get_num_afe()?;
        let vari = self.task.get_num_var()?;

        let mut asubi : Vec<i64> = (acci..acci+n).collect();
        let mut asubj : Vec<i32> = (vari..vari+n).collect();
        let mut acof  : Vec<f64> = vec![1.0; n];

        let d0 : usize = shape[0..self.conedim].iter().product();
        let d1 : usize = shape[self.conedim].iter().product();
        let d2 : usize = shape[self.conedim+1..].iter().product();
        let conesize = d1;
        let numcone  = d0*d2;

        let domidx = match ct {
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
        };

        task.append_afes(n as i64).unwrap();
        task.append_vars(n.try_into().unwrap());
        task.append_accs_seq(vec![domidx; numcone].as_slice(),afei,dom.ofs.as_slice()).unwrap();
        task.put_afe_f_entry_list(asubi.as_slice(),asubj.as_slice(),acof.as_slice()).unwrap();

        self.vars.reserve(n);
        self.cons.reserve(n);

        iproduct!(0..d0,0..d1,0..d2).enumerate()
            .for_each(|(i,(i0,i1,i2))| {
                self.vars.push((vari+i,-(self.cons.len() as i64 + 1)));
                self.cons.push((acci+i1,i0*d2+i2))
            } );

        Variable{
            idxs     : (vari..vari+n).collect(),
            sparsity : None,
            shape    : dom.shape
        }
    }


    // fn parametrized_conic_variable(&mut self, _name : Option<&str>, size : usize, num : usize, ct : ParamConicDomainType, alpha : &[f64], ofs : Option<&[f64]>) -> Vec<usize> {
    //     let n = size * num;
    //     let (firstidx,firsttaskidx) = self.alloc_linear_var(n);
    //     let lasttaskidx = firsttaskidx + n as i32;

    //     let firstafeidx = self.task.get_num_afe().unwrap();
    //     self.task.append_afes(size.try_into().unwrap()).unwrap();
    //     let lastafeidx = firstafeidx + n as i64;
    //     let afeidxs : Vec<i64> = (firstafeidx..lastafeidx).collect();
    //     let varidxs : Vec<i32> = (firsttaskidx..lasttaskidx).collect();
    //     self.task.put_afe_f_entry_list(afeidxs.as_slice(),
    //                               varidxs.as_slice(),
    //                               vec![1.0; n].as_slice()).unwrap();
    //     self.task.put_var_bound_slice_const(firsttaskidx,lasttaskidx,mosek::Boundkey::FR,0.0,0.0).unwrap();
    //     let firstaccidx = self.task.get_num_acc().unwrap();
    //     let dom = match ct {
    //         ParamConicDomainType::PrimalPowerCone => self.task.append_primal_power_cone_domain(size.try_into().unwrap(), alpha).unwrap(),
    //         ParamConicDomainType::DualPowerCone   => self.task.append_dual_power_cone_domain(size.try_into().unwrap(), alpha).unwrap()
    //     };

    //     match ofs {
    //         None => self.task.append_accs_seq(vec![dom; num].as_slice(),
    //                                           n as i64,
    //                                           firstafeidx,
    //                                           vec![0.0; n].as_slice()).unwrap(),
    //         Some(offset) => self.task.append_accs_seq(vec![dom; num].as_slice(),
    //                                                   n as i64,
    //                                                   firstafeidx,
    //                                                   offset).unwrap()
    //     }

    //     (firstidx..firstidx+size).collect()
    // }

    ////////////////////////////////////////////////////////////
    // Constraint interface

    fn constraint_(& mut self,
                   name : Option<&str>,
                   domidx : i64) {
    }

    pub fn constraint<E : expr::ExprTrait, D : DomainTrait>(& mut self, name : Option<&str>, expr : &E, dom : D) -> Constraint {
        expr.eval(& mut self.rs,& mut self.ws,& mut self.xs);
        dom.create_constraint(& mut self)
    }

    fn linear_constraint(& mut self,
                         name : Option<&str>,
                         dom  : LinearDomain) -> Constraint {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain");
        }
        let nnz = subj.len();
        let nelm = ptr.len()-1;

        if subj.iter().max() >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }

        let acci = self.task.get_num_acc()?;
        let afei = self.task.get_num_afe()?;

        self.task.append_afes(nelm)?;
        let domidx = match dom.dt {
            NonNegative => self.task.append_rplus_domain(nelm as i64)?,
            NonPositive => self.task.append_rminus_domain(nelm as i64)?,
            Zero        => self.task.append_rzero_domain(nelm as i64)?,
            Free        => self.task.append_r_domain(nelm as i64)?,
        };


        self.task.append_acc_seq(domidx, afei,dom.ofs.as_slice())?;


        let firstcon = self.cons.len();
        self.cons.reserve(nelm);
        (0..nelm).for_each(|i| self.cons.push((acci,i)));

        let asubj : Vec<i32> = Vec::with_capacity(nnz);
        let acof  : Vec<f64> = Vec::with_capacity(nnz);
        let aptr  : Vec<i64> = Vec::with_capacity(nelm+1);
        let afix  : Vec<f64> = Vec::with_capacity(nelm);
        aptr.push(0);
        ptr[..ptr.len()-1].iter().zip(ptr[1..].iter()).for_each(|(&p0,&p1)| {
            let mut cfix = 0.0;
            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&idx,&c)| {
                let j = *unsafe{ self.vars.get_unchecked(idx) };
                if j == 0 {
                    cfix += c;
                }
                else if j > 0 {
                    asubj.push((j-1) as i32);
                    acof.push(c);
                }
            });
            aptr.push(asubj.len());
            afix.push(cfix);
        });

        for (p0,p1,fixterm) in izip!(aptr[0..aptr.len()-1].iter(),aptr[1..].iter(),afix.iter()) {
            self.task.put_afe_f_row(afei,afei+nelm,&asubj[p0..p1],&acof[p0..p1]);
            self.task.put_afe_g(afei,fixterm);
        }

        Constraint{
            idxs : (firstcon..firstcon+nelm).collect(),
            shape : dom.shape
        }
    }

    fn conic_constraint(& mut self,
                        name : Option<&str>,
                        dom  : ConicDomain) -> Constraint {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain");
        }
        let nnz  = subj.len();
        let nelm = ptr.len()-1;

        if subj.iter().max() >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }
        if shape.iter().zip(dom.shape.iter()).all(|(&d0,&d1)| d0==d1 ) {
            panic!("Mismatching domain/expression shapes");
        }

        let acci = self.task.get_num_acc()?;
        let afei = self.task.get_num_afe()?;

        let nlinnz = subj.iter().filter(|&j| 0 < self.vars.get_uncheched(j) ).count();
        let npsdnz = nnz - nlinnz;

        let mut asubj : Vec<i32> = Vec::with_capacity(nlinnz);
        let mut acof  : Vec<f64> = Vec::with_capacity(nlinnz);
        let mut aptr  : Vec<i64> = Vec::with_capacity(nelm+1);

        let mut afix  : Vec<f64> = Vec::with_capacity(nelm);

        let mut abarsubi : Vec<i64> = Vec::with_capacity(nlinnz);
        let mut abarsubj : Vec<i32> = Vec::with_capacity(nlinnz);
        let mut abarsubk : Vec<i32> = Vec::with_capacity(nlinnz);
        let mut abarsubl : Vec<i32> = Vec::with_capacity(nlinnz);
        let mut abarcof  : Vec<f64> = Vec::with_capacity(nlinnz);

        aptr.push(0);
        ptr[..ptr.len()-1].iter().zip(ptr[1..].iter()).enumerate().for_each(|(i,(&p0,&p1))| {
            let mut cfix = 0.0;
            subj[p0..p1].iter().zip(cof[p0..p1].iter()).for_each(|(&idx,&c)| {
                let j = *unsafe{ self.vars.get_unchecked(idx) };
                if j == 0 {
                    cfix += c;
                }
                else if j > 0 {
                    asubj.push((j-1) as i32);
                    acof.push(c);
                }
                else {
                    let (j,ofs) = unsafe { *self.barvarelm.get_unchecked(-j-1) };
                    abarsubi.push(afei+i as i64);
                    abarsubj.push(j as i32);
                    let k = ((((1+2*ofs) as f64).sqrt() - 1.0) / 2.0).floor() as usize;
                    let l = ofs - k;
                    abarsubk.push(k);
                    abarsubl.push(l);
                }
            });
            aptr.push(asubj.len());
            afix.push(cfix);
        });

        let conesize = shape[self.conedim];
        let numcone  = shape.iter().product::<usize>() / conesize;

        let domidx = match ct {
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(size.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(size.try_into().unwrap()).unwrap(),
        };

        task.append_afes(nelm).unwrap();
        task.append_accs_seq(vec![domidx; numcone].as_slice(),afei,dom.ofs.as_slice()).unwrap();

        let d0 : usize = shape[0..self.conedim].iter().product();
        let d1 : usize = shape[self.conedim].iter().product();
        let d2 : usize = shape[self.conedim+1..].iter().product();
        let afeidxs : Vec<i64> = iproduct!(0..d0,0..d2,0..d1)
            .map(|(i0,i2,i1)| afei + (i0*d1*d2 + i1*d2 + i2) as i64)
            .collect();

        if nlinnz > 0 {
            task.put_afe_f_row_list(afeidx.as_slice(),
                                    aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i64).collect::<Vec<f64>>(),
                                    &aptr[..nelm],
                                    asubj.as_slice(),
                                    acof.as_slice()).unwrap();
        }
        task.put_afe_g_list(afeidx.as_slice(),afix.as_slice()).unwrap();
        if nbarnz > 0 {
            task.put_afe_barf_block_triplet(abarsubi.as_slice(),
                                            abarsubj.as_slice(),
                                            abarsubk.as_slice(),
                                            abarsubl.as_slice(),
                                            abarcof.as_slice()).unwrap();
        }

        let coni = self.cons.len();
        self.cons.reserve(nelm);
        iproduct!(0..d0,0..d1,0..d2)
            .for_each(|(i0,i1,i2)| self.cons.push((acci+i1,i0*d2+i2)));

        Constraint{
            idxs : (coni..coni+nelm).collect(),
            shape : dom.shape
        }
    }

    ////////////////////////////////////////////////////////////
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
