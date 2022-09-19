extern crate mosek;
extern crate itertools;

//mod utils;
pub mod expr;
use itertools::{iproduct};

/////////////////////////////////////////////////////////////////////
// Model, constraint and variables

#[derive(Clone,Copy)]
enum VarAtom {
    // Task variable index
    Linear(i32),
    // Task bar element (barj,k,l)
    BarElm(i32,i32,i32),
    // Conic variable (j,offset)
    ConicElm(i32,usize)
}
enum ConAtom {
    ConicElm(i64,usize)
}
pub struct Model {
    /// The MOSEK task
    task : mosek::Task,
    vars      : Vec<VarAtom>,
    cons      : Vec<ConAtom>,

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
    // fn new(idxs : Vec<usize>) -> Variable {
    //     let n = idxs.len();
    //     Variable {
    //         idxs : idxs,
    //         sparsity : None,
    //         shape : vec![n]
    //     }
    // }

    pub fn with_shape(self, shape : Vec<usize>) -> Variable {
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

    pub fn with_sparsity(self, sp : Vec<usize>) -> Variable {
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

    pub fn with_shape_and_sparsity(self,shape : Vec<usize>, sp : Vec<usize>) -> Variable {
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

    pub fn flatten(self) -> Variable {
        Variable {
            idxs : self.idxs,
            sparsity : self.sparsity,
            shape : vec![self.shape.iter().product()]
        }
    }

    // Other functions to be implemented:
    ///// Take the diagonal element of a square, cube,... variable
    //pub fn diag(&self) -> Variable
    //pub fn into_diag(&self) -> Variable
    //pub slice(& self,from : &[usize], to : &[usize])
    //pub stack(dim : usize, xs : &[&Variable]) -> Variable
    //pub hstack(xs : &[Variable]) -> Variable
    //pub vstack(xs : &[Variable]) -> Variable
}

impl expr::ExprTrait for &Variable {
    fn eval(&self,rs : & mut expr::WorkStack, _ws : & mut expr::WorkStack, _xs : & mut expr::WorkStack) {
        let (rptr,rsp,rsubj,rcof) = rs.alloc_expr(self.shape.as_slice(),
                                                  self.idxs.len(),
                                                  self.idxs.len());
        rptr.iter_mut().enumerate().for_each(|(i,p)| *p = i);
        rsubj.clone_from_slice(self.idxs.as_slice());
        rcof.fill(1.0);
        match (rsp,&self.sparsity) {
            (Some(rsp),Some(sp)) => rsp.clone_from_slice(sp.as_slice()),
            _ => {}
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
        m.conic_variable(name,self)
    }

    /// Add a constraint with expression expected to be on the top of the rs stack.
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.conic_constraint(name,self)
    }
}

// impl DomainTrait for &[i32] {
//     fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
//         m.linear_variable(name,
//                           LinearDomain{
//                               dt:LinearDomainType::Free,
//                               ofs:vec![0.0; self.iter().product::<usize>()],
//                               shape:self.map(|i| i as usize).collect(),
//                               sp:None})
//     }
//     fn create_constraint(self, m : & mut Model, name : Option<&str>, rs : & mut expr::WorkStack) -> Constraint {
//         let c = m.linear_constraint(name,
//                                     LinearDomain{
//                                         dt:LinearDomainType::Free,
//                                         ofs:vec![0.0; self.iter().product::<i32>()],
//                                         shape:self.map(|i| i as usize).collect(),
//                                         sp:None})
//     }
// }

impl DomainTrait for &[usize] {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,self)
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:vec![0.0; self.iter().product::<usize>()],
                                shape:self.to_vec(),
                                sp:None})
    }
}

impl DomainTrait for Vec<usize> {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,self.as_slice())
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
         m.linear_constraint(name,
                             LinearDomain{
                                 dt:LinearDomainType::Free,
                                 ofs:vec![0.0; self.iter().product::<usize>()],
                                 shape:self,
                                 sp:None})
    }
}
impl DomainTrait for usize {
    fn create_variable(self, m : & mut Model, name : Option<&str>) -> Variable {
        m.free_variable(name,&[self])
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,
                            LinearDomain{
                                dt:LinearDomainType::Free,
                                ofs:vec![0.0; self],
                                shape:vec![self],
                                sp:None})
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
        m.linear_variable(name,self)
    }
    fn create_constraint(self, m : & mut Model, name : Option<&str>) -> Constraint {
        m.linear_constraint(name,self)
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
            cons      : Vec::new(),
            rs : expr::WorkStack::new(0),
            ws : expr::WorkStack::new(0),
            xs : expr::WorkStack::new(0)
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

    fn linear_variable(&mut self, _name : Option<&str>,dom : LinearDomain) -> Variable {
        let n = dom.ofs.len();
        let vari = self.task.get_num_var().unwrap();
        let varend : i32= ((vari as usize)+n).try_into().unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.vars.reserve(n);

        let firstvar = self.vars.len();
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j)));

        match dom.dt {
            LinearDomainType::Free        => self.task.put_var_bound_slice_const(vari,vari+n as i32,mosek::Boundkey::FR,0.0,0.0).unwrap(),
            LinearDomainType::Zero        => {
                let bk = vec![mosek::Boundkey::FX; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap();
            },
            LinearDomainType::NonNegative => {
                let bk = vec![mosek::Boundkey::LO; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap();
            },
            LinearDomainType::NonPositive => {
                let bk = vec![mosek::Boundkey::UP; n];
                self.task.put_var_bound_slice(vari,varend,bk.as_slice(),dom.ofs.as_slice(),dom.ofs.as_slice()).unwrap()
            }
        }

        Variable {
            idxs : (firstvar..firstvar+n).collect(),
            sparsity : dom.sp,
            shape : dom.shape
        }
    }

    fn free_variable(&mut self, _name : Option<&str>, shape : &[usize]) -> Variable {
        let vari = self.task.get_num_var().unwrap();
        let n : usize = shape.iter().product();
        let varend : i32 = ((vari as usize) + n).try_into().unwrap();
        let firstvar = self.vars.len();
        self.task.append_vars(n as i32).unwrap();
        self.task.put_var_bound_slice_const(vari,varend,mosek::Boundkey::FR,0.0,0.0).unwrap();
        Variable{
            idxs : (firstvar..firstvar+n).collect(),
            shape : shape.to_vec(),
            sparsity : None}
    }

    fn conic_variable(&mut self, _name : Option<&str>, dom : ConicDomain) -> Variable {
        let n    = dom.shape.iter().product();
        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();
        let vari = self.task.get_num_var().unwrap();

        let asubi : Vec<i64> = (acci..acci+n as i64).collect();
        let asubj : Vec<i32> = (vari..vari+n as i32).collect();
        let acof  : Vec<f64> = vec![1.0; n];

        let d0 : usize = dom.shape[0..dom.conedim].iter().product();
        let d1 : usize = dom.shape[dom.conedim];
        let d2 : usize = dom.shape[dom.conedim+1..].iter().product();
        let conesize = d1;
        let numcone  = d0*d2;

        let domidx = match dom.dt {
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
        };

        self.task.append_afes(n as i64).unwrap();
        self.task.append_vars(n.try_into().unwrap()).unwrap();
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),n as i64,afei,dom.ofs.as_slice()).unwrap();
        self.task.put_afe_f_entry_list(asubi.as_slice(),asubj.as_slice(),acof.as_slice()).unwrap();

        let firstvar = self.vars.len();
        self.vars.reserve(n);
        self.cons.reserve(n);

        iproduct!(0..d0,0..d1,0..d2).enumerate()
            .for_each(|(i,(i0,i1,i2))| {
                self.vars.push(VarAtom::ConicElm(vari+i as i32,self.cons.len()));
                self.cons.push(ConAtom::ConicElm(acci+i1 as i64,i0*d2+i2))
            } );

        Variable{
            idxs     : (firstvar..firstvar+n).collect(),
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

    pub fn constraint<E : expr::ExprTrait, D : DomainTrait>(& mut self, name : Option<&str>, expr : &E, dom : D) -> Constraint {
        expr.eval_finalize(& mut self.rs,& mut self.ws,& mut self.xs);
        dom.create_constraint(self,name)
    }

    fn linear_constraint(& mut self,
                         _name : Option<&str>,
                         dom  : LinearDomain) -> Constraint {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain");
        }
        let nnz = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }

        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();

        let nlinnz = subj.iter().filter(|&&j| if let VarAtom::BarElm(_,_,_) = unsafe { *self.vars.get_unchecked(j) } { false } else { true } ).count();
        let npsdnz = nnz - nlinnz;

        self.task.append_afes(nelm as i64).unwrap();
        let domidx = match dom.dt {
            LinearDomainType::NonNegative => self.task.append_rplus_domain(nelm as i64).unwrap(),
            LinearDomainType::NonPositive => self.task.append_rminus_domain(nelm as i64).unwrap(),
            LinearDomainType::Zero        => self.task.append_rzero_domain(nelm as i64).unwrap(),
            LinearDomainType::Free        => self.task.append_r_domain(nelm as i64).unwrap(),
        };

        match dom.sp {
            None => self.task.append_acc_seq(domidx, afei,dom.ofs.as_slice()).unwrap(),
            Some(sp) => {
                let mut ofs = vec![0.0; nelm];
                if sp.len() != dom.ofs.len() { panic!("Broken sparsity pattern") };
                if let Some(&v) = sp.iter().max() { if v >= nelm { panic!("Broken sparsity pattern"); } }
                sp.iter().zip(dom.ofs.iter()).for_each(|(&ix,&c)| unsafe { *ofs.get_unchecked_mut(ix) = c; } );
                self.task.append_acc_seq(domidx, afei,ofs.as_slice()).unwrap();
            }
        }

        let firstcon = self.cons.len();
        self.cons.reserve(nelm);
        (0..nelm).for_each(|i| self.cons.push(ConAtom::ConicElm(acci,i)));

        let mut asubj : Vec<i32> = Vec::with_capacity(nnz);
        let mut acof  : Vec<f64> = Vec::with_capacity(nnz);
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
                if idx == 0 {
                    cfix += c;
                }
                else {
                    match *unsafe{ self.vars.get_unchecked(idx-1) } {
                        VarAtom::Linear(j) => {
                            asubj.push(j);
                            acof.push(c);
                        },
                        VarAtom::ConicElm(j,_coni) => {
                            asubj.push(j);
                            acof.push(c);
                        },
                        VarAtom::BarElm(j,k,l) => {
                            abarsubi.push(afei+i as i64);
                            abarsubj.push(j);
                            abarsubk.push(k);
                            abarsubl.push(l);
                            abarcof.push(c);
                        }
                    }
                }
            });
            aptr.push(asubj.len() as i64);
            afix.push(cfix);
        });

        let afeidxs : Vec<i64> = (afei..afei+nelm as i64).collect();
        self.task.put_afe_f_row_list(
            afeidxs.as_slice(),
            aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0).try_into().unwrap()).collect::<Vec<i32>>().as_slice(),
            &aptr[..nelm],
            asubj.as_slice(),
            acof.as_slice()).unwrap();
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();
        if npsdnz > 0 {
            self.task.put_afe_barf_block_triplet(abarsubi.as_slice(),
                                                 abarsubj.as_slice(),
                                                 abarsubk.as_slice(),
                                                 abarsubl.as_slice(),
                                                 abarcof.as_slice()).unwrap();
        }

        Constraint{
            idxs : (firstcon..firstcon+nelm).collect(),
            shape : dom.shape
        }
    }

    fn conic_constraint(& mut self,
                        _name : Option<&str>,
                        dom  : ConicDomain) -> Constraint {
        let (shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ! dom.shape.iter().zip(shape.iter()).all(|(&a,&b)| a==b) {
            panic!("Mismatching shapes of expression and domain");
        }
        let nnz  = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }
        if shape.iter().zip(dom.shape.iter()).all(|(&d0,&d1)| d0==d1 ) {
            panic!("Mismatching domain/expression shapes");
        }

        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();

        let nlinnz = subj.iter().filter(|&&j| if let VarAtom::BarElm(_,_,_) = unsafe { *self.vars.get_unchecked(j) } { false } else { true } ).count();
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
                if idx == 0 {
                    cfix += c;
                }
                else {
                    match *unsafe{ self.vars.get_unchecked(idx-1) } {
                        VarAtom::Linear(j) => {
                            asubj.push(j);
                            acof.push(c);
                        },
                        VarAtom::ConicElm(j,_coni) => {
                            asubj.push(j);
                            acof.push(c);
                        },
                        VarAtom::BarElm(j,k,l) => {
                            abarsubi.push(afei+i as i64);
                            abarsubj.push(j);
                            abarsubk.push(k);
                            abarsubl.push(l);
                            abarcof.push(c);
                        }
                    }
                }
            });
            aptr.push(asubj.len() as i64);
            afix.push(cfix);
        });

        let conesize = shape[dom.conedim];
        let numcone  = shape.iter().product::<usize>() / conesize;

        let domidx = match dom.dt {
            ConicDomainType::QuadraticCone        => self.task.append_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
            ConicDomainType::RotatedQuadraticCone => self.task.append_r_quadratic_cone_domain(conesize.try_into().unwrap()).unwrap(),
        };

        self.task.append_afes(nelm as i64).unwrap();
        self.task.append_accs_seq(vec![domidx; numcone].as_slice(),
                                  nelm as i64,
                                  afei,
                                  dom.ofs.as_slice()).unwrap();
        let d0 : usize = shape[0..dom.conedim].iter().product();
        let d1 : usize = shape[dom.conedim];
        let d2 : usize = shape[dom.conedim+1..].iter().product();
        let afeidxs : Vec<i64> = iproduct!(0..d0,0..d2,0..d1)
            .map(|(i0,i2,i1)| afei + (i0*d1*d2 + i1*d2 + i2) as i64)
            .collect();

        // if let Some(name) = name {
        //     let accshape = shape[0..dom.conedim].iter().join(shape[dom.conedim+1..].iter()).collect();
        //     //iproduct!(0..d0,0..d2).for_each(|(i0,i2)| self.task.put_acc_name(format!("{}[{},*,{}]")).unwrap() )
                
        //     }
        // }
        
        if nlinnz > 0 {
            self.task.put_afe_f_row_list(afeidxs.as_slice(),
                                         aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                         &aptr[..nelm],
                                         asubj.as_slice(),
                                         acof.as_slice()).unwrap();
        }
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();
        if npsdnz > 0 {
            self.task.put_afe_barf_block_triplet(abarsubi.as_slice(),
                                                 abarsubj.as_slice(),
                                                 abarsubk.as_slice(),
                                                 abarsubl.as_slice(),
                                                 abarcof.as_slice()).unwrap();
        }

        let coni = self.cons.len();
        self.cons.reserve(nelm);
        iproduct!(0..d0,0..d1,0..d2)
            .for_each(|(i0,i1,i2)| self.cons.push(ConAtom::ConicElm(acci+i1 as i64,i0*d2+i2)));

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
    use super::expr::Expr;
    #[test]
    fn it_works() {
        let mut m = Model::new(Some("SuperModel"));
        let mut v1 = m.variable(None, greater_than(5.0));
        let mut v2 = m.variable(None, 10);
        let mut v3 = m.variable(None, vec![3,3]);
        let mut v4 = m.variable(None, in_quadratic_cone(5));
        let mut v5 = m.variable(None, greater_than(vec![1.0,2.0,3.0,4.0]).with_shape(vec![2,2]));
        let mut v6 = m.variable(None, greater_than(vec![1.0,3.0]).with_shape_and_sparsity(vec![2,2],vec![0,3]));

        let e1 = Expr::from_variable(&v1);
        let e2 = Expr::from_variable(&v3);
    }
}
