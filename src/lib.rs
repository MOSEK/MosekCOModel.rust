extern crate mosek;
extern crate itertools;

mod utils;
pub mod expr;
use itertools::{iproduct};

//use utils::*;

/////////////////////////////////////////////////////////////////////
// Model, constraint and variables

/// Objective sense
#[derive(Clone,Copy)]
pub enum Sense {
    Maximize,
    Minimize
}

#[derive(Clone,Copy)]
enum VarAtom {
    // Task variable index
    Linear(i32),
    // Task bar element (barj,k,l)
    BarElm(i32,usize),
    // Conic variable (j,offset)
    ConicElm(i32,usize)
}
#[derive(Clone,Copy)]
enum ConAtom {
    ConicElm(i64,usize)
}

#[derive(Clone,Copy)]
pub enum SolutionType {
    Default,
    Basic,
    Interior,
    Integer
}


#[derive(Clone,Copy)]
enum SolutionStatus {
    Optimal,
    Feasible,
    CertInfeas,
    CertIllposed,
    Unknown,
    Undefined
}

struct SolutionPart {
    status : SolutionStatus,
    var : Vec<f64>,
    con : Vec<f64>
}
impl SolutionPart {
    fn new(numvar : usize, numcon : usize) -> SolutionPart { SolutionPart{status : SolutionStatus::Unknown, var : vec![0.0; numvar], con : vec![0.0; numcon] } }
    fn resize(& mut self,numvar : usize, numcon : usize) {
        self.var.resize(numvar, 0.0);
        self.con.resize(numcon, 0.0);
    }
}

struct Solution {
    primal : SolutionPart,
    dual   : SolutionPart
}

impl Solution {
    fn new() -> Solution { Solution{primal : SolutionPart::new(0,0) , dual : SolutionPart::new(0,0)  } }
}
/// The Model object encapsulates an optimization problem and a
/// mapping from the structured API to the internal Task items.
pub struct Model {
    /// The MOSEK task
    task : mosek::Task,
    vars      : Vec<VarAtom>,
    cons      : Vec<ConAtom>,

    sol_bas : Solution,
    sol_itr : Solution,
    sol_itg : Solution,

    /// Workstacks for evaluating expressions
    rs : expr::WorkStack,
    ws : expr::WorkStack,
    xs : expr::WorkStack
}

trait ModelItem {
    fn len(&self) -> usize;
    fn primal(&self,m : &Model,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.primal_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn dual(&self,m : &Model,solid : SolutionType) -> Result<Vec<f64>,String> {
        let mut res = vec![0.0; self.len()];
        self.dual_into(m,solid,res.as_mut_slice())?;
        Ok(res)
    }
    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String>;
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String>;
}

/// A Variable object is basically a wrapper around a variable index
/// list with a shape and a sparsity pattern.
#[derive(Clone)]
pub struct Variable {
    idxs     : Vec<usize>,
    sparsity : Option<Vec<usize>>,
    shape    : Vec<usize>
}

impl ModelItem for Variable {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; });
            }
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_var_solution(solid,self.idxs.as_slice(),res)?;
            if let Some(ref sp) = self.sparsity {
                sp.iter().enumerate().rev().for_each(|(i,&ix)| unsafe { *res.get_unchecked_mut(ix) = *res.get_unchecked(i); *res.get_unchecked_mut(i) = 0.0; })
            }
            Ok(sz)
        }
    }
}

/// A Constraint object is a wrapper around an array of constraint
/// indexes and a shape. Note that constraint objects are never sparse.
#[derive(Clone)]
pub struct Constraint {
    idxs     : Vec<usize>,
    shape    : Vec<usize>
}

impl ModelItem for Constraint {
    fn len(&self) -> usize { return self.shape.iter().product(); }
    fn primal_into(&self,m : &Model,solid : SolutionType, res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.primal_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
    fn dual_into(&self,m : &Model,solid : SolutionType,   res : & mut [f64]) -> Result<usize,String> {
        let sz = self.shape.iter().product();
        if res.len() < sz { panic!("Result array too small") }
        else {
            m.dual_con_solution(solid,self.idxs.as_slice(),res)?;
            Ok(sz)
        }
    }
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

impl expr::ExprTrait for Variable {
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
            vars      : vec![VarAtom::Linear(-1)],
            cons      : Vec::new(),
            sol_bas   : Solution::new(),
            sol_itr   : Solution::new(),
            sol_itg   : Solution::new(),
            rs : expr::WorkStack::new(0),
            ws : expr::WorkStack::new(0),
            xs : expr::WorkStack::new(0)
        }
    }

    pub fn write_problem(&self, filename : &str) {
        self.task.write_data(filename).unwrap();
    }
    
    ////////////////////////////////////////////////////////////
    // Variable interface

    /// Add a Variable
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `dom` The domain of the variable. This defines the bound
    ///   type, shape and sparsity of the variable. For sparse
    ///   variables, elements outside of the sparsity pattern are
    ///   treated as variables fixed to 0.0.
    pub fn variable<D : DomainTrait>(& mut self, name : Option<&str>, dom : D) -> Variable {
        dom.create_variable(self,name)
    }

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
        self.vars.reserve(n);
        (vari..vari+n as i32).for_each(|j| self.vars.push(VarAtom::Linear(j)));
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


    /// Add a constraint
    ///
    /// Note that even if the domain or the expression are sparse, a constraint will always be full.
    ///
    /// # Arguments
    /// - `name` Optional constraint name
    /// - `expr` Constraint expression. Note that the shape of the expression and the domain must match exactly.
    /// - `dom`  The domain of the constraint. This defines the bound type and shape.
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
        // let nnz = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }

        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();

        // let nlinnz = subj.iter().filter(|&&j| if let VarAtom::BarElm(_,_) = unsafe { *self.vars.get_unchecked(j) } { false } else { true } ).count();
        // let npsdnz = nnz - nlinnz;

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

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());
        let abarsubi : Vec<i64> = abarsubi.iter().map(|&i| i + afei).collect();

        let afeidxs : Vec<i64> = (afei..afei+nelm as i64).collect();
        if asubj.len() > 0 {
            self.task.put_afe_f_row_list(
                afeidxs.as_slice(),
                aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0).try_into().unwrap()).collect::<Vec<i32>>().as_slice(),
                &aptr[..nelm],
                asubj.as_slice(),
                acof.as_slice()).unwrap();
        }
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();

        if abarsubi.len() > 0 {
            for (i,j,subk,subl,cof) in utils::ijkl_slice_iterator(abarsubi.as_slice(),                                                 abarsubj.as_slice(),
                                                                  abarsubj.as_slice(),                                                 abarsubj.as_slice(),
                                                                  abarsubk.as_slice(),
                                                                  abarsubl.as_slice(),
                                                                  abarcof.as_slice()) {
                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
            }
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
        // let nnz  = subj.len();
        let nelm = ptr.len()-1;

        if *subj.iter().max().unwrap_or(&0) >= self.vars.len() {
            panic!("Invalid subj index in evaluated expression");
        }
        if shape.iter().zip(dom.shape.iter()).all(|(&d0,&d1)| d0==d1 ) {
            panic!("Mismatching domain/expression shapes");
        }

        let acci = self.task.get_num_acc().unwrap();
        let afei = self.task.get_num_afe().unwrap();

        let (asubj,
             acof,
             aptr,
             afix,
             abarsubi,
             abarsubj,
             abarsubk,
             abarsubl,
             abarcof) = split_expr(ptr,subj,cof,self.vars.as_slice());
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

        if asubj.len() > 0 {
            self.task.put_afe_f_row_list(afeidxs.as_slice(),
                                         aptr[..nelm].iter().zip(aptr[1..].iter()).map(|(&p0,&p1)| (p1-p0) as i32).collect::<Vec<i32>>().as_slice(),
                                         &aptr[..nelm],
                                         asubj.as_slice(),
                                         acof.as_slice()).unwrap();
        }
        self.task.put_afe_g_list(afeidxs.as_slice(),afix.as_slice()).unwrap();
        if abarsubi.len() > 0 {
            for (i,j,subk,subl,cof) in utils::ijkl_slice_iterator(abarsubi.as_slice(),                                                 abarsubj.as_slice(),
                                                                  abarsubj.as_slice(),                                                 abarsubj.as_slice(),
                                                                  abarsubk.as_slice(),
                                                                  abarsubl.as_slice(),
                                                                  abarcof.as_slice()) {
                let dimbarj = self.task.get_dim_barvar_j(j).unwrap();
                let matidx = self.task.append_sparse_sym_mat(dimbarj,subk,subl,cof).unwrap();
                self.task.put_afe_barf_entry(afei+i,j,&[matidx],&[1.0]).unwrap();
            }
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
    // Objective 
    fn set_objective(& mut self, name : Option<&str>, sense : Sense) {
        let (_shape,ptr,_sp,subj,cof) = self.rs.pop_expr();
        if ptr.len()-1 > 1 { panic!("Objective expressions may only contain one element"); }

        if let Some(name) = name { self.task.put_obj_name(name).unwrap(); }
        else { self.task.put_obj_name("").unwrap(); }
        let (csubj,
             ccof,
             _cptr,
             cfix,
             _barsubi,
             barsubj,
             barsubk,
             barsubl,
             barcof) = split_expr(ptr,subj,cof,self.vars.as_slice());

        let numvar : usize = self.task.get_num_var().unwrap().try_into().unwrap();

        if let Some(&j) = csubj.iter().min() { if j < 0 { panic!("Invalid expression") } }
        if let Some(&j) = csubj.iter().max() { if j as usize >= numvar { panic!("Invalid expression") } }

        let mut c = vec![0.0; numvar];
        csubj.iter().zip(ccof.iter()).for_each(|(&j,&v)| unsafe{ * c.get_unchecked_mut(j as usize) = v });
        let csubj : Vec<i32> = (0i32..numvar as i32).collect();
        self.task.put_c_list(csubj.as_slice(),
                             c.as_slice()).unwrap();
        self.task.put_cfix(cfix[0]).unwrap();

        self.task.put_barc_block_triplet(barsubj.as_slice(),barsubk.as_slice(),barsubl.as_slice(),barcof.as_slice()).unwrap();

        match sense {
            Sense::Minimize => self.task.put_obj_sense(mosek::Objsense::MINIMIZE).unwrap(),
            Sense::Maximize => self.task.put_obj_sense(mosek::Objsense::MAXIMIZE).unwrap()
        }
    }

    /// Set the objective
    pub fn objective<E : expr::ExprTrait>(& mut self,
                                    name  : Option<&str>,
                                    sense : Sense,
                                    expr  : & E) {
        expr.eval_finalize(& mut self.rs,& mut self.ws, & mut self.xs);
        self.set_objective(name,sense);
    }

    ////////////////////////////////////////////////////////////
    // Optimize

    pub fn solve(& mut self) {
        self.task.put_int_param(mosek::Iparam::REMOVE_UNUSED_SOLUTIONS, 1);
        self.task.optimize().unwrap();

        let numvar = self.task.get_num_var().unwrap() as usize;
        let numcon = self.task.get_num_con().unwrap() as usize;
        let numacc = self.task.get_num_acc().unwrap() as usize;
        let numaccelm = self.task.get_acc_n_tot().unwrap() as usize;
        let numbarvar = self.task.get_num_barvar().unwrap() as usize;
        let numbarvarelm : usize = (0..numbarvar).map(|j| self.task.get_len_barvar_j(j as i32).unwrap() as usize).sum();

        let mut xx = vec![0.0; numvar];
        let mut slx = vec![0.0; numvar];
        let mut sux = vec![0.0; numvar];
        let mut xc = vec![0.0; numcon];
        let mut y = vec![0.0; numcon];
        let mut slc = vec![0.0; numcon];
        let mut suc = vec![0.0; numcon];
        let mut accx = vec![0.0; numaccelm];
        let mut doty = vec![0.0; numaccelm];
        let mut barx = vec![0.0; numbarvarelm];
        let mut bars = vec![0.0; numbarvarelm];

        let mut dimbarvar : Vec<usize> = (0..numbarvar).map(|j| self.task.get_dim_barvar_j(j as i32).unwrap() as usize).collect();
        let mut accptr    = (0usize..1usize).iter().join((0..numacc).iter()
                                                         .map(|i| self.task.get_acc_n(i as i64).unwrap() as usize)
                                                         .fold_map(0,|&p,&n| n+p)).collect();
        let mut barvarptr = (0usize..1usize).iter().join((0..numbarvar).iter()
                                                         .map(|j| self.task.get_len_barvar_j(j as i32).unwrap() as usize)
                                                         .fold_map(0,|&p,&n| n+p)).collect();
        // let mut accptr    = (0usize..1usize).iter().join((0..numacc).iter().map(|i| self.task.get_acc_n(i as i64).unwrap() as usize).
        // }
        //     vec![0usize; numacc+1]; accptr[1..].iter_mut().enumerate().for_each(|(i,p)| *p =  self.task.get_acc_n(i as i64).unwrap() as usize);
        // let mut barvarptr = vec![0usize; numbarvar+1]; barvarptr[1..].iter_mut().enumerate().for_each(|(j,p)| *p =  self.task.get_len_barvar_j(j as i32).unwrap() as usize);

        // extract solutions
        [(mosek::Soltype::BAS,& mut self.sol_bas),
         (mosek::Soltype::ITR,& mut self.sol_itr),
         (mosek::Soltype::ITG,& mut self.sol_itg)].iter().for_each(|&(whichsol,sol)| {
             if ! self.task.solution_def(whichsol).unwrap() {
                 sol.primal.status = SolutionStatus::Undefined;
                 sol.dual.status   = SolutionStatus::Undefined;
             }
             else {
                 let (psta,dsta) = split_sol_sta(self.task.get_sol_sta(whichsol).unwrap());
                 sol.primal.status = psta;
                 sol.dual.status   = dsta;

                 if let SolutionStatus::Undefined = psta {}
                 else {
                     sol.primal.resize(self.vars.len(),self.cons.len());
                     self.task.get_xx(whichsol,xx.as_mut_slice()).unwrap();
                     self.task.get_xc(whichsol,xc.as_mut_slice()).unwrap();
                     if numbarvar > 0 { self.task.get_barx_slice(whichsol,0,numbarvar as i32,barx.len() as i64,barx.as_mut_slice()).unwrap(); }
                     if numacc > 0 { self.task.evaluate_accs(whichsol,accx.as_mut_slice()).unwrap(); }

                     self.vars.iter().zip(sol.primal.var.iter_mut()).for_each(|(&v,r)| {
                         *r = match v {
                             VarAtom::Linear(j) => xx[j as usize],
                             VarAtom::BarElm(j,ofs) => barx[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                             VarAtom::ConicElm(j,_coni) => xx[j as usize]
                         };
                     });
                     self.cons.iter().zip(sol.primal.con.iter_mut()).for_each(|(&v,r)| {
                         *r = match v {
                             ConAtom::ConicElm(acci,ofs) => accx[accptr[acci as usize]+ofs]
                         };
                     });
                 }

                 if let SolutionStatus::Undefined = dsta {}
                 else {
                     sol.dual.resize(self.vars.len(),self.cons.len());
                     self.task.get_slx(whichsol,slx.as_mut_slice()).unwrap();
                     self.task.get_sux(whichsol,sux.as_mut_slice()).unwrap();
                     self.task.get_slc(whichsol,slc.as_mut_slice()).unwrap();
                     self.task.get_suc(whichsol,suc.as_mut_slice()).unwrap();
                     self.task.get_y(whichsol,y.as_mut_slice()).unwrap();
                     if numbarvar > 0 { self.task.get_bars_slice(whichsol,0,numbarvar as i32,bars.len() as i64,bars.as_mut_slice()).unwrap(); }
                     if numacc > 0 { self.task.get_acc_dot_y_s(whichsol,doty.as_mut_slice()).unwrap(); }

                     self.vars.iter().zip(sol.dual.var.iter_mut()).for_each(|(&v,r)| {
                         *r = match v {
                             VarAtom::Linear(j) => slc[j as usize] - suc[j as usize],
                             VarAtom::BarElm(j,ofs) => bars[barvarptr[j as usize]+row_major_offset_to_col_major(ofs,dimbarvar[j as usize])],
                             VarAtom::ConicElm(_j,coni) => {
                                 match self.cons[coni] {
                                     ConAtom::ConicElm(acci,ofs) => doty[accptr[acci as usize]+ofs]
                                 }
                             }
                         };
                     });
                     self.cons.iter().zip(sol.dual.con.iter_mut()).for_each(|(&v,r)| {
                         *r = match v {
                             ConAtom::ConicElm(acci,ofs) => doty[accptr[acci as usize]+ofs]
                         };
                     });
                 }
             }
        })
    }
    ////////////////////////////////////////////////////////////
    // Solutions

    fn select_sol(&self, solid : SolutionType) -> Option<&Solution> {
        match solid {
            SolutionType::Basic    => Some(&self.sol_bas),
            SolutionType::Interior => Some(&self.sol_itr),
            SolutionType::Integer  => Some(&self.sol_itg),
            SolutionType::Default  => {
                match self.sol_itg.primal.status {
                    SolutionStatus::Undefined =>
                        match (self.sol_bas.primal.status,self.sol_bas.dual.status) {
                            (SolutionStatus::Undefined,SolutionStatus::Undefined) =>
                                match (self.sol_itr.primal.status,self.sol_itr.dual.status) {
                                    (SolutionStatus::Undefined,SolutionStatus::Undefined) => None,
                                    _ => Some(&self.sol_itr)
                                }
                            _ => Some(&self.sol_bas)
                        },
                    _ => Some(& self.sol_itg)
                }
            }
        }
    }
    fn primal_var_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.primal.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.primal.var.len() { panic!("Variable indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.var.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }

    fn dual_var_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.dual.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.dual.var.len() { panic!("Variable indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.dual.var.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }
    fn primal_con_solution(&self, solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.primal.status {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.primal.con.len() { panic!("Constraint indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.con.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }
    fn dual_con_solution(&self,   solid : SolutionType, idxs : &[usize], res : & mut [f64]) -> Result<(),String> {
        if let Some(sol) = self.select_sol(solid) {
            if let SolutionStatus::Undefined = sol.dual.status  {
                Err("Solution part is not defined".to_string())
            }
            else {
                if let Some(&v) = idxs.iter().max() { if v >= sol.dual.con.len() { panic!("Constraint indexes are outside of range") } }
                res.iter_mut().zip(idxs.iter()).for_each(|(r,&i)| *r = unsafe { *sol.primal.con.get_unchecked(i) });
                Ok(())
            }
        }
        else {
            Err("Solution value is undefined".to_string())
        }
    }

    pub fn primal_solution<I:ModelItem>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.primal(self,solid) }
    pub fn dual_solution<I:ModelItem>(&self, solid : SolutionType, item : &I) -> Result<Vec<f64>,String> { item.dual(self,solid) }
    pub fn primal_solution_into<I:ModelItem>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }
    pub fn dual_solution_into<I:ModelItem>(&self, solid : SolutionType, item : &I, res : &mut[f64]) -> Result<usize,String> { item.primal_into(self,solid,res) }
}


/// Split an evaluated expression into linear and semidefinite parts
fn split_expr(eptr    : &[usize],
              esubj   : &[usize],
              ecof    : &[f64],
              vars    : &[VarAtom]) -> (Vec<i32>, //subj
                                        Vec<f64>, //cof
                                        Vec<i64>, //ptr
                                        Vec<f64>, //fix
                                        Vec<i64>, //barsubi
                                        Vec<i32>, //barsubj
                                        Vec<i32>, //barsubk
                                        Vec<i32>, //barsubl
                                        Vec<f64>) { //barcof
    let nnz = *eptr.last().unwrap();
    let nelm = eptr.len()-1;
    let nlinnz = esubj.iter().filter(|&&j| if let VarAtom::BarElm(_,_) = unsafe { *vars.get_unchecked(j) } { false } else { true } ).count();
    let npsdnz = nnz - nlinnz;

    let mut subj : Vec<i32> = Vec::with_capacity(nlinnz);
    let mut cof  : Vec<f64> = Vec::with_capacity(nlinnz);
    let mut ptr  : Vec<i64> = Vec::with_capacity(nelm+1);
    let mut fix  : Vec<f64> = Vec::with_capacity(nelm);
    let mut barsubi : Vec<i64> = Vec::with_capacity(npsdnz);
    let mut barsubj : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barsubk : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barsubl : Vec<i32> = Vec::with_capacity(npsdnz);
    let mut barcof  : Vec<f64> = Vec::with_capacity(npsdnz);

    ptr.push(0);
    eptr[..nelm].iter().zip(eptr[1..].iter()).enumerate().for_each(|(i,(&p0,&p1))| {
        let mut cfix = 0.0;
        esubj[p0..p1].iter().zip(ecof[p0..p1].iter()).for_each(|(&idx,&c)| {
            if idx == 0 {
                cfix += c;
            }
            else {
                match *unsafe{ vars.get_unchecked(idx) } {
                    VarAtom::Linear(j) => {
                        subj.push(j);
                        cof.push(c);
                    },
                    VarAtom::ConicElm(j,_coni) => {
                        subj.push(j);
                        cof.push(c);
                    },
                    VarAtom::BarElm(j,ofs) => {
                        let (k,l) = row_major_offset_to_ij(ofs);
                        barsubi.push(i as i64);
                        barsubj.push(j);
                        barsubk.push(k as i32);
                        barsubl.push(l as i32);
                        barcof.push(c);
                    }
                }
            }
        });
        ptr.push(subj.len() as i64);
        fix.push(cfix);
    });

    (subj,
     cof,
     ptr,
     fix,
     barsubi,
     barsubj,
     barsubk,
     barsubl,
     barcof)
}

fn split_sol_sta(solsta : i32) -> (SolutionStatus,SolutionStatus) {
    match solsta {
        mosek::Solsta::UNKNOWN => (SolutionStatus::Unknown,SolutionStatus::Unknown),
        mosek::Solsta::OPTIMAL => (SolutionStatus::Optimal,SolutionStatus::Optimal),
        mosek::Solsta::PRIM_FEAS => (SolutionStatus::Feasible,SolutionStatus::Unknown),
        mosek::Solsta::DUAL_FEAS => (SolutionStatus::Feasible,SolutionStatus::Unknown),
        mosek::Solsta::PRIM_AND_DUAL_FEAS => (SolutionStatus::Unknown,SolutionStatus::Feasible),
        mosek::Solsta::PRIM_INFEAS_CER => (SolutionStatus::Undefined,SolutionStatus::CertInfeas),
        mosek::Solsta::DUAL_INFEAS_CER => (SolutionStatus::CertInfeas,SolutionStatus::Undefined),
        mosek::Solsta::PRIM_ILLPOSED_CER => (SolutionStatus::Undefined,SolutionStatus::CertIllposed),
        mosek::Solsta::DUAL_ILLPOSED_CER => (SolutionStatus::CertIllposed,SolutionStatus::Undefined),
        mosek::Solsta::INTEGER_OPTIMAL => (SolutionStatus::Optimal,SolutionStatus::Undefined),
        _ => (SolutionStatus::Unknown,SolutionStatus::Unknown)
    }
}

/// Convert linear row-major order offset into a lower triangular
/// matrix to an (i,j) pair.
fn row_major_offset_to_ij(ofs : usize) -> (usize,usize) {
    let i = (((1.0+8.0*ofs as f64).sqrt()-1.0)/2.0).floor() as usize;
    let j = ofs - i*(i+1)/2;
    (i,j)
}
/// Convert linear column-major order offset into a lower triangular
/// matrix and a dimension to a (i,j) pair`
fn row_major_offset_to_col_major(ofs : usize, dim : usize) -> usize {
    let (i,j) = row_major_offset_to_ij(ofs);
    ((2*dim-1)*j - j*j)/2 + i
}
    





    
/////////////////////////////////////////////////////////////////////
// TEST


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
