//! Backend for `mosekcomodel` for [HIGHS](https://highs.dev/) solver. The implementation is still
//! fairly basic.
//!
//! # Example lo1
//!
//! ```math
//! maximize 3*x0 + 1*x1 + 5*x2 + x3
//! such that              
//!          3*x0 + 1*x1 + 2*x2        = 30,
//!          2*x0 + 1*x1 + 3*x2 + 1*x3 > 15,
//!                 2*x1 +      + 3*x3 < 25
//! and
//!          0 < x1 < 10
//!          x0,x2,x3 > 0,
//! ```
//! Can be implemented as:
//! ```
//! extern crate mosekcomodel;
//! 
//! use mosekcomodel::*;
//! use mosekcomodel_highs::Model;
//! 
//! fn lo1() -> (SolutionStatus,SolutionStatus,Result<Vec<f64>,String>) {
//!     let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
//!     let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
//!     let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
//!     let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];
//! 
//!     // Create a model with the name 'lo1'
//!     let mut m = Model::new(Some("lo1"));
//!     // Create variable 'x' of length 4
//!     let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));
//! 
//!     // Create constraints
//!     let _ = m.constraint(None, x.index(1), less_than(10.0));
//!     let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
//!     let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
//!     let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));
//! 
//!     // Set the objective function to (c^t * x)
//!     m.objective(Some("obj"), Sense::Maximize, x.dot(c));
//! 
//!     // Solve the problem
//!     //m.write_problem("lo1-nosol.ptf");
//!     m.solve();
//! 
//!     // Get the solution values
//!     let (psta,dsta) = m.solution_status(SolutionType::Default);
//!     let xx = m.primal_solution(SolutionType::Default,&x);
//! 
//!     (psta,dsta,m.primal_solution(SolutionType::Default,&x))
//! }
//! 
//! fn main() {
//!     let (psta,dsta,xx) = lo1();
//!     println!("Status = {:?}/{:?}",psta,dsta);
//!     println!("x = {:?}", xx);
//! }
//! ```
//!
//! <script type="text/javascript" id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"> </script>
extern crate mosekcomodel;
extern crate highs;

use mosekcomodel::*;
use mosekcomodel::utils::iter::ChunksByIterExt;
use mosekcomodel::utils::*;
use std::path::Path;
use itertools::izip;

pub type Model = ModelAPI<ModelHighs>;

#[derive(Clone,Copy)]
enum Item {
    Linear{index:usize},
    RangedUpper{index:usize},
    RangedLower{index:usize},
}
impl Item {
    fn index(&self) -> usize { 
        match self {
            Item::Linear { index }      => *index,
            Item::RangedUpper { index } => *index,
            Item::RangedLower { index } => *index
        }
    } 
}
/// Simple model object.
#[derive(Default)]
#[allow(unused)]
pub struct ModelHighs {
    name : Option<String>,

    var_range_lb  : Vec<f64>,
    var_range_ub  : Vec<f64>,
    var_range_int : Vec<bool>,

    vars          : Vec<Item>,

    a_ptr         : Vec<[usize;2]>,
    a_subj        : Vec<usize>,
    a_cof         : Vec<f64>,
    con_lb        : Vec<f64>,
    con_ub        : Vec<f64>,

    con_a_row     : Vec<usize>, // index into a_ptr
    cons          : Vec<Item>,

    sense_max     : bool,
    c_subj        : Vec<usize>,
    c_cof         : Vec<f64>,
}

impl BaseModelTrait for ModelHighs {
    fn new(name : Option<&str>) -> Self {
        ModelHighs {
            name         : name.map(|v| v.to_string()),
            var_range_lb : vec![1.0],
            var_range_ub : vec![1.0],
            var_range_int : vec![false],
            vars : vec![Item::Linear { index: 0 }],
            .. Default::default()
        }
    }
    fn free_variable<const N : usize>
        (&mut self,
         _name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();
        let first = self.var_range_lb.len();
        let last  = first + n;

        self.var_range_lb.resize(last,f64::NEG_INFINITY);
        self.var_range_ub.resize(last,f64::INFINITY);
        self.var_range_int.resize(last,false);

        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last {
            self.vars.push(Item::Linear{index:i});
        }

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), None, shape))
    }

    fn linear_variable<const N : usize,R>
        (&mut self, 
         _name : Option<&str>,
         dom  : LinearDomain<N>) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result,String>    
        where 
            Self : Sized
    {
        let (dt,b,sp,shape,is_integer) = dom.dissolve();
        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_range_lb.len();
        let last  = first + n;


        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last { self.vars.push(Item::Linear{index:i}) }
        match dt {
            LinearDomainType::Zero => {
                self.var_range_lb.resize(last,0.0);
                self.var_range_ub.resize(last,0.0);
            },
            LinearDomainType::Free => {
                self.var_range_lb.resize(last,f64::NEG_INFINITY);
                self.var_range_ub.resize(last,f64::INFINITY);
            },
            LinearDomainType::NonNegative => {
                self.var_range_lb.resize(last,0.0);
                self.var_range_lb[first..last].copy_from_slice(b.as_slice());
                self.var_range_ub.resize(last,f64::INFINITY);
            },
            LinearDomainType::NonPositive => {
                self.var_range_lb.resize(last,f64::NEG_INFINITY);
                self.var_range_ub.resize(last,0.0);
                self.var_range_ub[first..last].copy_from_slice(b.as_slice());
            },
        }
        self.var_range_int.resize(last,is_integer);

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, _name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (shape,bl,bu,sp,is_integer) = dom.dissolve();

        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_range_lb.len();
        let last  = first + n;

        let ptr0 = self.vars.len();
        let ptr1 = self.vars.len()+n;
        let ptr2 = self.vars.len()+2*n;
        self.vars.reserve(n*2);
        for i in first..last { self.vars.push(Item::RangedLower{index:i}) }
        for i in first..last { self.vars.push(Item::RangedUpper{index:i}) }
        self.var_range_lb.resize(last,0.0);
        self.var_range_ub.resize(last,0.0);
        self.var_range_int.resize(last,is_integer);

        self.var_range_lb[ptr0..ptr1].copy_from_slice(bl.as_slice());
        self.var_range_ub[ptr1..ptr2].copy_from_slice(bu.as_slice());

        Ok((Variable::new((ptr0..ptr1).collect::<Vec<usize>>(), sp.clone(), &shape),
            Variable::new((ptr1..ptr2).collect::<Vec<usize>>(), sp, &shape)))
    }

    fn linear_constraint<const N : usize>
        (& mut self, 
         _name  : Option<&str>,
         dom   : LinearDomain<N>,
         _eshape : &[usize], 
         ptr   : &[usize], 
         subj  : &[usize], 
         cof   : &[f64]) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
    {
        let (dt,b,_sp,shape,_is_integer) = dom.dissolve();

        assert_eq!(b.len(),ptr.len()-1); 
        let nrow = b.len();

        let a_row0 = self.a_ptr.len();
        let con_row0 = self.con_a_row.len();
        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        {
            for (b,n) in ptr.iter().zip(ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
                self.a_ptr.push([b,n]);
            }
        }

        let con0 = self.cons.len();
        self.a_subj.extend_from_slice(subj);
        self.a_cof.extend_from_slice(cof);
        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }
        self.cons.reserve(n); for i in con_row0..con_row0+n { self.cons.push(Item::Linear { index: i }) }
        
        match dt {
            LinearDomainType::Zero => {
                self.con_lb.extend_from_slice(b.as_slice());
                self.con_ub.extend_from_slice(b.as_slice());
            },
            LinearDomainType::Free => { 
                self.con_lb.resize(con_row0+nrow,f64::NEG_INFINITY);
                self.con_ub.resize(con_row0+nrow,f64::INFINITY);
            },
            LinearDomainType::NonNegative => {
                self.con_lb.extend_from_slice(b.as_slice());
                self.con_ub.resize(con_row0+nrow,f64::INFINITY);
            },
            LinearDomainType::NonPositive => {
                self.con_lb.resize(con_row0+nrow,f64::NEG_INFINITY);
                self.con_ub.extend_from_slice(b.as_slice());
            },
        }

        Ok(Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape))
    }

    fn ranged_constraint<const N : usize>
        (& mut self, 
         _name : Option<&str>, 
         dom  : LinearRangeDomain<N>,
         _eshape : &[usize], 
         ptr : &[usize], 
         subj : &[usize], 
         cof : &[f64]) -> Result<<LinearRangeDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
    {
        let (shape,bl,bu,_,_) = dom.dissolve();

        let a_row0 = self.a_ptr.len()-1;
        let con_row0 = self.con_a_row.len();

        let n = shape.iter().product::<usize>();
        
        self.a_ptr.reserve(n);
        for (b,n) in izip!(ptr.iter(),ptr[1..].iter()).scan(self.a_subj.len(),|p,(&p0,&p1)| { let (b,n) = (*p,p1-p0); *p += n; Some((b,n)) }) {
            self.a_ptr.push([b,n]);
        }

        self.a_subj.extend_from_slice(subj);
        self.a_cof.extend_from_slice(cof);
        self.con_lb.extend_from_slice(bl.as_slice());
        self.con_ub.extend_from_slice(bu.as_slice());

        self.con_a_row.reserve(n); for i in a_row0..a_row0+n { self.con_a_row.push(i); }

        let con0 = self.cons.len();
        self.cons.reserve(n*2);
        for i in con_row0..con_row0+n { self.cons.push(Item::RangedLower { index: i }); }
        for i in con_row0..con_row0+n { self.cons.push(Item::RangedUpper { index: i }); }

        Ok((Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape),
            Constraint::new((con0+n..con0+2*n).collect::<Vec<usize>>(), &shape)))
    }

    fn update(& mut self, idxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>
    {
        if shape.iter().product::<usize>() != idxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = idxs.iter().max() {
            if i >= self.cons.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

        for (subj,cof,i) in izip!(subj.chunks_ptr(ptr),cof.chunks_ptr(ptr),idxs.iter().map(|&i| self.cons[i].index())) {
            let n = subj.len();

            let ai = self.con_a_row[i];

            let entry = self.a_ptr[ai];
            if entry[1] >= n {
                self.a_subj[entry[0]..entry[0]+n].copy_from_slice(subj);
                self.a_cof[entry[0]..entry[0]+n].copy_from_slice(cof);
                self.a_ptr[i][1] = n;
            }
            else {
                self.a_ptr[ai][1] = 0;
                let lb = self.con_lb[ai];
                let ub = self.con_ub[ai];
                self.con_a_row[i] = self.a_ptr.len();
                self.con_lb.push(lb);
                self.con_ub.push(ub);
                self.a_ptr.push([self.a_subj.len(),n]);
                self.a_subj.extend_from_slice(subj);
                self.a_cof.extend_from_slice(cof);
            }
        }
        Ok(())
    }

    fn write_problem<P>(&self, _filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        Err("Writing problem not supported".to_string())
    }

    /// NOTE: Highs apepars to support ranged constraints and variables, and dual solution values,
    /// however, it is not possible to directly get the dual values for the individual bounds. 
    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution) -> Result<(),String>
    {
        let mut p = highs::RowProblem::default();

        let mut c = vec![0.0; self.var_range_lb.len()];
        c.permute_by_mut(self.c_subj.as_slice()).zip(self.c_cof.iter()).for_each(|(t,&v)| *t = v);
        
        let isint = self.var_range_int.iter().any(|&v| v);
            
        let cols : Vec<highs::Col> = 
            izip!(c.iter(), self.var_range_lb.iter(), self.var_range_ub.iter(),self.var_range_int.iter())
                .map(|(&cj,&bl,&bu,&isint)| {
                    //println!("Variable: c_j = {}, bl = {}, bu = {}",cj,bl,bu);
                    match (bl > f64::NEG_INFINITY,bu < f64::INFINITY) {
                        (true,true)   => if ! isint { p.add_column(cj, bl..bu) } else { p.add_integer_column(cj, bl..bu) },
                        (true,false)  => if ! isint { p.add_column(cj, bl..) }   else { p.add_integer_column(cj, bl..bu) },
                        (false,true)  => if ! isint { p.add_column(cj, ..bu) }   else { p.add_integer_column(cj, bl..bu) },
                        (false,false) => if ! isint { p.add_column::<f64,std::ops::RangeFull>(cj, ..) } else { p.add_integer_column(cj, bl..bu) },
                    }})
                .collect();
        
        let ptrb : Vec<usize> = self.a_ptr.iter().map(|v| v[0]).collect();
        let ptre : Vec<usize> = self.a_ptr.iter().map(|v| v[0]+v[1]).collect();
        izip!(self.con_lb.iter(),
              self.con_ub.iter(),
              self.a_subj.chunks_ptr2(ptrb.as_slice(),ptre.as_slice()),
              self.a_cof.chunks_ptr2(ptrb.as_slice(),ptre.as_slice()))
            .for_each(|(&bl,&bu,subj,cof)|
                { 
                    let expr : Vec<(highs::Col,f64)> = cols.permute_by(subj).cloned().zip(cof.iter().cloned()).collect();
                    //println!("Constraint: bl = {}, bu = {}, expr = {:?}",bl,bu,expr);
                    match (bl > f64::NEG_INFINITY,bu < f64::INFINITY) {
                        (true,true)   => p.add_row(bl..bu, expr),
                        (true,false)  => p.add_row(bl..,   expr),
                        (false,true)  => p.add_row(..bu,   expr),
                        (false,false) => p.add_row::<f64,std::ops::RangeFull,_,_>(..,expr),
                    }
                });

        let m = p.optimise(if self.sense_max { highs::Sense::Maximise } else { highs::Sense::Minimise });

        let sm = m.solve();
         

        sol_bas.primal.status = SolutionStatus::Undefined;
        sol_bas.dual.status   = SolutionStatus::Undefined;
        sol_itr.primal.status = SolutionStatus::Undefined;
        sol_itr.dual.status   = SolutionStatus::Undefined;
        sol_itg.primal.status = SolutionStatus::Undefined;
        sol_itg.dual.status   = SolutionStatus::Undefined;

        if let highs::HighsModelStatus::Optimal = sm.status() {
            let sol = sm.get_solution();
            let pobj = c.iter().zip(sol.columns().iter()).map(|(a,b)| a*b).sum();
        
            if isint {
                sol_itg.resize(self.vars.len(),self.cons.len());
                sol_itg.primal.status = SolutionStatus::Optimal;

                sol_itg.primal.obj = pobj;

                let xc = sol.rows();
                let xx = sol.columns();

                for (item,xres) in self.vars.iter().zip(sol_itg.primal.var.iter_mut()) {
                    match item {
                        Item::Linear { index }      => *xres = xx[*index],
                        Item::RangedUpper { index } => *xres = xx[*index],
                        Item::RangedLower { index } => *xres = xx[*index],
                    }
                }
                for (item,xres) in self.cons.iter().zip(sol_itg.primal.con.iter_mut()) {
                    match item {
                        Item::Linear { index }      => *xres = xc[*index],
                        Item::RangedUpper { index } => *xres = xc[*index],
                        Item::RangedLower { index } => *xres = xc[*index],
                    }
                }
            }
            else {
                //TODO Compute the dual objective
                sol_bas.dual.obj = 0.0;
                sol_bas.primal.obj = pobj;

                sol_bas.resize(self.vars.len(),self.cons.len());
                sol_bas.primal.status = SolutionStatus::Optimal;
                sol_bas.dual.status = SolutionStatus::Optimal;
                let xc = sol.rows();
                let xx = sol.columns();

                let sc = sol.dual_rows();
                let sx = sol.dual_columns();

                sol_bas.dual.var.iter_mut().for_each(|v| *v = 0.0);
                sol_bas.dual.con.iter_mut().for_each(|v| *v = 0.0);

                for (item,xres,sres,&bl) in izip!(self.vars.iter(),sol_bas.primal.var.iter_mut(),sol_bas.dual.var.iter_mut(),self.var_range_lb.iter()) {
                    match item {
                        Item::Linear { index }      => { *xres = xx[*index]; *sres = sx[*index]; },
                        Item::RangedLower { index } => { 
                            *xres = xx[*index]; 
                            if *xres < bl + 1e-7 { *sres = sx[*index]; } // at lower bound
                        },
                        Item::RangedUpper { index } => { 
                            *xres = xx[*index]; 
                            if *xres >= bl + 1e-7 { *sres = sx[*index]; } // not at lower bound
                        },
                    }
                }
                for (item,xres,sres,&bl) in izip!(self.cons.iter(),sol_bas.primal.con.iter_mut(),sol_bas.dual.con.iter_mut(),self.con_lb.iter()) {
                    match item {
                        Item::Linear { index }      => { *xres = xc[*index]; *sres = sc[*index]; },
                        Item::RangedLower { index } => { 
                            *xres = xc[*index];
                            if *xres < bl + 1e-7 { *sres = sc[*index]; }
                        },
                        Item::RangedUpper { index } => { 
                            *xres = xc[*index]; 
                            if *xres >= bl + 1e-7 { *sres = sc[*index]; }
                        },
                    }
                }
           }
        }

        Ok(())
    }

    fn objective(&mut self, _name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.sense_max = match sense { Sense::Maximize => true, Sense::Minimize => false };
        self.c_subj.resize(subj.len(),0); self.c_subj.copy_from_slice(subj);
        self.c_cof.resize(cof.len(),0.0); self.c_cof.copy_from_slice(cof);
        Ok(())
    }

    fn set_parameter<V>(&mut self, _parname : V::Key, _parval : V) -> Result<(),String> where V : SolverParameterValue<Self>,Self: Sized
    {
        Err("Parameters not supported".to_string())
    }
}

