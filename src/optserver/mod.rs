//! This module implements a backend that can offload to a mosek OptServer. 
//!
//! It works by 
use crate::*;
use crate::model::{DJCDomainTrait, DJCModelTrait, ModelWithIntSolutionCallback, ModelWithLogCallback, PSDModelTrait, VectorConeModelTrait};
use crate::domain::*;
use crate::utils::iter::{ChunksByIterExt, PermuteByEx, PermuteByMutEx};
use itertools::{iproduct, izip};
use json::JSON;
use std::fs::{write, File};
use std::net::TcpStream;
use std::path::Path;

mod http;
mod json;

pub type Model = ModelAPI<Backend>;

#[derive(Clone,Copy)]
enum Item {
    Linear{index:usize},
    RangedUpper{index:usize},
    RangedLower{index:usize},
}

impl Item {
    fn index(&self) -> usize { 
        match self {
            Item::Linear { index } => *index,
            Item::RangedUpper { index } => *index,
            Item::RangedLower { index } => *index
        }
    } 
}

#[derive(Clone,Copy)]
struct Element {
    lb:f64,ub:f64
}

/// Simple model object that supports input of linear, conic and disjunctive constraints. It only
/// stores data, it does not support solving or writing problems.
#[derive(Default)]
pub struct Backend {
    name : Option<String>,

    var_elt       : Vec<Element>, // Either lb,ub,int or index,coneidx,offset
    var_int       : Vec<bool>,

    vars          : Vec<Item>,
    var_names     : Vec<Option<String>>,

    a_ptr         : Vec<[usize;2]>,
    a_subj        : Vec<usize>,
    a_cof         : Vec<f64>,

    con_elt       : Vec<Element>,
    con_a_row     : Vec<usize>, // index into a_ptr
    cons          : Vec<Item>,
    con_names     : Vec<Option<String>>,


    sense_max     : bool,
    c_subj        : Vec<usize>,
    c_cof         : Vec<f64>,

    address       : String
}

impl BaseModelTrait for Backend {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name : name.map(|v| v.to_string()),
            var_elt    : Default::default(), 
            var_int    : Default::default(), 

            vars       : Default::default(), 
            var_names  : Default::default(),

            a_ptr      : Default::default(), 
            a_subj     : Default::default(), 
            a_cof      : Default::default(), 

            con_elt    : Default::default(), 
            con_a_row  : Default::default(), 
            cons       : Default::default(), 
            con_names  : Default::default(),

            sense_max  : Default::default(), 
            c_subj     : Default::default(), 
            c_cof      : Default::default(), 

            address    : Default::default(), 
        }
    }
    fn free_variable<const N : usize>
        (&mut self,
         name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();
        let first = self.var_elt.len();
        let last  = first + n;

        self.var_elt.resize(last,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY });
        self.var_int.resize(last,false);
        self.var_names.resize(last,None);

        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last {
            self.vars.push(Item::Linear{index:i});
        }

        if let Some(name) = name {
            (0..n).scan([0usize;N],|i,_| { 
                let r = format!("{}{:?}",name,i); 
                i.iter_mut().zip(shape.iter()).rev().fold(1,|c,(v,&d)| { *v += c; if *v >= d { *v = 0; 1 } else { 0 }}); 
                Some(r)
            })
                .zip(self.var_names[first..last].iter_mut())
                .for_each(|(n,vn)| *vn = Some(n));
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
        let first = self.var_int.len();
        let last  = first + n;


        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last { self.vars.push(Item::Linear{index:i}) }
        match dt {
            LinearDomainType::Zero => {
                self.var_elt.resize(last,Element{ lb: 0.0, ub: 0.0 });
            },
            LinearDomainType::Free => {
                self.var_elt.resize(last,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY});
            },
            LinearDomainType::NonNegative => {
                self.var_elt.reserve(last);
                for lb in b {
                    self.var_elt.push(Element{ lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.var_elt.reserve(last);
                for ub in b {
                    self.var_elt.push(Element{ lb : f64::NEG_INFINITY, ub });
                }
            },
        }
        self.var_int.resize(last,is_integer);

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, _name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (shape,bl,bu,sp,is_integer) = dom.dissolve();

        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());
        let first = self.var_int.len();
        let last  = first + n;

        let ptr0 = self.vars.len();
        let ptr1 = self.vars.len()+n;
        let ptr2 = self.vars.len()+2*n;
        self.vars.reserve(n*2);
        for i in first..last { self.vars.push(Item::RangedLower{index:i}) }
        for i in first..last { self.vars.push(Item::RangedUpper{index:i}) }

        self.var_elt.reserve(last);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.var_elt.push(Element{ lb, ub })
        }
        self.var_int.resize(last,is_integer);

        Ok((Variable::new((ptr0..ptr1).collect::<Vec<usize>>(), sp.clone(), &shape),
            Variable::new((ptr1..ptr2).collect::<Vec<usize>>(), sp, &shape)))
    }

    fn linear_constraint<const N : usize>
        (& mut self, 
         name  : Option<&str>,
         dom   : LinearDomain<N>,
         _eshape : &[usize], 
         ptr   : &[usize], 
         subj  : &[usize], 
         cof   : &[f64]) -> Result<<LinearDomain<N> as ConstraintDomain<N,Self>>::Result,String> 
    {
        let (dt,b,sp,shape,_is_integer) = dom.dissolve();

        if let Some(sp) = &sp {
            assert_eq!(b.len(),sp.len()); 
        }
        else {
            assert_eq!(b.len(),ptr.len()-1); 
        }
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
                self.con_elt.reserve(con_row0+nrow);
                for b in b {
                    self.con_elt.push(Element{ lb: b, ub: b });
                }
            },
            LinearDomainType::Free => { 
                self.con_elt.resize(con_row0+nrow,Element{ lb: f64::NEG_INFINITY, ub: f64::INFINITY });
            },
            LinearDomainType::NonNegative => {
                self.con_elt.reserve(con_row0+nrow);
                for lb in b {
                    self.con_elt.push(Element{ lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.con_elt.reserve(con_row0+nrow);
                for ub in b {
                    self.con_elt.push(Element{ lb : f64::NEG_INFINITY, ub });
                }
            },
        }

        Ok(Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape))
    }

    fn ranged_constraint<const N : usize>
        (& mut self, 
         name : Option<&str>, 
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
        self.con_elt.reserve(con_row0+n);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.con_elt.push(Element{ lb, ub });
        }

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
                self.con_elt.push(self.con_elt[ai]);
                self.a_ptr[ai][1] = 0;
                self.con_a_row[i] = self.a_ptr.len();
                self.a_ptr.push([self.a_subj.len(),n]);
                self.a_subj.extend_from_slice(subj);
                self.a_cof.extend_from_slice(cof);
            }
        }
        Ok(())
    }

    fn write_problem<P>(&self, filename : P) -> Result<(),String> where P : AsRef<Path>
    {
        let p = filename.as_ref();
        if let Some(ext) = p.extension().and_then(|ext| ext.to_str()) {
            match ext {
                "json"|"jtask" => {
                    let mut f = File::create(p).map_err(|e| e.to_string())?;
                    self.format_json_to(&mut f).map_err(|e| e.to_string())
                },
                _ => Err("Writing problem not supported".to_string())
            }
        }
        else {
            Err("Writing problem not supported".to_string())
        }
    }

    fn solve(& mut self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution) -> Result<(),String>
    {
        use http::*;
        let mut con = TcpStream::connect(self.address.as_str()).map_err(|e| e.to_string())?;


        let mut resp = Request::post("/api/v1/sbumit+solve")
            .add_header("Content-Type", "application/x-mosek-jtask")
            .add_header("Accept", "application/x-mosek-jtask")
            .submit_with_writer(&mut con,|w| self.format_json_to(w).map_err(|e| e.to_string()))?;
      
        if resp.code() != 200 {
            return Err(format!("OptServer responded with code {}",resp.code()))
        }

        let mut rescode = None;

        for (k,v) in resp.headers() {
            if k.eq_ignore_ascii_case(b"Content-Type") {
                if v.eq(b"application/json") {
                    // ok
                }
                else {
                    return Err(format!("Unsupported solution format: {}",std::str::from_utf8(v).unwrap_or("<invalid utf-8>")));
                }
            }
            else if k.eq_ignore_ascii_case(b"X-Mosek-Res") {
                if let Ok(r) = std::str::from_utf8(v) {
                    rescode = Some(r.to_string());
                }
            }
        }

        let data = JSON::read(& mut resp).map_err(|e| e.to_string())?;
        if let JSON::Dict(d) = data {
            if let Some((k,v)) = d.0.iter().find(|kv| kv.0.as_str() == "Task/solutions") {
                if let JSON::Dict(d) = v {
                    for (k,v) in d.0.iter() {
                        let sol =
                            match k.as_str() {   
                                "basic" => sol_bas,
                                "interior" => sol_itr,
                                "integer" => sol_itg,
                                _ => break
                            };

                        if let JSON::Dict(d) = v {
                            let mut xx  = Vec::new();
                            let mut slx = Vec::new();
                            let mut sux = Vec::new();
                            let mut xc  = Vec::new();
                            let mut slc = Vec::new();
                            let mut suc = Vec::new();
                            for (k,v) in d.0.iter() {
                                match (k.as_str(),v) {
                                    ("solsta",JSON::String(v)) => {
                                        match v.as_str() {
                                            "unknown" => { sol.primal.status = SolutionStatus::Unknown; sol.dual.status = SolutionStatus::Unknown; }
                                            "optimal" => { sol.primal.status = SolutionStatus::Optimal; sol.dual.status = SolutionStatus::Optimal; }
                                            "integer_optimal" => { sol.primal.status = SolutionStatus::Optimal; sol.dual.status = SolutionStatus::Undefined; }
                                            "prim_and_dual_feas" => { sol.primal.status = SolutionStatus::Feasible; sol.dual.status = SolutionStatus::Feasible; }
                                            "prim_feas" => { sol.primal.status = SolutionStatus::Feasible; sol.dual.status = SolutionStatus::Unknown; }
                                            "dual_feas" => { sol.primal.status = SolutionStatus::Unknown; sol.dual.status = SolutionStatus::Feasible; }
                                            "prim_infeas_cer" =>  { sol.primal.status = SolutionStatus::Undefined; sol.dual.status = SolutionStatus::CertInfeas; }  
                                            "dual_infeas_cer" =>  { sol.primal.status = SolutionStatus::CertInfeas; sol.dual.status = SolutionStatus::Undefined }  
                                            "prim_illposed_cer" =>  { sol.primal.status = SolutionStatus::Undefined; sol.dual.status = SolutionStatus::CertInfeas; }  
                                            "dual_illposed_cer" =>  { sol.primal.status = SolutionStatus::CertInfeas; sol.dual.status = SolutionStatus::Undefined; }  
                                            _ => {}
                                        }
                                    },
                                    ("xx",v)  => if let Ok(val) = v.try_into::<Vec<f64>>() { xx = val; },
                                    ("slx",v) => if let Ok(val) = v.try_into::<Vec<f64>>() { slx = val; },
                                    ("sux",v) => if let Ok(val) = v.try_into::<Vec<f64>>() { sux = val; },
                                    ("xc",v)  => if let Ok(val) = v.try_into::<Vec<f64>>() { xc = val; },
                                    ("slc",v) => if let Ok(val) = v.try_into::<Vec<f64>>() { slc = val; },
                                    ("suc",v) => if let Ok(val) = v.try_into::<Vec<f64>>() { suc = val; },
                                    _ => {}
                                }
                            }
                        }
//prosta: problem status (string).
//solsta: solution status (string).
//xx, xc, y, slc, suc, slx, sux, snx: one for each component of the solution of the same name (list(double)).
//skx, skc, skn: status keys (list(string)).
//doty: the dual
//    solution, grouped by ACC (list(list(double))).
//    barx, bars: the primal/dual semidefinite solution, grouped by matrix variable (list(list(double))).
                    }
                }
            }
        }
        else {
            return Err("Invalid solution format".to_string());
        }

        // interpret data as solution

        Ok(())
    }

    fn objective(&mut self, _name : Option<&str>, sense : Sense, subj : &[usize],cof : &[f64]) -> Result<(),String>
    {
        self.sense_max = match sense { Sense::Maximize => true, Sense::Minimize => false };
        self.c_subj.resize(subj.len(),0); self.c_subj.copy_from_slice(subj);
        self.c_cof.resize(cof.len(),0.0); self.c_cof.copy_from_slice(cof);
        Ok(())
    }
}

pub struct SolverAddress(pub String);

impl SolverParameterValue<Backend> for SolverAddress {
    type Key = ();
    fn set(self,_parname : Self::Key, model : & mut Backend) -> Result<(),String> {
        model.address = self.0;
        Ok(())
    }
}


fn bnd_to_bk(lb : f64, ub : f64) -> &'static str {
    match (lb.is_finite(),ub.is_finite()) {
        (false,false) => "fr",
        (false,true)  => "up",
        (true,false)  => "lo",
        (true,true)   => if lb < ub { "ra" } else { "fx" }
    }
}
impl Backend {
    /// JSON Task format writer.
    ///
    /// See https://docs.mosek.com/latest/capi/json-format.html
    fn format_json_to<S>(&self, strm : &mut S) -> std::io::Result<()> 
        where 
            S : std::io::Write 
    {
        use json::JSON;
        
        let mut doc = json::Dict::new();
        doc.append("$schema",JSON::String("http://mosek.com/json/schema#".to_string()));

        if let Some(name) = &self.name {
            doc.append("Task/name", name.clone());
        }

        doc.append(
            "Task/info",
            json::Dict::from(|taskinfo| {
                taskinfo.append("numvar",self.vars.len() as i64);
                taskinfo.append("numcon",self.con_elt.len() as i64);
            }));
            
        doc.append(
            "Task/data",
            json::Dict::from(|taskdata| {
                taskdata.append("var",json::Dict::from(|d| {
                    d.append("bk",  JSON::List(self.var_elt.iter().map(|e| bnd_to_bk(e.lb,e.ub).into()).collect()));
                    d.append("bl",  JSON::List(self.var_elt.iter().map(|&e| JSON::Float(e.lb)).collect()));
                    d.append("bu",  JSON::List(self.var_elt.iter().map(|&e| JSON::Float(e.ub)).collect()));
                    d.append("type",JSON::List(self.var_int.iter().map(|&e| if e { "int".into() } else { "cont".into() }).collect()));
                }));
                taskdata.append("con",json::Dict::from(|d| {
                    d.append("bk",  JSON::List(self.con_elt.iter().map(|e| bnd_to_bk(e.lb,e.ub).into()).collect()));
                    d.append("bl",  JSON::List(self.con_elt.iter().map(|e| JSON::Float(e.lb)).collect()));
                    d.append("bu",  JSON::List(self.con_elt.iter().map(|e| JSON::Float(e.ub)).collect()));
                }));
                taskdata.append(
                    "objective",
                    json::Dict::from(|d| {
                        d.append("sense", if self.sense_max { "max" } else { "min" });
                        d.append("cfix",0.0f64);
                        d.append("c", json::Dict::from(|d2| {
                            d2.append("subj",JSON::List(self.c_subj.iter().map(|&i| JSON::Int(i as i64)).collect()));
                            d2.append("val", self.c_cof.as_slice());
                    }));
                }));
                taskdata.append(
                    "A", 
                    json::Dict::from(|d| {
                        d.append("subi",JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_subj[row[0]..row[0]+row[1]].iter()).map(|&i| JSON::Int(i as i64)).collect()));
                        d.append("subj",JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).enumerate().flat_map(|(i,row)| std::iter::repeat(i).take(row[1])).map(|i| JSON::Int(i as i64)).collect()));
                        d.append("val", JSON::List(self.a_ptr.permute_by(self.con_a_row.as_slice()).flat_map(|row| self.a_cof[row[0]..row[0]+row[1]].iter()).map(|&d| JSON::Float(d)).collect()));
                }));
        }));

        JSON::Dict(doc).write(strm)
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_optserver() {
        let addr = "solve.mosek.com:30080".to_string();
        let mut m = Model::new(Some("SuperModel"));
        m.set_parameter((), SolverAddress(addr));

        let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
        let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
        let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
        let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

        // Create variable 'x' of length 4
        let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

        // Create constraints
        let _ = m.constraint(None, x.index(1), less_than(10.0));
        let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
        let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
        let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

        // Set the objective function to (c^t * x)
        m.objective(Some("obj"), Sense::Maximize, x.dot(c));

        // Solve the problem
        m.write_problem("lo1-nosol.json");
        m.solve();

        // Get the solution values
        let (psta,dsta) = m.solution_status(SolutionType::Default);
        println!("Status = {:?}/{:?}",psta,dsta);
        let xx = m.primal_solution(SolutionType::Default,&x);
        println!("x = {:?}", xx);
    }

}
