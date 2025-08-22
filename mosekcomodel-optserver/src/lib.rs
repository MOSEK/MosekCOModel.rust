//! This module implements a backend that uses a MOSEK OptServer instance for solving, for example
//! [solve.mosek.com:30080](http://solve.mosek.com). 
//!
use mosekcomodel::domain::{QuadraticCone, VectorDomainType};
use mosekcomodel::utils::Permutation;
use mosekcomodel::*;
use mosekcomodel::model::{IntSolutionManager, ModelWithIntSolutionCallback, ModelWithLogCallback};
use mosekcomodel::utils::iter::{Chunkation, ChunksByIterExt, PermuteByEx, PermuteByMutEx};
use itertools::{iproduct, izip, Permutations};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader,BufRead,Read};
use std::path::Path;

//mod http;
mod json;
mod bio;
mod msto;

pub type Model = ModelAPI<Backend>;


#[derive(Clone,Copy)]
struct ConItem {
    block_index : usize,
    offset     : usize,
}

#[derive(Clone,Copy)]
enum VarItem {
    /// Variable element corresponds to underlying variable `index`
    Linear,
    /// Variable element corresponds to underlying variable `index`, and the dual corresponds to
    /// the lower bound on the variable.
    LinearLowerBound,
    /// Variable element corresponds to underlying variable `index`, and the dual corresponds to
    /// the upper bound on the variable.
    LinearUpperBound,
    /// Variable element corresponds to underlying variable `index` and 
    Conic{conidx : usize}
}

#[derive(Clone)]
enum VecCone {
    Zero{dim : usize},
    NonNegative{dim : usize},
    NonPositive{dim : usize},
    Unbounded{dim : usize},
    Quadratic{dim : usize},
    RotatedQuadratic{dim : usize},
    PrimalPower{dim : usize, alpha : Vec<f64>},
    DualPower{dim : usize, alpha : Vec<f64>},
    PrimalExp,
    DualExp,
    PrimalGeometricMean{dim : usize},
    DualGeometricMean{dim : usize},
    ScaledVectorizedPSD{dim : usize},
}
impl VecCone {
    pub fn dim(&self) -> usize {
        use VecCone::*;
        match self {
            Zero{dim} => *dim,
            NonNegative{dim} => *dim,
            NonPositive{dim} => *dim,
            Unbounded{dim} => *dim,
            Quadratic{dim} => *dim,
            RotatedQuadratic{dim} => *dim,
            PrimalPower{dim, ..} => *dim,
            DualPower{dim, ..} => *dim,
            PrimalExp => 3,
            DualExp => 3,
            PrimalGeometricMean{dim} => *dim,
            DualGeometricMean{dim} => *dim,
            ScaledVectorizedPSD{dim} => *dim,
        }
    }
}

/// Simple model object that supports input of linear, conic and disjunctive constraints. It only
/// stores data, it does not support solving or writing problems.
#[derive(Default)]
pub struct Backend {
    name : Option<String>,
    
    /// Message callback
    log_cb        : Option<Box<dyn Fn(&str)>>,
    /// Integer solution callback
    sol_cb        : Option<Box<dyn FnMut(&IntSolutionManager)>>,

    /// Lower bounds of underlying variables
    var_lb : Vec<f64>,
    /// Upper bounds of underlying variables
    var_ub : Vec<f64>,

    /// Indicates per scalar variable which are integer constrained. 
    var_int       : Vec<bool>,
    /// Variable names.
    var_names     : Vec<Option<String>>,
    /// Corresponding native indexes
    var_idx : Vec<usize>,
    /// Variables. Each value is the index of the corresponding scalar constraint defining the
    /// domain.
    vars          : Vec<VarItem>,

    /// Matrix storage
    mx            : msto::MatrixStore,

    /// Scalar constraint elements, each element corresponds to a scalar linear constraint or an
    /// single element in a conic constraint.
    cons          : Vec<ConItem>,
    con_mx_row    : Vec<usize>,
    con_rhs       : Vec<f64>,
    con_block_ptr : Vec<usize>,
    con_block_dom       : Vec<VecCone>,
    con_names     : Vec<Option<String>>,

    sense_max     : bool,
    c_subj        : Vec<usize>,
    c_cof         : Vec<f64>,

    address       : Option<reqwest::Url>,
    dpar          : HashMap<String,f64>,
    ipar          : HashMap<String,i32>,
}

impl BaseModelTrait for Backend {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name : name.map(|v| v.to_string()),
            vars : vec![VarItem::Linear],
            var_idx : vec![usize::MAX],
            ..Default::default()
        }
    }

    fn free_variable<const N : usize>
        (&mut self,
         name  : Option<&str>,
         shape : &[usize;N]) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result, String> where Self : Sized 
    {
        let n = shape.iter().product::<usize>();

        let lb = vec![f64::NEG_INFINITY;n];
        let ub = vec![f64::INFINITY;n];
        let idxs = self.native_linear_variable(lb.as_slice(),ub.as_slice(),false);


        let first = self.vars.len();
        let last = first + n;

        if let Some(name) = name {
            self.linear_names(idxs.start,idxs.end, shape, None, name);
        }

        Ok(Variable::new((first..last).collect::<Vec<usize>>(), None, shape))
    }

    fn linear_variable<const N : usize,R>
        (&mut self, 
         name : Option<&str>,
         dom  : LinearDomain<N>) -> Result<<LinearDomain<N> as VarDomainTrait<Self>>::Result,String>    
        where 
            Self : Sized
    {
        let (dt,b,sp,shape,is_integer) = dom.dissolve();
        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());

        let (idxs,vit) =
            match dt {
                LinearDomainType::Free        => { (self.native_linear_variable(vec![f64::NEG_INFINITY;n].as_slice(), vec![f64::INFINITY; n].as_slice(), is_integer),VarItem::Linear) },
                LinearDomainType::Zero        => { (self.native_linear_variable(b.as_slice(),b.as_slice(), is_integer), VarItem::Linear) },
                LinearDomainType::NonNegative => { (self.native_linear_variable(b.as_slice(), vec![f64::INFINITY; n].as_slice(), is_integer),VarItem::LinearLowerBound) },
                LinearDomainType::NonPositive => { (self.native_linear_variable(vec![f64::NEG_INFINITY;n].as_slice(), b.as_slice(), is_integer),VarItem::LinearUpperBound) },
            };
        let (first,last) = (self.var_idx.len(),self.vars.len() + n);

        self.var_idx.reserve(n); for i in idxs.clone() { self.var_idx.push(i); }
        self.vars.resize(last, vit);
        
        if let Some(name) = name {
            self.linear_names(idxs.start, idxs.end, &shape, sp.as_deref(), name);
        }

        Ok(Variable::new((first..last).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
        where 
            Self : Sized 
    {
        let (shape,bl,bu,sp,is_integer) = dom.dissolve();
        let n = sp.as_ref().map(|v| v.len()).unwrap_or(shape.iter().product::<usize>());

        let idxs = self.native_linear_variable(bl.as_slice(),bu.as_slice(), is_integer);
        let first = self.var_idx.len();
        self.var_idx.reserve(n*2);
        for i in idxs.clone() { self.var_idx.push(i); }
        for i in idxs.clone() { self.var_idx.push(i); }
        self.vars.reserve(n*2);
        self.vars.resize(first+n,   VarItem::LinearLowerBound);
        self.vars.resize(first+n*2, VarItem::LinearUpperBound);

        if let Some(name) = name {
            self.linear_names(idxs.start, idxs.end, &shape, sp.as_deref(), name);
        }

        Ok((Variable::new((first..first+n).collect::<Vec<usize>>(), sp.clone(), &shape),
            Variable::new((first+n..first+2*n).collect::<Vec<usize>>(), sp, &shape)))
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
        let (dt,rhs,sp,shape,_is_integer) = dom.dissolve();

        if sp.is_some() { return Err("Sparse domain on costraint not allowd".to_string()); }
        assert_eq!(rhs.len(),ptr.len()-1); 
        assert_eq!(ptr.len(),shape.iter().product::<usize>()+1);
        let n = rhs.len();
        
        let first = self.con_rhs.len();

        let ptrchunks = Chunkation::new(ptr).unwrap();
        let rowidxs = izip!(ptrchunks.chunks(subj).unwrap(), 
                            ptrchunks.chunks(cof).unwrap())
            .map(|(subj,cof)| {
                //println!("{}:{}: linear_constraint(), subj = {:?}",file!(),line!(),subj);
                if let (Some(j),Some(c)) = (subj.first(),cof.first()) {
                    if *j == 0 {
                        self.mx.append_row(&subj[1..], &cof[..cof.len()-1], *c)
                    }
                    else {
                        self.mx.append_row(subj, cof, 0.0)
                    }
                }
                else {
                    self.mx.append_row(subj, cof, 0.0)
                }
            });

        let block_index = self.con_block_ptr.len();
        self.con_block_ptr.push(first);

        match dt {
            LinearDomainType::Free        => self.con_block_dom.push(VecCone::Unbounded   { dim: n }),
            LinearDomainType::Zero        => self.con_block_dom.push(VecCone::Zero        { dim: n }),
            LinearDomainType::NonNegative => self.con_block_dom.push(VecCone::NonNegative { dim: n }),
            LinearDomainType::NonPositive => self.con_block_dom.push(VecCone::NonPositive { dim: n }),
        }
        self.con_mx_row.extend(rowidxs);
        self.con_rhs.extend_from_slice(rhs.as_slice());

        self.cons.extend((0..n).map(|offset| ConItem{block_index, offset}));

        if let Some(name) = name {
            let mut name_index_buf = [1usize; N];
            for _ in 0..n {
                name_index_buf.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                self.con_names.push(Some(format!("{}{:?}", name, name_index_buf)));
            }
        }
        else {
            self.con_names.resize(self.con_names.len()+n,None);
        }

        Ok(Constraint::new((first..first+n).collect::<Vec<usize>>(), &shape))
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
        let (shape,bl,bu,sp,_) = dom.dissolve();
        if sp.is_some() { return Err("Sparse domain on costraint not allowd".to_string()); }
        assert_eq!(bl.len(),ptr.len()-1); 
        assert_eq!(bu.len(),ptr.len()-1); 
        assert_eq!(ptr.len(),shape.iter().product::<usize>()+1);

        let n = bl.len();
        
        let first0 = self.con_rhs.len();
        let last0  = first0+n;
        let first1 = last0;
        let last1  = first1+n*2;

        let ptrchunks = Chunkation::new(ptr).unwrap();
        let rowidxs : Vec<usize> = izip!(ptrchunks.chunks(subj).unwrap(), ptrchunks.chunks(cof).unwrap())
            .map(|(subj,cof)| {
                if let (Some(j),Some(c)) = (subj.first(),cof.first()) {
                    //println!("{}:{}: ranged_constraint(), subj = {:?}",file!(),line!(),subj);
                    if *j == 0 {
                        self.mx.append_row(&subj[1..], &cof[..cof.len()-1], *c)
                    }
                    else {
                        self.mx.append_row(subj, cof, 0.0)
                    }
                }
                else {
                    self.mx.append_row(subj, cof, 0.0)
                }
            })
            .collect();
        let block_index = self.con_block_ptr.len();

        self.con_block_ptr.push(self.con_rhs.len());
        self.con_block_ptr.push(self.con_rhs.len()+n);

        self.con_block_dom.push(VecCone::NonNegative { dim: n });
        self.con_block_dom.push(VecCone::NonPositive { dim: n });

        self.con_mx_row.extend_from_slice(rowidxs.as_slice());
        self.con_mx_row.extend_from_slice(rowidxs.as_slice());
        self.con_rhs.extend_from_slice(bl.as_slice());
        self.con_rhs.extend_from_slice(bu.as_slice());

        self.cons.extend((0..n).map(|offset| ConItem{block_index, offset} ));
        self.cons.extend((0..n).map(|offset| ConItem{block_index : block_index+1, offset} ));

        if let Some(name) = name {
            let mut name_index_buf = [1usize; N];
            for _ in 0..n {
                name_index_buf.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                self.con_names.push(Some(format!("{}{:?}", name, name_index_buf)));
            }
        }
        else {
            for _ in 0..n {
                self.con_names.push(None);
            }
        }
        Ok((Constraint::new((first0..last0).collect::<Vec<usize>>(), &shape),
            Constraint::new((first1..last1).collect::<Vec<usize>>(), &shape)))
    }

    fn update(& mut self, rowidxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>
    {
        if shape.iter().product::<usize>() != rowidxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = rowidxs.iter().max() {
            if i >= self.cons.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

        let ptrchunker = Chunkation::new(ptr).unwrap();
        for (i,subj,cof) in izip!(rowidxs.iter(),
                                  ptrchunker.chunks(subj).unwrap(),
                                  ptrchunker.chunks(cof).unwrap()) {
            if let Some((j,c)) = subj.first().zip(cof.first()) {
                if *j == 0 {
                    self.mx.replace_row(*i, &subj[1..], &cof[1..], *c);
                }
                else {
                    self.mx.replace_row(*i, subj, cof, 0.0);
                }
            }
            else {
                self.mx.replace_row(*i, subj, cof, 0.0);
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
                    self.write_jtask(&mut File::create(p).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
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
        if let Some(url) = &self.address {
            let mut url = url.clone(); 
            url.set_path("/api/v1/submit+solve");

            let (req_r,mut req_w) = std::io::pipe().map_err(|e| e.to_string())?;
            let (mut resp_r,mut resp_w) = std::io::pipe().map_err(|e| e.to_string())?;

            let t = std::thread::spawn(move || {
                let client = reqwest::blocking::Client::new();
                client.post(url)
                    .header("Content-Type", "application/x-mosek-jtask")
                    .header("Accept", "application/x-mosek-multiplex")
                    .header("X-Mosek-Callback", "values")
                    .header("X-Mosek-Stream", "log")
                    .body(reqwest::blocking::Body::new(req_r))
                    .send()
                    .map_err(|e| e.to_string())
            });

            self.write_jtask(&mut req_w).map_err(|e| e.to_string())?;
            drop(req_w);

            let mut resp = t.join().unwrap()?;

            if ! resp.status().is_success() {
                return Err(format!("OptServer responded with code {}",resp.status()));
            }

            let mut content_type = None;
            for (k,v) in resp.headers().iter() {
                if k == "content-type" {
                    content_type = Some(v);
                }
            }
            if let Some(ct) = content_type {
                if ct != "application/x-mosek-multiplex" {
                    return Err(format!("Unexpected response format: {:?}",ct));
                }
            }
            else {
                return Err("Unexpected response format: Missing".to_string());
            }

            let t = std::thread::spawn(move || {
                resp.copy_to(&mut resp_w).map_err(|e| e.to_string())
            });

            let res = self.parse_multistream(&mut resp_r,sol_bas,sol_itr,sol_itg);
            t.join().unwrap().and_then(|_| res)?;
        }
        else {
            return Err("No optserver address given".to_string());
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
}


pub trait OptserverDomainTrait : VectorDomainTrait {

}

impl<D> VectorConeModelTrait<D> for Backend where D : VectorDomainTrait+'static {
    fn conic_variable<const N : usize>(&mut self, name : Option<&str>,dom : VectorDomain<N,D>) -> Result<Variable<N>,String> {
        let (dt,rhs,shape,conedim,is_int) = dom.dissolve();
        self.conic_variable(name,shape,conedim,dt.to_conic_domain_type(),rhs.as_slice(),is_int)

    }
    fn conic_constraint<const N : usize>(& mut self, name : Option<&str>, dom  : VectorDomain<N,D>, _shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<Constraint<N>,String> {
        let (dt,rhs,shape,conedim,_is_int) = dom.dissolve();

        self.conic_constraint(name, ptr, subj, cof, shape, conedim, dt.to_conic_domain_type(), rhs.as_slice())        
    }
}


impl ModelWithIntSolutionCallback for Backend {
    fn set_solution_callback<F>(&mut self, func : F) where F : 'static+FnMut(&IntSolutionManager) {
        self.sol_cb = Some(Box::new(func))
    }
}




mod msgread {
    use std::io::Read;

    pub struct MessageReader<'a,R> where R : Read {
        eof : bool,
        frame_remains : usize,
        final_frame : bool,
        s : & 'a mut R

    }
    impl<'a,R> MessageReader<'a,R> where R : Read {
        pub fn new(s : &'a mut R) -> MessageReader<'a,R> { MessageReader { eof: false, frame_remains: 0, s , final_frame : false}}
        #[allow(unused)]
        pub fn skip(&mut self) -> std::io::Result<()> {
            let mut buf = [0;4096];
            while 0 < self.read(&mut buf)? {}
            Ok(())
        }
    }

    impl<'a,R> Read for MessageReader<'a,R> where R : Read {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            //println!("MessageReader::read(), eof = {}, frame remains : {}, final_frame = {}",self.eof,self.frame_remains,self.final_frame);
            if self.eof {
                Ok(0)
            }
            else {
                if self.frame_remains == 0 {
                    if self.final_frame {
                        return Ok(0);
                    }
                    let mut buf = [0;2];
                    //println!("MessageReader::read(), read frame header...");
                    self.s.read_exact(&mut buf).map_err(|e| std::io::Error::other("Message stream error: Failed to read message header"))?;

                    self.final_frame = buf[0] > 127;
                    self.frame_remains = (((buf[0] & 0x7f) as usize) << 8) | (buf[1] as usize);
                    //println!("MessageReader::read(), new frame : {}",self.frame_remains);
                }

                let n = buf.len().min(self.frame_remains);
                //println!("MessageReader::read(), new frame : {}",self.frame_remains);
                let nr = self.s.read(&mut buf[..n])?;
                self.frame_remains -= nr;
                Ok(nr)
            }
        }
    }
}

use msgread::*;
pub struct SolverAddress(pub String);

impl SolverParameterValue<Backend> for SolverAddress {
    type Key = ();
    fn set(self,_parname : Self::Key, model : & mut Backend) -> Result<(),String> {
        model.address = Some(reqwest::Url::parse(self.0.as_str())
            .map_err(|_| "Invalid SolverAddress value".to_string())
            .and_then(|mut url| 
                if url.scheme() == "" {
                    url.set_scheme("http").unwrap();
                    Ok(url)
                }
                else if url.scheme().eq_ignore_ascii_case("http") || url.scheme().eq_ignore_ascii_case("https") { 
                    Ok(url) 
                } else {
                    Err(format!("Invalid url scheme: {}",url.as_str()))
                })?);
        Ok(())
    }
}

impl SolverParameterValue<Backend> for f64 {
    type Key = &'static str;
    fn set(self,parname : Self::Key, model : & mut Backend) -> Result<(),String> {
        _ = model.dpar.insert(parname.to_string(), self);
        Ok(())
    }
}
impl SolverParameterValue<Backend> for i32 {
    type Key = &'static str;
    fn set(self,parname : Self::Key, model : & mut Backend) -> Result<(),String> {
        _ = model.ipar.insert(parname.to_string(), self);
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
    fn parse_multistream<R>(&mut self, r : &mut R, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution) -> Result<(),String> where R : Read {
        //println!("----- Backend::parse_multistream()");
        // We should now receive an application/x-mosek-multiplex stream ending with either a
        // fail or a result.
        // parse incoming stream 
        let mut buf = Vec::new();
        let mut head = String::new();

        'outer: loop { // loop over messages
            // for each message loop over frames
            head.clear();
            {
                let mut mr = BufReader::new(MessageReader::new(r));
                mr.read_line(&mut head).map_err(|e| e.to_string())?;


                loop {
                    let n = mr.read_line(&mut head).map_err(|e| e.to_string())?;
                    if n <= 1 { break; }
                }
                //println!("Backend::parse_multistream(), head = [[[{}]]]",head);

                let head = head.trim_ascii_end();
                let mut lines = head.as_bytes().chunk_by(|&a,_| a != b'\n');
                let hd = lines.next().ok_or_else(|| "Invalid response format A".to_string())?.trim_ascii_end();
                let headers = lines
                    .map(|s| s.trim_ascii_end())
                    .map(|s| { if let Some(p) = subseq_location(s, b":") { (&s[..p],&s[p+1..]) } else { (&s[..0],s) } });
                
                match hd {
                    b"log" => {
                        if let Some(f) = &self.log_cb {
                            buf.clear();
                            mr.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                            if let Ok(s) = std::str::from_utf8(buf.as_slice()) {
                                f(s)
                            }
                        }
                    },
                    b"msg" => {},
                    b"wrn" => {},
                    b"err" => {},
                    b"cbmap" => {},
                    b"cbinfo" => {},
                    b"cbcode" => {},
                    b"cbwarn" => {},
                    b"cb-intsol" => {
                        if self.sol_cb.is_some() {
                            let mut xxbytes = Vec::new();
                            mr.read_to_end(&mut xxbytes).map_err(|e| e.to_string())?;
                            let n = xxbytes.len()/size_of::<f64>();
                            if n == self.var_lb.len() {
                                let mut xx : Vec<f64> = Vec::new(); xx.resize(n,0.0);
                                unsafe{xx.align_to_mut().1}.copy_from_slice(&xxbytes[..n*size_of::<f64>()]);

                                let mut solxx = Vec::new(); solxx.resize(self.vars.len(),0.0);

                                for (d,v) in solxx.iter_mut().zip( xx.permute_by(self.var_idx.as_slice())) { *d = *v; }
                                    
                                let obj : f64 = self.c_cof.iter().zip(solxx.permute_by(self.c_subj.as_slice())).map(|(c,x)| *c * *x).sum();

                                if let Some(cb) = & mut self.sol_cb {
                                    cb(&IntSolutionManager::new(obj,solxx));
                                }
                            }
                        }
                    },
                    b"ok" => {
                        let mut _trm = None;
                        let mut content_type = None;
                        for (k,v) in headers {
                            match k {
                                b"trm" => _trm = Some(v),
                                b"content-type" => {
                                    content_type = Some(v)
                                }
                                _ => {},
                           }
                        }

                        match content_type {
                            Some(b"application/x-mosek-b") => break 'outer self.read_bsolution(sol_bas,sol_itr,sol_itg,&mut mr).map_err(|e| e.to_string()),
                            Some(b"application/x-mosek-b+zstd") => {
                                let mut unzstd = zstd::Decoder::new(mr).map_err(|e| e.to_string())?;
                                break 'outer self.read_bsolution(sol_bas,sol_itr,sol_itg,&mut unzstd).map_err(|e| e.to_string())
                            },
                            Some(content_type) => break 'outer Err(format!("Unexpected solution format: {}",std::str::from_utf8(content_type).unwrap_or("<?>"))),
                            None => break 'outer Err(format!("Missing solution format"))
                        }

                    },
                    b"fail" => { 
                        let mut res = None;
                        for (k,v) in headers {
                            if k == b"res" { res = Some(v); }
                        }

                        buf.clear();
                        mr.read_to_end(&mut buf).map_err(|e| e.to_string())?;
                        let message = std::str::from_utf8(buf.as_slice()).unwrap_or("");
                        break 'outer Err(format!("Solve failed ({}): {}",
                                         std::str::from_utf8(res.unwrap_or(b"?")).unwrap_or("?"),
                                         message));
                    },
                    _ => {},
                }

                let mut flushbuf = [0u8;4096];
                while 0 < mr.read(&mut flushbuf).map_err(|e| e.to_string())? {}
            }
        }
    }
    
    fn copy_solution(&self,
                     psta : SolutionStatus,
                     dsta : SolutionStatus,
                     numvar : usize,
                     //numcon : usize,
                     pobj : f64,
                     dobj : f64,
                     //_varsta : Vec<u8>,
                     xx : Option<Vec<f64>>,
                     sx : Option<(Vec<f64>,Vec<f64>)>,
                     //_consta : Vec<u8>,
                     //xc : Option<Vec<f64>>,
                     sc : Option<Vec<f64>>,
                     sol : & mut Solution) -> std::io::Result<()>
    {
        let pdef = if let SolutionStatus::Undefined = psta { false } else { true };
        let ddef = if let SolutionStatus::Undefined = dsta { false } else { true };

        sol.primal.status = psta;
        sol.dual.status = dsta;

        if pdef {
            sol.primal.obj = pobj;
            sol.primal.con.clear();
            if let Some(xx) = xx {
                sol.primal.var.resize(self.var_idx.len(),0.0);
                sol.primal.var[0] = 1.0;
                xx.permute_by(&self.var_idx[1..])
                    .zip(&mut sol.primal.var[1..])
                    .for_each(|(&src,dst)| *dst = src);

                self.mx.eval_into(sol.primal.var.as_slice(),&mut sol.primal.con).unwrap();
            }
            else if ! self.vars.is_empty() {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
            else {
                sol.primal.con.resize(self.con_rhs.len(),0.0);
            }
        }
        
        if ddef {
            if let Some(sc) = sc.as_ref() {
                if self.con_mx_row.len() != sc.len() {
                    return Err(std::io::Error::other("Incorrect solution dimension in sol/acc/primal"));
                }
            }
            sol.dual.obj = dobj;
            if let Some(((sl,su),sc)) = sx.as_ref().zip(sc.as_ref()) {
                if sl.len() != numvar || su.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/var/dual")); }
                sol.dual.var.resize(self.var_idx.len(),0.0);
                for (solx,index,e) in izip!(sol.dual.var[1..].iter_mut(),self.var_idx[1..].iter(),self.vars.iter()) {
                    *solx = 
                        match e {
                           VarItem::Linear           =>  sl[*index]-su[*index],
                           VarItem::LinearLowerBound => -su[*index],
                           VarItem::LinearUpperBound =>  sl[*index],
                           VarItem::Conic { conidx } =>  sc[*conidx],
                        };
                }
            }
            else if ! self.vars.is_empty() {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
            
            if let Some(sc) = sc.as_ref() {
                sol.dual.con.copy_from_slice(sc.as_slice());
            }
            else if ! self.cons.is_empty() {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
        }

        Ok(())
    }

    /// The b-solution format.
    ///
    /// # Info section
    /// The info section is identical to the btask info section. 
    /// ```text
    /// MSKSOLN [B
    /// version III
    /// numvar I
    /// numbarvar I
    /// numcon I
    /// numcone I
    /// numacc L
    /// ```
    /// 
    /// # Solution 
    /// Solution section is the same as in btask. 
    ///
    /// Note that both primal and dual solution values are present, even for values that are
    /// logically unknown or undefined (for e.g. certificates or unknown status).
    ///
    /// ## Basic solution
    /// The entire section is optional. 
    /// ```text
    /// sol/basic               [B[B   -- prosta, solsta
    /// sol/basic/pobj          d      -- if primal solution is not undefined
    /// sol/basic/dobj          d      -- if dual solution is not undefined
    /// sol/basic/var/sta       [B 
    /// sol/basic/var/primal    [d  
    /// sol/basic/var/dual      [d[d  
    /// sol/basic/con/sta       [B     -- if numcon > 0
    /// sol/basic/con/primal    [d     -- if numcon > 0 and primal solution is defined
    /// sol/basic/con/dual      [d[d[d -- if numcon > 0 and dual solution is defined
    /// ```
    ///
    /// ## Interior solution
    /// The entire section is optional. 
    /// ```text
    /// sol/interior               [B[B   -- prosta, solsta
    /// sol/interior/pobj          d      -- if primal solution is not undefined
    /// sol/interior/dobj          d      -- if dual solution is not undefined
    /// sol/interior/var/sta       [B 
    /// sol/interior/var/primal    [d  
    /// sol/interior/var/dual      [d[d   -- if numcone = 0 
    ///                            [d[d[d -- if numcone > 0
    /// sol/interior/barvar/primal [d     -- if numbarvar > 0 and primal solution is defined
    /// sol/interior/barvar/dual   [d     -- if numbarvar > 0 and dual solution is defined
    /// sol/interior/con/sta       [B     -- if numcon > 0
    /// sol/interior/con/primal    [d     -- if numcon > 0 and primal solution is defined
    /// sol/interior/con/dual      [d[d[d -- if numcon > 0 and dual solution is defined
    /// sol/interior/cone/sta      [B     -- if numcone > 0
    /// sol/interior/acc/dual      [d     -- if numacc > 0 and dual solution is defined
    /// ```
    ///
    /// ## Integer solution
    /// The entire section is optional. 
    /// ```text
    /// sol/integer               [B[B   -- prosta, solsta
    /// sol/integer/pobj          d      -- if primal solution is not undefined
    /// sol/integer/var/sta       [B 
    /// sol/integer/var/primal    [d  
    /// sol/integer/barvar/primal [d     -- if numbarvar > 0 and primal solution is defined
    /// sol/integer/con/sta       [B     -- if numcon > 0
    /// sol/integer/con/primal    [d     -- if numcon > 0 and primal solution is defined
    /// sol/integer/cone/sta      [B     -- if numcone > 0
    /// ``` 
    ///
    /// ## Names
    /// ```text
    /// name/var       [I[B
    /// name/barvar    [I[B 
    /// name/con       [I[B 
    /// name/cone      [I[B 
    /// name/acc       [I[B
    /// name/problem   [B
    /// name/objective [B
    /// ```
    ///
    /// # Information items
    /// ```text
    /// inf/f64    [B[B[d
    /// inf/i32    [B[B[i 
    /// inf/i64    [B[B[l 
    /// ```
    fn read_bsolution<R>(&self, sol_bas : & mut Solution, sol_itr : &mut Solution, sol_itg : &mut Solution,r : &mut R) -> std::io::Result<()> where R : Read 
    {
        let mut r = bio::Des::new(r)?;
        
        let mut note : Vec<u8> = Vec::new();
        r.expect(b"MSKSOLN",b"[B")?
            .read_into(&mut note)?;

        let version = 
            {
                let mut entry = r.expect(b"version",b"III")?;
                (entry.next_value::<u32>()?,
                 entry.next_value::<u32>()?,
                 entry.next_value::<u32>()?)
            };

        if version.0 != 10 || version.1 != 2 {
            return Err(std::io::Error::other(format!("Unsupported solution format version: {}.{}.{}",version.0,version.1,version.2)));
        }

        sol_bas.primal.status = Undefined;
        sol_bas.dual.status   = Undefined;  
        sol_itr.primal.status = Undefined;
        sol_itr.dual.status   = Undefined;
        sol_itg.primal.status = Undefined;
        sol_itg.dual.status   = Undefined;

        let numvar    = r.expect(b"numvar",b"I")?.next_value::<u32>()? as usize;
        let numbarvar = r.expect(b"numbarvar",b"I")?.next_value::<u32>()?;
        let numcon    = r.expect(b"numcon",b"I")?.next_value::<u32>()? as usize;
        let numcone   = r.expect(b"numcone",b"I")?.next_value::<u32>()?;
        let numacc    = r.expect(b"numacc",b"L")?.next_value::<u64>()? as usize;

        if self.var_lb.len() != numvar { return Err(std::io::Error::other("Invalid solution dimension")); }
        if self.var_ub.len() != numvar { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numcone { return Err(std::io::Error::other("Invalid solution dimension")); }
        if self.con_block_ptr.len() != numacc { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numbarvar { return Err(std::io::Error::other("Invalid solution dimension")); }
    
        let mut acc_pattern : Vec<u64> = Vec::new();
        if let Some((b"acc/pattern",b"[L")) = r.peek()? {
            r.next_entry()?.unwrap().read_into(&mut acc_pattern)?;
        }
        //let mut entry = r.next_entry()?.ok_or_else(|| std::io::Error::other("Missing solution entries"))?;
 
        use SolutionStatus::*;

        if let Some((b"sol/interior",b"[B[B")) = r.peek()? {
            sol_itr.primal.var.resize(self.vars.len(),0.0);
            sol_itr.dual.var.resize(self.vars.len(),0.0);
            sol_itr.primal.con.resize(self.cons.len(),0.0);
            sol_itr.dual.con.resize(self.cons.len(),0.0);

            let sta    = r.expect(b"sol/interior",b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef   = !matches!(sta.0,Undefined);
            let ddef   = !matches!(sta.1,Undefined);
            let pobj   = if pdef { r.expect(b"sol/interior/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let dobj   = if ddef { r.expect(b"sol/interior/dobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let _varsta = r.expect(b"sol/interior/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx     = if pdef { Some(r.expect(b"sol/interior/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sx     = if ddef { Some(r.expect(b"sol/interior/var/dual",b"[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            //let consta = r.expect(b"sol/interior/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            //let xc     = if numcon > 0 && pdef { Some(r.expect(b"sol/interior/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sc     = if numcon > 0 && ddef { Some(r.expect(b"sol/interior/con/dual",b"[d[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let sn     = if numacc > 0 && ddef { Some(r.expect(b"sol/interior/acc/dual",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };

            self.copy_solution(sta.0,sta.1,
                               numvar, 
                               pobj,dobj,
                               //varsta,
                               xx,sx,
                               //consta,
                               //xc,
                               sn,
                               sol_itr)?;
        }

        if let Some((b"sol/basic",b"[B[B")) = r.peek()? {
            let sta    = r.expect(b"sol/basic",b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef   = !matches!(sta.0,Undefined);
            let ddef   = !matches!(sta.1,Undefined);
            let pobj   = if pdef { r.expect(b"sol/basic/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let dobj   = if ddef { r.expect(b"sol/basic/dobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let varsta = r.expect(b"sol/basic/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx     = if pdef { Some(r.expect(b"sol/basic/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sx     = if ddef { Some(r.expect(b"sol/basic/var/dual",b"[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            //let consta = r.expect(b"sol/basic/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let _xc     = if numcon > 0 && pdef { Some(r.expect(b"sol/basic/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sc     = if numcon > 0 && ddef { Some(r.expect(b"sol/basic/con/dual",b"[d[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let sn     = if numacc > 0 && ddef { Some(r.expect(b"sol/basic/acc/dual",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };

            self.copy_solution(sta.0,sta.1,
                               numvar,
                               pobj,dobj,
                               //varsta,
                               xx,sx,
                               //consta,
                               //xc,
                               sn,
                               sol_bas)?;
        }

        if let Some((b"sol/integer",b"[B[B")) = r.peek()? {
            let sta    = r.expect(b"sol/integer",b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef   = !matches!(sta.0,Undefined);
            let pobj   = if pdef { r.expect(b"sol/integer/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let varsta = r.expect(b"sol/integer/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx     = if pdef { Some(r.expect(b"sol/integer/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            //let consta = r.expect(b"sol/integer/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let _xc     = if numcon > 0 && pdef { Some(r.expect(b"sol/integer/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            
            self.copy_solution(sta.0,sta.1,
                               numvar,
                               pobj,0.0,
                               //varsta,
                               xx,None,
                               //consta,
                               //xc,
                               None,
                               sol_itg)?;
        }

        r.expect(b"name/var",b"[I[B")?.skip_all()?;
        r.expect(b"name/barvar",    b"[I[B")?.skip_all()?;
        r.expect(b"name/con",       b"[I[B")?.skip_all()?;
        r.expect(b"name/cone",      b"[I[B")?.skip_all()?;
        r.expect(b"name/acc",       b"[I[B")?.skip_all()?;
        r.expect(b"name/problem",   b"[B")?.skip_all()?;
        r.expect(b"name/objective", b"[B")?.skip_all()?;

        r.expect(b"inf/f64", b"[B[B[d")?.skip_all()?;
        r.expect(b"inf/i32", b"[B[B[i")?.skip_all()?;
        r.expect(b"inf/i64", b"[B[B[l")?.skip_all()?;

        Ok(())
    }

    /// JSON Task format writer.
    ///
    /// Relevant parts from [https://docs.mosek.com/latest/capi/json-format.html]:
    /// - `$schema`: JSON schema = "`http://mosek.com/json/schema#`".
    /// - `Task/name`: The name of the task (string).
    /// - `Task/INFO`: Information about problem data dimensions and similar. These are treated as hints when reading the file.
    ///     - `numvar`: number of variables (int32).
    ///     - `numcon`: number of constraints (int32).
    ///     - `numcone`: number of cones (int32, deprecated).
    ///     - `numbarvar`: number of symmetric matrix variables (int32).
    ///     - `numanz`: number of nonzeros in A (int64).
    ///     - `numsymmat`: number of matrices in the symmetric matrix storage E (int64).
    ///     - `numafe`: number of affine expressions in AFE storage (int64).
    ///     - `numfnz`: number of nonzeros in F (int64).
    ///     - `numacc`: number of affine conic constraints (ACCs) (int64).
    ///     - `numdjc`: number of disjunctive constraints (DJCs) (int64).
    ///     - `numdom`: number of domains (int64).
    ///     - `mosekver`: MOSEK version (list(int32)).
    /// - `Task/data`: Numerical and structural data of the problem.
    ///     - `var`: Information about variables. All fields present must have the same length as bk. All or none of bk, bl, and bu must appear.
    ///         - `name`: Variable names (list(string)).
    ///         - `bk`: Bound keys (list(string)).
    ///         - `bl`: Lower bounds (list(double)).
    ///         - `bu`: Upper bounds (list(double)).
    ///         - `type`: Variable types (list(string)).
    ///     - `con`: Information about linear constraints. All fields present must have the same length as bk. All or none of bk, bl, and bu must appear.
    ///         - `name`: Constraint names (list(string)).
    ///         - `bk`: Bound keys (list(string)).
    ///         - `bl`: Lower bounds (list(double)).
    ///         - `bu`: Upper bounds (list(double)).
    ///     - `objective`: Information about the objective.
    ///         - `name`: Objective name (string).
    ///         - `sense`: Objective sense (string).
    ///         - `c`: The linear part of the objective as a sparse vector. Both arrays must have the same length.
    ///     - `subj`: indices of nonzeros (list(int32)).
    ///     - `val`: values of nonzeros (list(double)).
    /// - `cfix`: Constant term in the objective (double).
    /// - `A`: The linear constraint matrix as a sparse matrix. All arrays must have the same length.
    ///     - `subi`: row indices of nonzeros (list(int32)).
    ///     - `subj`: column indices of nonzeros (list(int32)).
    ///     - `val`: values of nonzeros (list(double)).
    /// - `AFE`: The affine expression storage.
    ///     - `numafe`: number of rows in the storage (int64).
    ///     - `F`: The matrix as a sparse matrix. All arrays must have the same length.
    ///         - `subi`: row indices of nonzeros (list(int64)).
    ///         - `subj`: column indices of nonzeros (list(int32)).
    ///         - `val`: values of nonzeros (list(double)).
    ///     - `g`: The vector of constant terms as a sparse vector. Both arrays must have the same length.
    ///         - `subi`: indices of nonzeros (list(int64)).
    ///         - `val`: values of nonzeros (list(double)).
    /// - `domains`: Information about domains. All fields present must have the same length as type.
    ///     - `name`: Domain names (list(string)).
    ///     - `type`: Description of the type of each domain (list). Each element of the list is a list describing one domain using at least one field:
    ///         domain type (string).
    ///         (except pexp, dexp) dimension (int64).
    ///         (only ppow, dpow) weights (list(double)).
    /// - `ACC`: Information about affine conic constraints (ACC). All fields present must have the same length as domain.
    ///     - `name`: ACC names (list(string)).
    ///     - `domain`: Domains (list(int64)).
    ///     - `afeidx`: AFE indices, grouped by ACC (list(list(int64))).
    ///     - `b`: constant vectors 
    ///     , grouped by ACC (list(list(double))).
    /// - `DJC`: Information about disjunctive constraints (DJC). All fields present must have the same length as termsize.
    ///     - `name`: DJC names (list(string)).
    ///     - `termsize`: Term sizes, grouped by DJC (list(list(int64))).
    ///     - `domain`: Domains, grouped by DJC (list(list(int64))).
    ///     - `afeidx`: AFE indices, grouped by DJC (list(list(int64))).
    ///     - `b`: constant vectors 
    ///     , grouped by DJC (list(list(double))).
    /// - `Task/solutions`: Solutions. This section can contain up to three subsections called:
    ///     - `interior`
    ///     - `basic`
    ///     - `integer`
    ///     corresponding to the three solution types in MOSEK. Each of these sections has the same structure:
    ///     - `prosta`: problem status (string).
    ///     - `solsta`: solution status (string).
    ///     - `xx`, `xc`, `y`, `slc`, `suc`, `slx`, `sux`, `snx`: one for each component of the solution of the same name (list(double)).
    ///     - `skx`, `skc`, `skn`: status keys (list(string)).
    ///     - `doty`: the dual solution, grouped by ACC (list(list(double))).
    /// - `Task/parameters`: Parameters.
    ///     - `iparam`: Integer parameters (dictionary). A dictionary with entries of the form
    ///       name:value, where name is a shortened parameter name (without leading MSK_IPAR_) and
    ///       value is either an integer or string if the parameter takes values from an enum.
    ///     - `dparam`: Double parameters (dictionary). A dictionary with entries of the form
    ///       name:value, where name is a shortened parameter name (without leading MSK_DPAR_) and
    ///       value is a double.
    ///     - `sparam`: String parameters (dictionary). A dictionary with entries of the form
    ///       `name:value`, where name is a shortened parameter name (without leading MSK_SPAR_) and
    ///       value is a string. Note that this section is allowed but MOSEK ignores it both when
    ///       writing and reading JTASK files.
    fn write_jtask<S>(&self, strm : &mut S) -> std::io::Result<()> 
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
                taskinfo.append("numcon",0);
                taskinfo.append("numafe",self.mx.num_row() as i64);
                taskinfo.append("numacc",self.con_block_dom.len() as i64);
            }));
            
        doc.append(
            "Task/data",
            json::Dict::from(|taskdata| {
                {
                    let bk = self.var_lb.iter().zip(self.var_ub.iter()).map(|(lb,ub)| bnd_to_bk(*lb, *ub).to_string()).collect();

                    taskdata.append("var",json::Dict::from(|d| {
                        d.append("bk",  JSON::StringArray(bk));
                        d.append("bl",  JSON::FloatArray(self.var_lb.clone()));
                        d.append("bu",  JSON::FloatArray(self.var_ub.clone()));
                        d.append("type",JSON::StringArray(self.var_int.iter().map(|&e| if e { "int".into() } else { "cont".into() }).collect()));
                    }));
                }
                {
                    taskdata.append("con",json::Dict::from(|d| {
                        d.append("bk",  JSON::StringArray(Vec::new()));
                        d.append("bl",  JSON::FloatArray(Vec::new()));
                        d.append("bu",  JSON::FloatArray(Vec::new()));
                    }));
                }
                taskdata.append(
                    "objective",
                    json::Dict::from(|d| {
                        d.append("sense", if self.sense_max { "max" } else { "min" });
                        d.append("cfix",0.0f64);
                        d.append("c", json::Dict::from(|d2| {
                            d2.append("subj",JSON::IntArray(self.var_idx.permute_by(self.c_subj.as_slice()).map(|&i| i as i64).collect()));
                            d2.append("val", self.c_cof.as_slice());
                    }));
                }));
                taskdata.append(
                    "A", 
                    json::Dict::from(|d| {
                        d.append("subi",JSON::IntArray(Vec::new()));
                        d.append("subj",JSON::IntArray(Vec::new()));
                        d.append("val", JSON::FloatArray(Vec::new()));
                }));

                taskdata.append(
                    "AFE",
                    json::Dict::from(|d| {
                        let nnz = self.mx.num_nonzeros();
                        let mut subi = Vec::with_capacity(nnz);
                        let mut subj = Vec::with_capacity(nnz);
                        let mut val  = Vec::with_capacity(nnz);
                        let mut gs   = Vec::with_capacity(self.mx.num_row());

                        self.mx.row_iter()
                            .enumerate()
                            .for_each(|(i,(jj,cc,g))| {
                                let base = subi.len();
                                let n = jj.len();
                                subi.resize(base+n, i as i64);
                                subj.extend(self.var_idx.permute_by(jj).map(|&i| i as i64));
                                val.extend_from_slice(cc);
                                gs.push(g);
                            });
                        d.append("numafe", JSON::Int( self.mx.num_row() as i64));

                        d.append("F", json::Dict::from(|d| {
                            d.append("subi",JSON::IntArray(subi));
                            d.append("subj",JSON::IntArray(subj));
                            d.append("val", JSON::FloatArray(val));
                        }));
                        //d.append("g",JSON::FloatArray(self.mx.row_iter().map(|(_,_,g)| g).collect()));
                        d.append(
                            "g",
                            json::Dict::from(|d| {
                                d.append("subi",JSON::IntArray((0..self.mx.num_row() as i64).collect()));
                                d.append("val", JSON::FloatArray(self.mx.row_iter().map(|(_,_,g)| g).collect()));
                                }));
                    }));

                taskdata.append(
                    "domains",
                    JSON::Dict(
                        json::Dict::from(|d| 
                            d.append(
                                "type",
                                JSON::List(self.con_block_dom.iter().map(|c| dom2json(c)).collect::<Vec<JSON>>())))));
                taskdata.append(
                    "ACC",
                    json::Dict::from(|d| {
                        d.append("domain",JSON::IntArray((0..self.con_block_dom.len()).map(|i| i as i64).collect()));
                        let mut acc_afe_idxs : Vec<Vec<i64>> = self.con_block_dom.iter().map(|d| vec![0i64; d.dim()]).collect();
                        acc_afe_idxs.iter_mut().flat_map(|r| r.iter_mut()).zip(self.con_mx_row.iter()).for_each(|(d,&s)| *d = s as i64);
                        let mut acc_b : Vec<Vec<f64>> = self.con_block_dom.iter().map(|dom| vec![0f64; dom.dim()]).collect();
                        acc_b.iter_mut().flat_map(|row| row.iter_mut()).zip(self.con_rhs.iter()).for_each(|(d,&s)| *d = s);
                        d.append("afeidx",JSON::List( acc_afe_idxs.into_iter().map(|row| JSON::IntArray(row)).collect() ));
                        d.append("b",     JSON::List( acc_b.into_iter().map(|row| JSON::FloatArray(row)).collect() ));
                    }));
        }));
        doc.append(
            "Task/parameters",
            json::Dict::from(|d| {
                if ! self.dpar.is_empty() {
                    d.append(
                        "dparam",
                        json::Dict::from(|d| for (k,v) in self.dpar.iter() { d.append(k.as_str(), JSON::Float(*v)); }))
                }
                if ! self.ipar.is_empty() {
                    d.append(
                        "iparam",
                        json::Dict::from(|d| for (k,v) in self.ipar.iter() { d.append(k.as_str(), JSON::Int(*v as i64)); }))
                }
            }));
        JSON::Dict(doc).write(strm)
    }


    fn linear_names<const N : usize>(&mut self, first : usize, last : usize, shape : &[usize;N], sp : Option<&[usize]>, name : &str) {
        let mut name_index_buf = [1usize; N];
        let mut strides = [0;N];
        strides.iter_mut().zip(shape.iter()).rev().fold(1,|s,(st,&d)| { *st = s; d * s });
        if let Some(sp) = &sp {
            for (&i,n) in sp.iter().zip(self.var_names.iter_mut()) {
                name_index_buf.iter_mut().zip(strides.iter()).fold(i,|i,(ni,&st)| { *ni = i/st; i%st });
                *n= Some(format!("{}{:?}", name, name_index_buf));
            }
        }
        else {
            for n in self.var_names.iter_mut() {
                name_index_buf.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                *n = Some(format!("{}{:?}", name, name_index_buf));
            }
        }
    }


    /// Appends a range of native linear variables with upper and lower boundsm returns the range
    /// of the new variables.
    fn native_linear_variable(&mut self, lb : &[f64], ub : &[f64], is_int : bool) -> std::ops::Range<usize> {
        assert_eq!(lb.len(),ub.len());
        let n = lb.len();
        let first = self.var_lb.len();
        let last  = first + n;
        
        let first_nvar = self.var_lb.len();
        let last_nvar  = first_nvar+n;

        self.var_lb.extend_from_slice(lb);
        self.var_ub.extend_from_slice(ub);
        self.var_int.resize(last_nvar, is_int);
        self.var_names.resize(last_nvar, None);

        first..last
    }



    fn conic_variable<const N : usize>(
        &mut self, 
        name  : Option<&str>,
        // domain
        shape   : [usize;N],
        conedim : usize,
        dt      : VectorDomainType,
        offset  : &[f64],
        is_int  : bool) -> Result<Variable<N>,String> 
    {
        use VecCone::*;

        let n = shape.iter().product();
        let dim = shape[conedim];
        let numcone = n/dim;

        assert_eq!(offset.len(),n);


        let ct = 
            match dt {
                VectorDomainType::QuadraticCone          => Quadratic{dim},
                VectorDomainType::RotatedQuadraticCone   => RotatedQuadratic{dim},
                VectorDomainType::SVecPSDCone            => ScaledVectorizedPSD{dim},
                VectorDomainType::GeometricMeanCone      => PrimalGeometricMean {dim},
                VectorDomainType::DualGeometricMeanCone  => DualGeometricMean{dim},
                VectorDomainType::ExponentialCone        => PrimalExp,
                VectorDomainType::DualExponentialCone    => DualExp,
                VectorDomainType::PrimalPowerCone(alpha) => PrimalPower{dim,alpha},
                VectorDomainType::DualPowerCone(alpha)   => DualPower{dim,alpha},
                // linear types                                                   
                VectorDomainType::NonNegative            => NonNegative{dim},
                VectorDomainType::NonPositive            => NonPositive{dim},
                VectorDomainType::Zero                   => Zero{dim},
                VectorDomainType::Free                   => Unbounded {dim},
            };

        let base = self.var_lb.len();
        self.var_lb.resize(base+n,f64::NEG_INFINITY);
        self.var_ub.resize(base+n,f64::INFINITY);
        self.var_int.resize(base+n,is_int);
        self.var_names.resize(base+n,None);

        let firstvar = self.var_idx.len();
        let lastvar = firstvar + n;
     
        let con_base = self.con_rhs.len();
        self.con_rhs.extend_from_slice(offset);
        self.con_names.resize(con_base+n, None);

        self.con_mx_row.extend( (firstvar..lastvar).map(|i| self.mx.append_row(&[i], &[1.0], 0.0)) );
        self.con_block_dom.extend((0..numcone).map(|_| ct.clone()));
        self.con_block_ptr.extend((0..numcone).map(|i| base+i*dim ));

        self.var_idx.extend(base..base+n);
        self.vars.extend((con_base..con_base+dim).map(|conidx| VarItem::Conic { conidx }));


        let shape3 = [ shape[..conedim].iter().product(),dim,shape[conedim+1..].iter().product()];
        let strides3 = [ shape3[1]*shape[2], 1, shape3[2]];
        
        let perm : Vec::<usize> = iproduct!(0..shape3[0],0..shape3[1],0..shape3[2]).map(|(i0,i1,i2)| strides3[0]*i0 + strides3[1]*i1 + strides3[2]*i2).collect();
       
        let varidxs = perm.iter().map(|i| firstvar + i).collect();

        self.var_names.resize(base+n,None);

        if let Some(name) = name {
            let mut idx = [1; N];

            self.var_names[base..].permute_by_mut(perm.as_slice())
                .for_each(|n| {
                    *n = Some(format!("{}{:?}",name,idx));
                    _ = idx.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                });
        }

        Ok(Variable::new(varidxs, None, &shape))
    }
   
    fn conic_constraint<const N : usize>(
        &mut self, 
        name    : Option<&str>,
        // expr
        ptr     : &[usize], 
        subj    : &[usize], 
        cof     : &[f64],
        // domain
        shape   : [usize;N],
        conedim : usize,
        dt      : VectorDomainType,
        offset  : &[f64]) -> Result<Constraint<N>,String>
    {
        use VecCone::*;
        let n = shape.iter().product();
        let dim = shape[conedim];
        let numcone = n/dim;

        assert_eq!(offset.len(),n);

        let ct = 
            match dt {
                VectorDomainType::QuadraticCone          => Quadratic{dim},
                VectorDomainType::RotatedQuadraticCone   => RotatedQuadratic{dim},
                VectorDomainType::SVecPSDCone            => ScaledVectorizedPSD{dim},
                VectorDomainType::GeometricMeanCone      => PrimalGeometricMean {dim},
                VectorDomainType::DualGeometricMeanCone  => DualGeometricMean{dim},
                VectorDomainType::ExponentialCone        => PrimalExp,
                VectorDomainType::DualExponentialCone    => DualExp,
                VectorDomainType::PrimalPowerCone(alpha) => PrimalPower{dim,alpha},
                VectorDomainType::DualPowerCone(alpha)   => DualPower{dim,alpha},
                // linear types                                                   
                VectorDomainType::NonNegative            => NonNegative{dim},
                VectorDomainType::NonPositive            => NonPositive{dim},
                VectorDomainType::Zero                   => Zero{dim},
                VectorDomainType::Free                   => Unbounded {dim},
            };

        let firstcon = self.con_rhs.len();

        let ptr_perm = Chunkation::new(ptr).unwrap();
        self.con_mx_row.extend(
            izip!(ptr_perm.chunks(subj).unwrap(),
                  ptr_perm.chunks(cof).unwrap())
                .map(|(subj,cof)| {
                    //println!("{}:{}: conic_constraint(), subj = {:?}",file!(),line!(),subj);
                    if let (Some(j),Some(c)) = (subj.first(),cof.first()) {
                        if *j == 0 {
                            self.mx.append_row(&subj[1..], &cof[..cof.len()-1], *c)
                        }
                        else {
                            self.mx.append_row(subj, cof, 0.0)
                        }
                    }
                    else {
                        self.mx.append_row(subj, cof, 0.0)
                    }}));
        self.con_rhs.extend_from_slice(offset);
        self.con_names.resize(firstcon+n,None);
        let firstblock = self.con_block_dom.len();
        self.con_block_ptr.extend( (firstcon..firstcon+n).step_by(dim) );
        self.con_block_dom.extend( (0..numcone).map(|i| ct.clone() ));

        let shape3   = [ shape[..conedim].iter().product(),dim,shape[conedim+1..].iter().product()];
        let strides3 = [ shape3[1]*shape3[2], 1, shape3[2]];
        
        let perm : Vec::<usize> = iproduct!(0..shape3[0],0..shape3[1],0..shape3[2]).map(|(i0,i1,i2)| strides3[0]*i0 + strides3[1]*i1 + strides3[2]*i2).collect();
        let rescon0 = self.cons.len();

        self.cons.extend( iproduct!(firstblock..firstblock+numcone,0..dim).map(|(block_index,offset)| ConItem{ block_index, offset}));

        if let Some(name) = name {
            let mut idx = [1; N];

            self.con_names[firstcon..]
                .permute_by_mut(perm.as_slice())
                .for_each(|n| {
                    *n = Some(format!("{}{:?}",name,idx));
                    _ = idx.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                });
        }

        Ok(Constraint::new(perm.iter().map(|&i| rescon0+i).collect(), &shape))
    }
}





impl ModelWithLogCallback for Backend {
    fn set_log_handler<F>(& mut self, func : F) where F : 'static+Fn(&str) {
        self.log_cb = Some(Box::new(func));
    }
}

fn subseq_location_from<T>(pos : usize, src : &[T], seq : &[T]) -> Option<usize> where T : Eq {
    if pos+seq.len() > src.len() {
        return None;
    }

    for i in pos..src.len()-seq.len()+1 {
        if unsafe{ *src.get_unchecked(i) == *seq.get_unchecked(0) } {
            let mut found = true;
            for (j0,j1) in (i..i+seq.len()).enumerate() {
                if unsafe{ *src.get_unchecked(j1) != *seq.get_unchecked(j0) } {
                    found = false;
                    break;
                }
            }
            if found {
                return Some(i);
            }
        }
    }
    None
}
fn subseq_location<T>(src : &[T], seq : &[T]) -> Option<usize> where T : Eq {
    subseq_location_from(0,src, seq)
}

/// Convert a solution status string to primal/dual solution status.
fn str_to_pdsolsta(solsta : &[u8]) -> std::io::Result<(SolutionStatus,SolutionStatus)> {
    match solsta {
        b"UNKNOWN" =>            Ok((SolutionStatus::Unknown,SolutionStatus::Unknown)),
        b"OPTIMAL" =>            Ok((SolutionStatus::Optimal,SolutionStatus::Optimal)),
        b"PRIM_FEAS" =>          Ok((SolutionStatus::Feasible,SolutionStatus::Unknown)),
        b"DUAL_FEAS" =>          Ok((SolutionStatus::Unknown,SolutionStatus::Feasible)),
        b"PRIM_AND_DUAL_FEAS" => Ok((SolutionStatus::Feasible,SolutionStatus::Feasible)),
        b"PRIM_INFEAS_CER" =>    Ok((SolutionStatus::Undefined,SolutionStatus::CertInfeas)),
        b"DUAL_INFEAS_CER" =>    Ok((SolutionStatus::CertInfeas,SolutionStatus::Undefined)),
        b"PRIM_ILLPOSED_CER" =>  Ok((SolutionStatus::Undefined,SolutionStatus::CertIllposed)),
        b"DUAL_ILLPOSED_CER" =>  Ok((SolutionStatus::CertIllposed,SolutionStatus::Undefined)),
        b"INTEGER_OPTIMAL" =>    Ok((SolutionStatus::Optimal,SolutionStatus::Undefined)),
        _ => Err(std::io::Error::other(format!("Invalid solution format: {}",std::str::from_utf8(solsta).unwrap_or("<invalid utf-8>"))))
    }
}


fn dom2json(c : &VecCone) -> json::JSON {    
    use json::JSON;
    use VecCone::*;
    match c {
      Zero{dim}                  => JSON::List(vec![ JSON::String("rzero".to_string()), JSON::Int(*dim as i64)]),
      NonNegative{dim}           => JSON::List(vec![ JSON::String("rplus".to_string()), JSON::Int(*dim as i64)]),
      NonPositive{dim}           => JSON::List(vec![ JSON::String("rminus".to_string()), JSON::Int(*dim as i64)]),
      Unbounded{dim}             => JSON::List(vec![ JSON::String("r".to_string()), JSON::Int(*dim as i64)]),
      Quadratic{dim}             => JSON::List(vec![ JSON::String("quad".to_string()), JSON::Int(*dim as i64)]),
      RotatedQuadratic{dim}      => JSON::List(vec![ JSON::String("rquad".to_string()), JSON::Int(*dim as i64)]),
      PrimalGeometricMean{dim}   => JSON::List(vec![ JSON::String("pgmean".to_string()), JSON::Int(*dim as i64)]),
      DualGeometricMean{dim}     => JSON::List(vec![ JSON::String("dgmean".to_string()), JSON::Int(*dim as i64)]),
      PrimalExp                  => JSON::List(vec![ JSON::String("pexp".to_string())]),
      DualExp                    => JSON::List(vec![ JSON::String("dexp".to_string())]),
      PrimalPower { dim, alpha } => JSON::List(vec![ JSON::String("ppow".to_string()), JSON::Int(*dim as i64), JSON::FloatArray(alpha.clone())]), 
      DualPower { dim, alpha }   => JSON::List(vec![ JSON::String("dpow".to_string()), JSON::Int(*dim as i64), JSON::FloatArray(alpha.clone())]), 
      ScaledVectorizedPSD { dim } => JSON::List(vec![ JSON::String("svecpsd".to_string()), JSON::Int(*dim as i64)]), 
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_optserver() {
        //let addr = "http://solve.mosek.com:30080".to_string();
        let addr = "http://localhost:9999".to_string();
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

        m.write_problem("lo1-nosol.jtask");

        m.set_log_handler(|msg| print!("{}",msg));
        m.solve();

        // Get the solution values
        let (psta,dsta) = m.solution_status(SolutionType::Default);
        println!("Status = {:?}/{:?}",psta,dsta);
        let xx = m.primal_solution(SolutionType::Default,&x);
        println!("x = {:?}", xx);
    }
}
