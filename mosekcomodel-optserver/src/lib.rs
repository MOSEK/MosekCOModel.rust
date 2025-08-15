//! This module implements a backend that uses a MOSEK OptServer instance for solving, for example
//! [solve.mosek.com:30080](http://solve.mosek.com). 
//!
use mosekcomodel::*;
use mosekcomodel::model::{IntSolutionManager, ModelWithLogCallback};
use mosekcomodel::utils::iter:: PermuteByEx;
use itertools::izip;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader,BufRead,Read};
use std::path::Path;

//mod http;
mod json;
mod bio;

pub type Model = ModelAPI<Backend>;


#[derive(Clone,Copy)]
enum Element {
    Linear{lb:f64,ub:f64},
    Conic{conerow:usize}
}

#[derive(Clone,Copy)]
enum LinearItem{
    Linear,
    RangedUpper,
    RangedLower,
}

#[derive(Clone,Copy)]
enum Item {
    Linear{index:usize,kind:LinearItem},
    Conic{index:usize}
}

/// A scalar element in a conic constraint.
///
struct ConeElement { 
    /// Refers to the cone index, i.e. index into `cones` member in the model.
    index : usize, 
    /// Offset of the element into the cone.
    offset : usize 
}

enum VecConeType {
    Zero,
    NonNegative,
    NonPositive,
    Unbounded,
    Quadratic,
    RotatedQuadratic,
    PrimalPower,
    DualPower,
    PrimalExp,
    DualExp,
    PrimalGeometricMean,
    DualGeometricMean,
    ScaledVectorizedPSD,
}

impl Item {
    fn index(&self) -> usize { 
        use Item::*;
        match self {
            Linear { index,.. } => *index,
            Conic { index,.. } => *index,
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

    /// List of scalar variable elements. Each element is either a linear variable with bounds or a
    /// conic variable
    var_elt       : Vec<Element>, // Either lb,ub,int or index,coneidx,offset
    /// Indicates per scalar variable which are integer constrained. 
    var_int       : Vec<bool>,
    /// Variable names.
    var_names     : Vec<Option<String>>,
    /// Variable interfaces. Ranged variables produce two interface elements per scalar variable,
    /// one for upper bound and one for lower bound.
    vars          : Vec<Item>,

    /// Matrix storage
    mx            : msto::MatrixStore,

    /// Scalar constraint elements, each element corresponds to a scalar linear constraint or an
    /// single element in a conic constraint.
    con_elt       : Vec<Element>,
    /// Names of scalar constraint elements
    con_names     : Vec<Option<String>>,

    /// Scalar interface constraints.
    cons          : Vec<Item>,

    /// Cone definitions. Each cone consists of a type and a dimension
    cones         : Vec<(VecConeType,usize)>,
    /// Conic scalar elements. Each element corresponds to a single element in a single cone in
    /// `cones`. 
    cone_elt      : Vec<ConeElement>,

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
            ..Default::default()
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

        self.var_elt.resize(last,Element::Linear { lb: f64::NEG_INFINITY, ub: f64::INFINITY });
        self.var_int.resize(last,false);
        self.var_names.resize(last,None);

        let firstvari = self.vars.len();
        self.vars.reserve(n);
        for i in first..last {
            self.vars.push(Item::Linear{index:i,kind:LinearItem::Linear});
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
         name : Option<&str>,
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
        for i in first..last { self.vars.push(Item::Linear{index:i,kind:LinearItem::Linear}) }
        match dt {
            LinearDomainType::Zero => {
                self.var_elt.resize(last,Element::Linear { lb: 0.0, ub: 0.0 });
            },
            LinearDomainType::Free => {
                self.var_elt.resize(last,Element::Linear { lb: f64::NEG_INFINITY, ub: f64::INFINITY});
            },
            LinearDomainType::NonNegative => {
                self.var_elt.reserve(last);
                for lb in b {
                    self.var_elt.push(Element::Linear { lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.var_elt.reserve(last);
                for ub in b {
                    self.var_elt.push(Element::Linear{ lb : f64::NEG_INFINITY, ub });
                }
            },
        }
        self.var_int.resize(last,is_integer);

        if let Some(name) = name {
            let mut name_index_buf = [1usize; N];
            let mut strides = [0;N];
            strides.iter_mut().zip(shape.iter()).rev().fold(1,|s,(st,&d)| { *st = s; d * s });
            if let Some(sp) = &sp {
                for &i in sp.iter() {
                    name_index_buf.iter_mut().zip(strides.iter()).fold(i,|i,(ni,&st)| { *ni = i/st; i%st });
                    self.var_names.push(Some(format!("{}{:?}", name, name_index_buf)));
                }
            }
            else {
                for _ in 0..n {
                    name_index_buf.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                    self.var_names.push(Some(format!("{}{:?}", name, name_index_buf)));
                }
            }
        }
        else {
            for _ in 0..n {
                self.var_names.push(None);
            }
        }

        Ok(Variable::new((firstvari..firstvari+n).collect::<Vec<usize>>(), sp, &shape))
    }
    
    fn ranged_variable<const N : usize,R>(&mut self, name : Option<&str>,dom : LinearRangeDomain<N>) -> Result<<LinearRangeDomain<N> as VarDomainTrait<Self>>::Result,String> 
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
        for index in first..last { self.vars.push(Item::Linear{index,kind:LinearItem::RangedLower}) }
        for index in first..last { self.vars.push(Item::Linear{index,kind:LinearItem::RangedUpper}) }

        self.var_elt.reserve(last);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.var_elt.push(Element::Linear { lb, ub })
        }
        self.var_int.resize(last,is_integer);
        
        if let Some(name) = name {
            let mut name_index_buf = [1usize; N];
            let mut strides = [0;N];
            strides.iter_mut().zip(shape.iter()).rev().fold(1,|s,(st,&d)| { *st = s; d * s });
            if let Some(sp) = &sp {
                for &i in sp.iter() {
                    name_index_buf.iter_mut().zip(strides.iter()).fold(i,|i,(ni,&st)| { *ni = i/st; i%st });
                    self.var_names.push(Some(format!("{}{:?}", name, name_index_buf)));
                }
            }
            else {
                for _ in 0..n {
                    name_index_buf.iter_mut().zip(shape.iter()).rev().fold(1,|c,(i,&d)| { *i += c; if *i > d { *i = 1; 1 } else { 0 } });
                    self.var_names.push(Some(format!("{}{:?}", name, name_index_buf)));
                }
            }
        }
        else {
            for _ in 0..n {
                self.var_names.push(None);
            }
        }

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

        if sp.is_some() { return Err("Sparse domain on costraint not allowd".to_string()); }
        assert_eq!(b.len(),ptr.len()-1); 
        assert_eq!(ptr.len(),shape.iter().product::<usize>()+1);
        let n = b.len();

        let rowidxs = self.mx.append_rows(ptr,subj,cof);

        let con0 = self.cons.len();
        self.cons.reserve(n); for i in rowidxs.clone() { self.cons.push(Item::Linear { index: i, kind: LinearItem::Linear }) }
        
        match dt {
            LinearDomainType::Zero => {
                self.con_elt.reserve(rowidxs.end);
                for b in b {
                    self.con_elt.push(Element::Linear{ lb: b, ub: b });
                }
            },
            LinearDomainType::Free => { 
                self.con_elt.resize(rowidxs.end,Element::Linear{ lb: f64::NEG_INFINITY, ub: f64::INFINITY });
            },
            LinearDomainType::NonNegative => {
                self.con_elt.reserve(rowidxs.end);
                for lb in b {
                    self.con_elt.push(Element::Linear{ lb, ub: f64::INFINITY });
                }
            },
            LinearDomainType::NonPositive => {
                self.con_elt.reserve(rowidxs.end);
                for ub in b {
                    self.con_elt.push(Element::Linear{ lb : f64::NEG_INFINITY, ub });
                }
            },
        }

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
        let (shape,bl,bu,sp,_) = dom.dissolve();
        if sp.is_some() { return Err("Sparse domain on costraint not allowd".to_string()); }
        assert_eq!(bl.len(),ptr.len()-1); 
        assert_eq!(bu.len(),ptr.len()-1); 
        assert_eq!(ptr.len(),shape.iter().product::<usize>()+1);

        let rowidxs = self.mx.append_rows(ptr,subj,cof);
        let n = rowidxs.len();

        self.con_elt.reserve(rowidxs.end);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.con_elt.push(Element::Linear{ lb, ub });
        }

        let con0 = self.cons.len();
        self.cons.reserve(n*2);
        for index in rowidxs.clone() { self.cons.push(Item::Linear{ index, kind:LinearItem::RangedLower }); }
        for index in rowidxs.clone() { self.cons.push(Item::Linear{ index, kind:LinearItem::RangedUpper }); }
        
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
        Ok((Constraint::new((con0..con0+n).collect::<Vec<usize>>(), &shape),
            Constraint::new((con0+n..con0+2*n).collect::<Vec<usize>>(), &shape)))
    }

    fn update(& mut self, rowidxs : &[usize], shape : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) -> Result<(),String>
    {
        if shape.iter().product::<usize>() != rowidxs.len() { return Err("Mismatching constraint and experssion sizes".to_string()); }

        if let Some(&i) = rowidxs.iter().max() {
            if i >= self.cons.len() {
                return Err("Constraint index out of bounds".to_string());
            }
        }

        self.mx.replace_rows(rowidxs,ptr,subj,cof);

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
                let mut resp = client.post(url)
                    .header("Content-Type", "application/x-mosek-jtask")
                    .header("Accept", "application/x-mosek-multiplex")
                    .header("X-Mosek-Callback", "values")
                    .header("X-Mosek-Stream", "log")
                    .body(reqwest::blocking::Body::new(req_r))
                    .send()
                    .map_err(|e| e.to_string())?
                    ;

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


                _ = resp.copy_to(&mut resp_w).map_err(|e| e.to_string())?;

                Ok(())
            });

            self.write_jtask(&mut req_w).map_err(|e| e.to_string())?;
            drop(req_w);


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
                    self.s.read_exact(&mut buf)?;

                    self.final_frame = buf[0] > 127;
                    self.frame_remains = (((buf[0] & 0x7f) as usize) << 8) | (buf[1] as usize);
                    //println!("MessageReader::read() new frame : {}",self.frame_remains);
                }

                let n = buf.len().min(self.frame_remains);
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
                    //println!("read line: {} bytes", n);
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
                            if n == self.var_elt.len() {
                                let mut xx : Vec<f64> = Vec::new(); xx.resize(n,0.0);
                                unsafe{xx.align_to_mut().1}.copy_from_slice(&xxbytes[..n*size_of::<f64>()]);

                                let mut solxx = Vec::new(); solxx.resize(self.vars.len(),0.0);
                                for (v,d) in self.vars.iter().zip(solxx.iter_mut()) {
                                    *d = 
                                        match v {
                                            Item::Linear{ index,.. } => xx[*index],
                                            Item::Conic { index } => xx[*index],
                                        };
                                }

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
                     numcon : usize,
                     pobj : f64,
                     dobj : f64,
                     _varsta : Vec<u8>,
                     xx : Option<Vec<f64>>,
                     sx : Option<(Vec<f64>,Vec<f64>)>,
                     _consta : Vec<u8>,
                     xc : Option<Vec<f64>>,
                     sc : Option<(Vec<f64>,Vec<f64>,Vec<f64>)>,
                     sn : Option<Vec<f64>>,
                     sol : & mut Solution) -> std::io::Result<()>
    {
        let pdef = if let SolutionStatus::Undefined = psta { false } else { true };
        let ddef = if let SolutionStatus::Undefined = dsta { false } else { true };

        sol.primal.status = psta;
        sol.dual.status = dsta;

        if pdef {
            sol.primal.obj = pobj;
            if let Some(xx) = xx {
                if xx.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/var/primal")); }
                for (solx,e) in sol.primal.var.iter_mut().zip(self.vars.iter()) {
                    *solx = 
                       match e {
                           Item::Linear{ index,.. } => xx[*index],
                           Item::Conic { index } => xx[*index],
                        };
                }
            }
            else if numvar > 0 {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }


            if let Some(xc) = xc {
                if xc.len() != self.con_elt.iter().filter(|e| matches!(e,Element::Linear{..})).count() {
                    return Err(std::io::Error::other("Incorrect solution dimension in sol/var/primal")); 
                }
                for (i,solx,e) in izip!(0..,sol.primal.con.iter_mut(),self.cons.iter()) {
                    *solx = 
                        match e {
                           Item::Linear{ index,.. } => xc[*index],
                           Item::Conic { index } => {
                               let (subj,cof) = self.mx.get(i).unwrap();
                               sol.primal.var.permute_by(subj).zip(cof.iter()).map(|(a,b)| *a * *b).sum()
                           }
                        };
                }
            }
            else if numcon > 0 {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
        }
        
        if ddef {
            let numaccelm : usize = self.cones.iter().map(|c| c.1).sum();
            if let Some(sn) = sn.as_ref() {
                if numaccelm != sn.len() {
                    return Err(std::io::Error::other("Incorrect solution dimension in sol/acc/primal"));
                }
            }
            sol.dual.obj = dobj;
            if let Some((sl,su)) = sx {
                if sl.len() != numvar || su.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/var/dual")); }
                else { 
                    for (solx,e) in sol.dual.var.iter_mut().zip(self.vars.iter()) {
                        *solx = 
                            match e {
                               Item::Linear{ index, kind : LinearItem::Linear }      => sl[*index]-su[*index],
                               Item::Linear{ index, kind : LinearItem::RangedUpper } => -su[*index],
                               Item::Linear{ index, kind : LinearItem::RangedLower } =>  sl[*index],
                               Item::Conic { index } =>  sn.as_ref().map(|sn| sn[*index]).unwrap_or(0.0),
                            };
                    }
                }
            }
            else if numvar > 0 {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
            
            if let Some((sl,su,y)) = sc {
                if sl.len() != numvar || su.len() != numvar || y.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/con/primal")); }
                else {
                    for (solx,e) in sol.dual.con.iter_mut().zip(self.cons.iter()) {
                        *solx = match e {
                           Item::Linear{ index, kind : LinearItem::Linear }      =>   y[*index],
                           Item::Linear{ index, kind : LinearItem::RangedUpper } => -su[*index],
                           Item::Linear{ index, kind : LinearItem::RangedLower } =>  sl[*index],
                           Item::Conic { index } => sn.as_ref().map(|sn| sn[*index]).unwrap_or(0.0),
                        };
                    }
                }
            }
            else if numcon > 0 {
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

        if self.var_elt.len() != numvar { return Err(std::io::Error::other("Invalid solution dimension")); }
        if self.con_elt.len() != numcon { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numcone { return Err(std::io::Error::other("Invalid solution dimension")); }
        if self.cones.len() != numacc { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numbarvar { return Err(std::io::Error::other("Invalid solution dimension")); }

        let mut entry = r.next_entry()?.ok_or_else(|| std::io::Error::other("Missing solution entries"))?;
 
        use SolutionStatus::*;

        if entry.name() == b"sol/interior" {
            sol_itr.primal.var.resize(self.vars.len(),0.0);
            sol_itr.dual.var.resize(self.vars.len(),0.0);
            sol_itr.primal.con.resize(self.cons.len(),0.0);
            sol_itr.dual.con.resize(self.cons.len(),0.0);

            let sta       = entry.check_fmt(b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef = !matches!(sta.0,Undefined);
            let ddef = !matches!(sta.1,Undefined);
            let pobj = if pdef { r.expect(b"sol/interior/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let dobj = if ddef { r.expect(b"sol/interior/dobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let varsta = r.expect(b"sol/interior/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx = if pdef { Some(r.expect(b"sol/interior/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sx = if ddef { Some(r.expect(b"sol/interior/var/dual",b"[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let consta = r.expect(b"sol/interior/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xc = if numcon > 0 && pdef { Some(r.expect(b"sol/interior/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sc = if numcon > 0 && ddef { Some(r.expect(b"sol/interior/con/dual",b"[d[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let sn = if numacc > 0 && ddef { Some(r.expect(b"sol/interior/acc/dual",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };

            self.copy_solution(sta.0,sta.1,
                               numvar, numcon, 
                               pobj,dobj,
                               varsta,xx,sx,
                               consta,xc,sc,
                               sn,
                               sol_itr)?;
            entry = r.next_entry()?.ok_or_else(|| std::io::Error::other("Missing solution entries"))?;
        }

        if entry.name() == b"sol/basic" {
            let sta       = entry.check_fmt(b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef = !matches!(sta.0,Undefined);
            let ddef = !matches!(sta.1,Undefined);
            let pobj : f64 = if pdef { r.expect(b"sol/basic/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let dobj : f64 = if ddef { r.expect(b"sol/basic/dobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let varsta = r.expect(b"sol/basic/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx = if pdef { Some(r.expect(b"sol/basic/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sx = if ddef { Some(r.expect(b"sol/basic/var/dual",b"[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let consta = r.expect(b"sol/basic/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xc = if numcon > 0 && pdef { Some(r.expect(b"sol/basic/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let sc = if numcon > 0 && ddef { Some(r.expect(b"sol/basic/con/dual",b"[d[d[d").and_then(|mut entry| Ok((entry.read::<f64>()?,entry.read::<f64>()?,entry.read::<f64>()?)))?) } else { None };
            let sn = if numacc > 0 && ddef { Some(r.expect(b"sol/basic/acc/dual",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };

            self.copy_solution(sta.0,sta.1,
                               numvar, numcon, 
                               pobj,dobj,
                               varsta,xx,sx,
                               consta,xc,sc,
                               sn,
                               sol_bas)?;
            entry = r.next_entry()?.ok_or_else(|| std::io::Error::other("Missing solution entries"))?;
        }

        if entry.name() == b"sol/integer" {
            let sta       = entry.check_fmt(b"[B[B").and_then(|mut entry| { entry.skip_field()?; Ok(str_to_pdsolsta(entry.read::<u8>()?.as_slice())?) })?;
            let pdef = !matches!(sta.0,Undefined);
            let pobj = if pdef { r.expect(b"sol/integer/pobj",b"d").and_then(|mut entry| entry.next_value::<f64>())? } else { 0.0 };
            let varsta = r.expect(b"sol/integer/var/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xx = if pdef { Some(r.expect(b"sol/integer/var/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            let consta = r.expect(b"sol/integer/con/sta",b"[B").and_then(|mut entry| Ok(entry.read::<u8>()?))?;
            let xc = if numcon > 0 && pdef { Some(r.expect(b"sol/integer/con/primal",b"[d").and_then(|mut entry| Ok(entry.read::<f64>()?))?) } else { None };
            
            self.copy_solution(sta.0,sta.1,
                               numvar, numcon, 
                               pobj,0.0,
                               varsta,xx,None,
                               consta,xc,None,
                               None,
                               sol_itg)?;
            entry = r.next_entry()?.ok_or_else(|| std::io::Error::other("Missing solution entries"))?;
        }

        if entry.name() != b"name/var" || entry.fmt() != b"[I[B" { 
            return Err(std::io::Error::other(format!("Expected section 'name/var'/'[I[B', got '{}'/'{}'",
                                                     std::str::from_utf8(entry.name()).unwrap_or("<invalid utf-8>"),
                                                     std::str::from_utf8(entry.fmt()).unwrap_or("<invalid utf-8>")))); }
        entry.skip_all()?;
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
                taskinfo.append("numcon",self.con_elt.len() as i64);
            }));
            
        doc.append(
            "Task/data",
            json::Dict::from(|taskdata| {
                {
                    let mut bl = Vec::with_capacity(self.var_elt.len());
                    let mut bu = Vec::with_capacity(self.var_elt.len());
                    let mut bk = Vec::with_capacity(self.var_elt.len());

                    self.var_elt.iter().map(|e|
                        match e {
                            Element::Linear { lb, ub } => (bnd_to_bk(*lb, *ub), *lb,*ub),
                            Element::Conic { .. } => ("fr",f64::NEG_INFINITY, f64::INFINITY),
                        }).for_each(|v| { bk.push(v.0.into()); bl.push(v.1); bu.push(v.2); });

                    taskdata.append("var",json::Dict::from(|d| {
                        d.append("bk",  JSON::StringArray(bk));
                        d.append("bl",  JSON::FloatArray(bl));
                        d.append("bu",  JSON::FloatArray(bu));
                        d.append("type",JSON::StringArray(self.var_int.iter().map(|&e| if e { "int".into() } else { "cont".into() }).collect()));
                    }));
                }
                {
                    let mut bl = Vec::with_capacity(self.con_elt.len());
                    let mut bu = Vec::with_capacity(self.con_elt.len());
                    let mut bk = Vec::with_capacity(self.con_elt.len());

                    self.con_elt.iter().map(|e|
                        match e {
                            Element::Linear { lb, ub } => (bnd_to_bk(*lb, *ub), *lb,*ub),
                            Element::Conic { .. } => ("fr",f64::NEG_INFINITY, f64::INFINITY),
                        }).for_each(|v| { bk.push(v.0.into()); bl.push(v.1); bu.push(v.2); });

                    taskdata.append("con",json::Dict::from(|d| {
                        d.append("bk",  JSON::StringArray(bk));
                        d.append("bl",  JSON::FloatArray(bl));
                        d.append("bu",  JSON::FloatArray(bu));
                    }));
                }
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
                        let mut subi = Vec::new();
                        let mut subj = Vec::new();
                        let mut val  = Vec::new();

                        self.mx.row_iter().zip(self.con_elt.iter())
                            .filter_map(|(item,e)| if matches!(e,Element::Linear {..}) { Some(item) } else { None })
                            .enumerate()
                            .for_each(|(i,(jj,cc))| {
                                let base = subi.len();
                                let n = jj.len();
                                subi.resize(base+n, i as i64);
                                subj.resize(base+n,0); subj[base..].iter_mut().zip(jj.iter()).for_each(|(dst,src)| *dst = *src as i64);
                                val.extend_from_slice(cc);
                            });
                        d.append("subi",JSON::IntArray(subi));
                        d.append("subj",JSON::IntArray(subj));
                        d.append("val", JSON::FloatArray(val));
                }));
                taskdata.append(
                    "AFE",
                    json::Dict::from(|d| {
                        let mut subi = Vec::new();
                        let mut subj = Vec::new();
                        let mut val  = Vec::new();

                        self.mx.row_iter().zip(self.con_elt.iter())
                            .filter_map(|(item,e)| if matches!(e,Element::Conic {..}) { Some(item) } else { None })
                            .enumerate()
                            .for_each(|(i,(jj,cc))| {
                                let base = subi.len();
                                let n = jj.len();
                                subi.resize(base+n, i as i64);
                                subj.resize(base+n,0); subj[base..].iter_mut().zip(jj.iter()).for_each(|(dst,src)| *dst = *src as i64);
                                val.extend_from_slice(cc);
                            });
                        d.append("numafe", JSON::Int( self.cone_elt.len() as i64));

                        d.append("F", json::Dict::from(|d| {
                            d.append("subi",JSON::IntArray(subi));
                            d.append("subj",JSON::IntArray(subj));
                            d.append("val", JSON::FloatArray(val));
                        }));
                    }));
                taskdata.append(
                    "domains",
                    json.List(self.cones.iter()
                              .map(|c| {
                                  use VecConeType::*;
                                  match c.0 {
                                    Zero             => JSON::List(vec![ JSON::String("zero".to_string()), JSON::Int(c.1 as i64)]),
                                    NonNegative      => JSON::List(vec![ JSON::String("rplus".to_string()), JSON::Int(c.1 as i64)]),
                                    NonPositive      => JSON::List(vec![ JSON::String("rminus".to_string()), JSON::Int(c.1 as i64)]),
                                    Free             => JSON::List(vec![ JSON::String("r".to_string()), JSON::Int(c.1 as i64)]),
                                    Quadratic        => JSON::List(vec![ JSON::String("quad".to_string()), JSON::Int(c.1 as i64)]),
                                    RotatedQuadratic => JSON::List(vec![ JSON::String("rquad".to_string()), JSON::Int(c.1 as i64)]),
                                    PrimalGeometricMean  => JSON::List(vec![ JSON::String("".to_string()), JSON::Int(c.1 as i64)]),
                                    DualGeometricMean  => JSON::List(vec![ JSON::String("".to_string()), JSON::Int(c.1 as i64)]),
                                  }}).collect())
                        ));
                taskdata.append(
                    "ACC",
                    json::Dict::from(|d| {}));
                         
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


mod msto {
    use itertools::izip;
    use mosekcomodel::utils::iter::{ChunksByIterExt, Permutation, PermuteByMutEx};

    #[derive(Default)]
    pub struct MatrixStore {
        ptr  : Vec<usize>,
        len  : Vec<usize>,
        subj : Vec<usize>,
        cof  : Vec<f64>,

        map  : Vec<usize>
    }

    impl MatrixStore {
        pub fn new() -> MatrixStore { Default::default() }
        pub fn append_row(&mut self, subj : &[usize], cof : &[f64]) -> usize {        
            assert_eq!(subj.len(),cof.len());
            self.len.push(subj.len());
            self.subj.extend_from_slice(subj);
            self.cof.extend_from_slice(cof);
            
            let res = self.map.len();
            self.map.push(self.ptr.len());
            self.ptr.push(self.subj.len());
            res
        }

        pub fn append_rows(&mut self, ptr : &[usize], subj : &[usize], cof : &[f64]) -> std::ops::Range<usize> {
            assert_eq!(subj.len(),cof.len());
            assert!(ptr.iter().zip(ptr[1..].iter()).all(|(a,b)| *a <= *b));
            assert_eq!(*ptr.last().unwrap(),subj.len());
            let len0 = self.subj.len();
            self.subj.extend_from_slice(subj);
            self.cof.extend_from_slice(cof);
            
            let row0 = self.map.len();
            for i in self.ptr.len()..self.ptr.len()+ptr.len()-1 { self.map.push(i); }
            let row1 = self.map.len();
                       
            for (p,l) in ptr.iter().zip(ptr[1..].iter()).scan(len0,|len,(p0,p1)| { let l = *len; *len = p1-p0; Some((l,p1-p0)) }) {
                self.ptr.push(p);
                self.len.push(l);
            }

            row0..row1
        }

        pub fn get<'a>(&'a self, i : usize) -> Option<(&'a [usize],&'a [f64])> {
            self.map.get(i)
                .map(|&i| {
                    let p = unsafe{*self.ptr.get_unchecked(i)};
                    let l = unsafe{*self.len.get_unchecked(i)};
                    
                    (unsafe{self.subj.get_unchecked(p..p+l)},
                     unsafe{self.cof.get_unchecked(p..p+l)})
                })
        }
        pub fn get_mut<'a>(&'a mut self, i : usize) -> Option<(&'a mut[usize],&'a mut [f64])> {
            self.map.get(i)
                .map(|&i| {
                    let p = unsafe{*self.ptr.get_unchecked(i)};
                    let l = unsafe{*self.len.get_unchecked(i)};
                    
                    (unsafe{self.subj.get_unchecked_mut(p..p+l)},
                     unsafe{self.cof.get_unchecked_mut(p..p+l)})
                })
        }

        pub fn replace_rows(&mut self, rows : &[usize], ptr : &[usize], subj : &[usize], cof : &[f64]) {
            if !rows.is_empty() {
                assert_eq!(subj.len(),cof.len());
                assert_eq!(ptr.len(),rows.len()+1);
                assert!(ptr.iter().zip(ptr[1..].iter()).all(|(a,b)| *a <= *b));
                assert_eq!(*ptr.last().unwrap(),subj.len());
                assert!(*rows.iter().max().unwrap() < self.map.len());

                for (rowi,subj,cof) in izip!(self.map.permute_by_mut(rows),subj.chunks_ptr(ptr),cof.chunks_ptr(ptr)) {
                    let leni = unsafe{self.len.get_unchecked_mut(*rowi)};
                    let ptri = unsafe{self.ptr.get_unchecked(*rowi)};
                    if subj.len() <= *leni {
                        *leni = subj.len();
                        self.subj[*ptri..*ptri+*leni].copy_from_slice(subj);
                        self.cof[*ptri..*ptri+*leni].copy_from_slice(cof);
                    }
                    else {
                        *rowi = self.len.len();
                        self.len.push(subj.len());
                        self.ptr.push(self.subj.len());
                        self.subj.extend_from_slice(subj);
                        self.cof.extend_from_slice(cof);
                    }
                }
            }
        }

        pub fn row_iter<'a>(&'a self) -> impl Iterator<Item=(&'a [usize],&'a[f64])> {
            let perm = Permutation::new(self.map.as_slice());

            perm.permute(self.ptr.as_slice()).unwrap()
                .zip(perm.permute(self.len.as_slice()).unwrap()) .map(|(p,l)| {
                    (unsafe{self.subj.get_unchecked(*p..*p+*l)},
                     unsafe{self.cof.get_unchecked(*p..*p+*l)})
                })
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_optserver() {
        let addr = "http://solve.mosek.com:30080".to_string();
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
