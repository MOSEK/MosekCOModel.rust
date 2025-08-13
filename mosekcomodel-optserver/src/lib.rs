//! This module implements a backend that uses a MOSEK OptServer instance for solving, for example
//! [solve.mosek.com:30080](http://solve.mosek.com). 
//!
use mosekcomodel::*;
use mosekcomodel::model::{IntSolutionManager, ModelWithLogCallback};
use mosekcomodel::utils::iter::{ChunksByIterExt, PermuteByEx};
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

    log_cb        : Option<Box<dyn Fn(&str)>>,
    sol_cb        : Option<Box<dyn FnMut(f64,&IntSolutionManager),

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

    address       : Option<reqwest::Url>,
    dpar : HashMap<String,f64>,
    ipar : HashMap<String,i32>,
}

impl BaseModelTrait for Backend {
    fn new(name : Option<&str>) -> Self {
        Backend{
            name : name.map(|v| v.to_string()),
            log_cb     : None,
            sol_cb     : None,
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

            address    : None,
            dpar : Default::default(),
            ipar : Default::default(),
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
        for i in first..last { self.vars.push(Item::RangedLower{index:i}) }
        for i in first..last { self.vars.push(Item::RangedUpper{index:i}) }

        self.var_elt.reserve(last);
        for (&lb,&ub) in bl.iter().zip(bu.iter()) {
            self.var_elt.push(Element{ lb, ub })
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


                let n = resp.copy_to(&mut resp_w).map_err(|e| e.to_string())?;

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
                                    match v {
                                        Item::RangedLower { index } => *d = xx[*index],
                                        Item::RangedUpper { index } => *d = xx[*index],
                                        Item::Linear { index } => *d = xx[*index],
                                    }                                
                                }

                                let c : f64 = self.c_cof.iter().zip(solxx.permute_by(self.c_subj.as_slice())).map(|(c,x)| *c * *x).sum();
                                let mut solxc = Vec::new(); solxc.resize(self.cons.len(),0.0);
                                for (xc,arow) in solxc.iter_mut().zip(self.a_ptr.permute_by(self.con_a_row.as_slice())) {
                                    *xc = self.a_cof[arow[0]..arow[0]+arow[1]].iter().zip(solxx.permute_by(&self.a_subj[arow[0]..arow[0]+arow[1]])).map(|(c,x)| *c * *x).sum(); 
                                }

                                if let Some(cb) = & mut self.sol_cb {
                                    cb(c,solxx.as_slice(),solxc.as_slice());
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
                     varsta : Vec<u8>,
                     xx : Option<Vec<f64>>,
                     sx : Option<(Vec<f64>,Vec<f64>)>,
                     consta : Vec<u8>,
                     xc : Option<Vec<f64>>,
                     sc : Option<(Vec<f64>,Vec<f64>,Vec<f64>)>,
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
                else {
                    for (solx,e) in sol.primal.var.iter_mut().zip(self.vars.iter()) {
                       match e {
                           Item::Linear      { index } => *solx = xx[*index],
                           Item::RangedUpper { index } => *solx = xx[*index],
                           Item::RangedLower { index } => *solx = xx[*index],
                        }
                    }
                }
            }
            else if numvar > 0 {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
            
            if let Some(xc) = xc {
                if xc.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/con/primal")); }
                else {
                    for (solx,e) in sol.primal.con.iter_mut().zip(self.cons.iter()) {
                       match e {
                           Item::Linear      { index } => *solx = xc[*index],
                           Item::RangedUpper { index } => *solx = xc[*index],
                           Item::RangedLower { index } => *solx = xc[*index],
                        }
                    }
                }
            }
            else if numcon > 0 {
                return Err(std::io::Error::other("Missing solution section sol/var/primal"));
            }
        }
        
        if ddef {
            sol.dual.obj = dobj;
            if let Some((sl,su)) = sx {
                if sl.len() != numvar || su.len() != numvar { return Err(std::io::Error::other("Incorrect solution dimension in sol/var/dual")); }
                else {
                    for (solx,e) in sol.dual.var.iter_mut().zip(self.vars.iter()) {
                       match e {
                           Item::Linear      { index } => *solx = sl[*index]-su[*index],
                           Item::RangedUpper { index } => *solx = -su[*index],
                           Item::RangedLower { index } => *solx = sl[*index],
                        }
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
                       match e {
                           Item::Linear      { index } => *solx = y[*index],
                           Item::RangedUpper { index } => *solx = -su[*index],
                           Item::RangedLower { index } => *solx = sl[*index],
                        }
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
        sol_bas.dual.status = Undefined;
        sol_itr.primal.status = Undefined;
        sol_itr.dual.status = Undefined;
        sol_itg.primal.status = Undefined;
        sol_itg.dual.status = Undefined;


        let numvar    = r.expect(b"numvar",b"I")?.next_value::<u32>()? as usize;
        let numbarvar = r.expect(b"numbarvar",b"I")?.next_value::<u32>()?;
        let numcon    = r.expect(b"numcon",b"I")?.next_value::<u32>()? as usize;
        let numcone   = r.expect(b"numcone",b"I")?.next_value::<u32>()?;
        let numacc    = r.expect(b"numacc",b"L")?.next_value::<u64>()?;

        if self.var_elt.len() != numvar { return Err(std::io::Error::other("Invalid solution dimension")); }
        if self.con_elt.len() != numcon { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numcone { return Err(std::io::Error::other("Invalid solution dimension")); }
        if 0 != numacc { return Err(std::io::Error::other("Invalid solution dimension")); }
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

            self.copy_solution(sta.0,sta.1,
                               numvar, numcon, 
                               pobj,dobj,
                               varsta,xx,sx,
                               consta,xc,sc,
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

            self.copy_solution(sta.0,sta.1,
                               numvar, numcon, 
                               pobj,dobj,
                               varsta,xx,sx,
                               consta,xc,sc,
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
    /// See https://docs.mosek.com/latest/capi/json-format.html
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
