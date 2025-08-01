use std::{io::{Read, Write}, ptr::fn_addr_eq};


#[derive(Debug)]
pub struct Dict(pub Vec<(String,JSON)>);
impl Dict {
    pub fn new() -> Dict { Dict(Vec::new()) }
    pub fn append<K,V>(&mut self, k : K,v : V) where K : Into<String>, V : Into<JSON> { self.0.push((k.into(),v.into())) }
    pub fn from<F>(build : F) -> Dict where F : FnOnce(&mut Dict) {
        let mut d = Dict::new();
        build(&mut d);
        d
    }
}

#[derive(Debug)]
pub enum JSON {
    String(String),
    Int(i64),
    Float(f64),
    List(Vec<JSON>),
    Dict(Dict),
    Null
}

fn hexbyte_l(c : u8) -> u8 {
    match c & 0xf {
        c @ 0..=9 => b'0' + c,
        c => b'a' + c - 10
    }
}
fn hexbyte_u(c : u8) -> u8 { hexbyte_l(c >> 4) } 

fn byte_from_hex(c : u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c-b'0'),
        b'a'..=b'f' => Some(c-b'a'+10),
        b'A'..=b'F' => Some(c-b'A'+10),
        _ => None
    }
}

impl Into<JSON> for &str {
    fn into(self) -> JSON { JSON::String(self.to_string()) }
}

impl Into<JSON> for String {
    fn into(self) -> JSON { JSON::String(self) }
}

impl Into<JSON> for i64 {
    fn into(self) -> JSON { JSON::Int(self) }
}

impl Into<JSON> for f64 {
    fn into(self) -> JSON { JSON::Float(self) }
}
impl<T> Into<JSON> for &[T] where T : Into<JSON>+Copy {
    fn into(self) -> JSON { JSON::List(self.iter().map(|&v| v.into()).collect()) }
}
impl<T> Into<JSON> for Vec<T> where T : Into<JSON>+Copy {
    fn into(self) -> JSON { self.as_slice().into() }
}
impl Into<JSON> for Dict {
    fn into(self) -> JSON { JSON::Dict(self) }
}

impl TryFrom<&JSON> for String {
    type Error = ();
    fn try_from(value: &JSON) -> Result<Self, Self::Error> {
        if let JSON::String(s) = value { Ok(s.clone()) } else { Err(()) }
    }
}

impl TryFrom<&JSON> for usize {
    type Error = ();
    fn try_from(value: &JSON) -> Result<Self, Self::Error> {
        if let JSON::Int(s) = value { Ok((*s).try_into().map_err(|_|())?) } else { Err(()) }
    }
}

impl TryFrom<&JSON> for f64 {
    type Error = ();
    fn try_from(value: &JSON) -> Result<Self, Self::Error> {
        match value {
            JSON::Float(v) => Ok((*v).try_into().map_err(|_| ())?),
            JSON::Int(v) => Ok((*v) as f64),
            _ => Err(())
        }
    }
}

impl<'a,T> TryFrom<&'a JSON> for Vec<T> where T : TryFrom<&'a JSON> {
    type Error = ();
    fn try_from(value: &'a JSON) -> Result<Self, Self::Error> {
        if let JSON::List(l) = value {
            let mut res = Vec::new();
            
            for item in l.iter() {
                if let Ok(v) = item.try_into() {
                    res.push(v);
                }
                else {
                    return Err(());
                }
            }
            Ok(res)
        }
        else {
            Err(())
        }
    }
}


impl JSON {
    fn write_str<T>(s : & mut T, value : &str) -> std::io::Result<()> where T : Write {
        s.write_all(b"\"")?;
        for c in value.as_bytes().chunk_by(|&b0,&b1| b0 < 32 || b1 < 32 || b0 == b'"' || b1 == b'"') {
            match c {
                [b'"']  => s.write_all(b"\\\"")?,
                [b'\\'] => s.write_all(b"\\\\")?,
                [b'\r'] => s.write_all(b"\\r")?,
                [b'\n'] => s.write_all(b"\\n")?,
                [b'\t'] => s.write_all(b"\\t")?,
                [b@(0..32)]     => s.write_all(&[b'\\',b'x',hexbyte_u(*b),hexbyte_l(*b)])?,
                _ => _ = s.write(c)?
            }
        }
        s.write_all(b"\"")?;
        Ok(())
    }

    pub fn write<T>(&self, s : & mut T) -> std::io::Result<()> where T : Write {
        match self {
            JSON::String(value) => Self::write_str(s,value.as_str())?,
            JSON::Int(value)    => write!(s,"{}",value)?,
            JSON::Float(value)  => {
                if value.is_infinite() {
                    if *value < 0.0 { s.write_all(b"-1e500")?; }
                    else { s.write_all(b"1e500")?; } 
                }
                else {
                    write!(s,"{}",value)?;
                }
            },
            JSON::List(l)   => {
                s.write_all(b"[")?;
                let mut it = l.iter();
                if let Some(item) = it.next() {
                    item.write(s)?;
                    for item in it {
                        s.write_all(b",")?;
                        item.write(s)?;
                    }
                }
                s.write_all(b"]")?;
            },
            JSON::Dict(d) => { 
                s.write_all(b"{")?;
                let mut it = d.0.iter();
                if let Some((k,v)) = it.next() {
                    Self::write_str(s,k.as_str())?;
                    s.write_all(b":")?;
                    v.write(s)?;

                    for (k,v) in it {
                        s.write_all(b",")?;
                        Self::write_str(s,k.as_str())?;
                        s.write_all(b":")?;
                        v.write(s)?;
                    }
                }
                s.write_all(b"}")?;
            },
            JSON::Null => write!(s,"null")?
        }
        Ok(())
    }


    fn parse_number<'a,T>(s : &mut PeekReader<'a,T>, neg : bool) -> std::io::Result<JSON> where T : Read {
        let mut res = Vec::new();

        match s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))? {
            b @ (b'0'..=b'9') => {
                res.push(b);
                while let Some(c) = s.peek()? {
                    match c {
                        b @ (b'0'..=b'9'|b'.'|b'e'|b'E'|b'+'|b'-') => {
                            _ = s.get();
                            res.push(b);
                        },
                        _ => break
                    }
                }
                
                let res = std::str::from_utf8(res.as_slice()).map_err(|_| std::io::Error::other("Invalid JSON numeric format"))?;
                if let Ok(v) = res.parse::<i64>()  {
                    Ok(JSON::Int(if neg { -v } else { v }))
                }
                else {
                    let v = res.parse::<f64>().map_err(|_| std::io::Error::other("Invalid JSON numeric format"))?;
                    Ok(JSON::Float(if neg { -v } else { v }))
                }
            }, 
            _ => Err(std::io::Error::other("Invalid JSON integer syntax"))
        }
    }
    fn parse_string<'a,T>(s : &mut PeekReader<'a,T>) -> std::io::Result<String> where T : Read {
        let c0 = s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))?;
        if c0 != b'"' {
            Err(std::io::Error::other("Invalid JSON integer syntax"))
        }
        else {
            let mut res = Vec::new();
            loop {
                match s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))? {
                    b'"' => break,
                    b'\\' => {
                        match s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))? {
                            b'r'  => res.push(b'\r'),
                            b'n'  => res.push(b'\n'),
                            b't'  => res.push(b'\t'),
                            b'\\' => res.push(b'\\'),
                            b'x'  => {
                                let mut buf = [0;2];
                                for b in buf.iter_mut() {
                                    *b = byte_from_hex(s.get_expect()?).ok_or_else(|| std::io::Error::other("JSON string syntax error"))?;
                                }
                                res.push((buf[0] << 4) + buf[1]);
                            },
                            b => res.push(b),
                        }
                    },                            
                    b => res.push(b)
                }
            }
            Ok(std::str::from_utf8(res.as_slice()).map_err(|e| std::io::Error::other(e.to_string()))?.to_string())
        }
    }

    fn skip_space<'a,T>(s : &mut PeekReader<'a,T>) -> std::io::Result<()> where T : Read {
        while let Some(c) = s.peek()? {
            match c {
                b' '|b'\t'|b'\r'|b'\n' => _ = s.get(), 
                _ => break,
            }
        }
        Ok(())
    }

    fn read_hlp<'a,T>(s : &mut PeekReader<'a,T>) -> std::io::Result<JSON> where T : Read {
        Self::skip_space(s)?;
        match s.peek_expect()? {
            b'"' => { 
                let s = Self::parse_string(s)?;
                Ok(JSON::String(s))
            },
            b'[' => {
                _ = s.get();
                let mut res = Vec::new();

                'outer: loop {
                    match s.peek_expect()? {
                        b' '|b'\t'|b'\r'|b'\n' => _ = s.get(), 
                        b']' => {
                            _ = s.get();
                            break;
                        }
                        _ => {
                            res.push(Self::read_hlp(s)?);

                            loop {
                                match s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))? { 
                                    b' '|b'\t'|b'\r'|b'\n' => {},
                                    b']' => break 'outer,
                                    b',' => { 
                                        res.push(Self::read_hlp(s)?);
                                    },
                                    _ => return Err(std::io::Error::other("Syntax error in JSON list"))
                                }
                            }
                        }
                    }
                }

                Ok(JSON::List(res))
            },
            b'{' => { 
                _ = s.get();
                let mut res = Dict::new();
                
                Self::skip_space(s)?;

                match s.peek()?.ok_or_else(|| std::io::Error::other("JSON syntax error: Expected item"))? {
                    b'}' => {
                        _ = s.get();
                    },
                    _ => {
                        let k = Self::parse_string(s)?;
                        Self::skip_space(s)?;
                        match s.get_expect()? { 
                            b':' => { }
                            _ => return Err(std::io::Error::other("Syntax error in JSON dictionary"))
                        }
                        let v = Self::read_hlp(s)?;

                        res.append(k,v);

                        loop {
                            Self::skip_space(s)?;
                            match s.get_expect()? {
                                b'}' => break,
                                b',' => {
                                    Self::skip_space(s)?;
                                    let k = Self::parse_string(s)?;
                                    Self::skip_space(s)?;
                                    match s.get_expect()? { 
                                        b':' => { }
                                        _ => return Err(std::io::Error::other("Syntax error in JSON dictionary"))
                                    }
                                    let v = Self::read_hlp(s)?;

                                    res.append(k,v);
                                },
                                _ => return Err(std::io::Error::other("Syntax error in JSON dictionary"))
                            }
                        }
                    }
                }
                Ok(JSON::Dict(res))
            },
            b'-' => Self::parse_number(s,true),
            b'0'..=b'9' => Self::parse_number(s,false),
            b'n' => {
                _ = s.get();
                let b1 = s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error"))?;
                let b2 = s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error"))?;
                let b3 = s.get()?.ok_or_else(|| std::io::Error::other("JSON syntax error"))?;

                if b1 ==  b'u' && b2 == b'l' && b3 == b'l' {
                    Ok(JSON::Null)
                }
                else {
                    Err(std::io::Error::other("Invalid JSON syntax"))
                }
            },
            _ => Err(std::io::Error::other("Invalid JSON syntax")),
        }
    }


    pub fn read<T>(s : &mut T) -> std::io::Result<JSON> where T : Read {
        let mut r = PeekReader::new(s);
        Self::skip_space(&mut r)?;
        match r.peek_expect()? {
            b'{'|b'[' => Self::read_hlp(&mut r),
            b => Err(std::io::Error::other(format!("Invalid JSON top-level item: '{}'",b as char)))
        }
    }
}

struct PeekReader<'a,T> where T : std::io::Read {
    b : Option<u8>,
    s : &'a mut T
}

impl<'a,T> PeekReader<'a,T> where T : std::io::Read {
    fn new(s : &'a mut T) -> PeekReader<'a,T> { PeekReader { b: None, s }}
    fn peek(&mut self) -> std::io::Result<Option<u8>> {
        if let Some(c) = self.b {
            Ok(Some(c))
        }
        else {
            let mut buf = [0;1];
            self.s.read_exact(&mut buf).map_err(|_| std::io::Error::other("Invalid JSON list syntax"))?;
            self.b = Some(buf[0]);
            Ok(Some(buf[0]))
        }
    }
    fn get(&mut self) -> std::io::Result<Option<u8>> {
        if let Some(c) = self.b {
            self.b = None;
            Ok(Some(c))
        }
        else {
            let mut buf = [0;1];
            self.s.read_exact(&mut buf).map_err(|_| std::io::Error::other("Invalid JSON list syntax"))?;
            Ok(Some(buf[0]))
        }
    }
    fn peek_expect(&mut self) -> std::io::Result<u8> {
        self.peek().and_then(|s| s.ok_or_else(|| std::io::Error::other("Premature end of JSON file")))
    }
    fn get_expect(&mut self) -> std::io::Result<u8> {
        self.get().and_then(|s| s.ok_or_else(|| std::io::Error::other("Premature end of JSON file")))
    }
}
