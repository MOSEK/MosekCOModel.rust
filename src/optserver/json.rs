use std::{error::Error, io::{Read, Write}};


pub enum JSON {
    String(String),
    Int(i64),
    Float(f64),
    List(Vec<JSON>),
    Dict(Vec<(String,JSON)>),
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
                [b]     => s.write_all(&[b'\\',b'x',hexbyte_u(*b),hexbyte_l(*b)])?,
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
            JSON::Float(value)  => write!(s,"{}",value)?,
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
            JSON::Dict(l)   => { 
                s.write_all(b"{")?;
                let mut it = l.iter();
                if let Some((k,v)) = it.next() {
                    Self::write_str(s,k.as_str())?;
                    for (k,v) in it {
                        s.write_all(b",")?;
                        Self::write_str(s,k.as_str())?;
                    }
                }
                s.write_all(b"}")?;
            },
            JSON::Null      => write!(s,"null")?
        }
        Ok(())
    }

    fn read_hlp<T>(s : &mut T, mut c : Option<u8>) -> std::io::Result<(Option<JSON>,Option<u8>)> where T : Read {
        loop {
            let c = 
                if let Some(c) = c { c }
                else {
                    let mut buf = [0;1];
                    s.read_exact(&mut buf)?;
                    buf[0]
                };
            match c {
                b'"' => { 
                    let mut res = Vec::new();
                    loop {
                        let mut buf = [0;4];
                        s.read_exact(&mut buf[0..1])?;
                        match buf[0] {
                            b'"' => break,
                            b'\\' => {
                                s.read_exact(&mut buf[1..2])?;
                                match buf[1] {
                                    b'r'  => res.push(b'\r'),
                                    b'n'  => res.push(b'\n'),
                                    b't'  => res.push(b'\t'),
                                    b'\\' => res.push(b'\\'),
                                    b'x'  => {
                                        s.read_exact(&mut buf[2..4])?;
                                        let b0 = byte_from_hex(buf[2]).ok_or_else(|| Err(std::io::Error::other("JSON string syntax error"))).unwrap() << 4;
                                        let b1 = byte_from_hex(buf[3]).ok_or_else(|| Err(std::io::Error::other("JSON string syntax error"))).unwrap();
                                        res.push(b0 + b1);
                                    },
                                    _ => res.push(buf[1]),
                                }
                            },                            
                            _ => res.push(buf[0])
                        }
                    }
                    break Ok((Some(JSON::String(std::str::from_utf8(res.as_slice()).map_err(|e| std::io::Error::other(e.to_string()))?.to_string())),None))
                },
                b' '|b'\t'|b'\r'|b'\n' => {},
                b'[' => {
                    let mut res = Vec::new();
                    let mut buf = [0;1];
                        
                    let (item,c) = Self::read_hlp(s, Some(buf[0]))?;
                    if let Some(item) = item {
                        res.push(item);
                    }
                    else {
                        return Err(std::io::Error::other("Invalid JSON list syntax"));
                    }
                    let mut c = c.unwrap_or(b' ');
                    loop {
                        match c {
                            b',' => {},
                            b' '|b'\r'|b'\n'|b'\t' => {
                                s.read_exact(&mut buf)?;
                                c = buf[0];
                            },
                            b']' => return Ok((Some(JSON::List(res)),None)),
                            _ => return Err(std::io::Error::other("Invalid JSON list syntax"))
                        }
                    }
                },
                b'{' => {},
                b'0'..=b'9'|b'-' => {
                    let mut res = vec![c];
                    let mut buf = [0;3];
                    if 0 == s.read(&mut buf[0..1])? {
                        match c { 
                            b'0'..=b'9' => break Ok((JSON::Int((c-b'0') as i64),None)),
                            _ => break Err(std::io::Error::other("Invalid JSON integer syntax"))
                        }
                    }
                    if buf[0] == b'i' {
                        s.read_exact(&mut buf[1..3])?;
                        if &buf == b"inf" {
                            break Ok((JSON::Float(f64::NEG_INFINITY),None));
                        }
                        else {
                            break Err(std::io::Error::other("Invalid JSON integer syntax"));
                        }
                    }

                    loop {
                        if 0 == s.read(&mut buf[0..1])? {
                            let s = std::str::from_utf8(res.as_slice()).map_err(|e| std::io::Error::other("Invalid JSON numeric format"))?;
                            if let Ok(v) = s.parse::<i64>()  {
                                return Ok((JSON::Int(v),None));
                            }
                            else {
                                let v = s.parse::<f64>().map_err(|_| std::io::Error::other("Invalid JSON numeric format"))?;
                                return Ok((JSON::Float(v),None))
                            }
                        }
                        else {
                            match buf[0] {
                                b'0'..=b'9'|b'.'|b'e'|b'E'|b'+'|b'-' => res.push(buf[0]),
                                b => {
                                    let s = std::str::from_utf8(res.as_slice()).map_err(|e| std::io::Error::other("Invalid JSON numeric format"))?;
                                    if let Ok(v) = s.parse::<i64>()  {
                                        return Ok((JSON::Int(v),Some(b)));
                                    }
                                    else {
                                        let v = s.parse::<f64>().map_err(|_| std::io::Error::other("Invalid JSON numeric format"))?;
                                        return Ok((JSON::Float(v),Some(b)));
                                    }
                                }
                            }
                        }
                    }
                }, 
                b'i' => {
                    let mut buf = [0;2];
                    s.read_exact(&mut buf)?;
                    if &buf == b"nf" {
                        break Ok((JSON::Float(f64::INFINITY),None));
                    }
                    else {
                        break Err(std::io::Error::other("Invalid JSON syntax"));
                    }
                },
                b'n' => {
                    let mut buf = [0;3];
                    s.read_exact(&mut buf)?;
                    if &buf == b"ull" {
                        break Ok((JSON::Null,None));
                    }
                    else {
                        break Err(std::io::Error::other("Invalid JSON syntax"));
                    }
                },
                _ => return Err(std::io::Error::other("Invalid JSON syntax")),
            }
        }
    }


    pub fn read<T>(s : &mut T) -> std::io::Result<JSON> where T : Read {
        Self::read_hlp(s,None)
    }
}
