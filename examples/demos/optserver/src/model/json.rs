//! 
//! Basic and simple JSON formatting and parsing. 
//!
//! Please note that this is for demonstration purposes only and if likely quite slow and
//! inifficient.
use std::fmt::{Display, Write};

pub enum Item {
    String(String),
    Int(i64),
    Float(f64),
    Null,
    NaN,
    List(Vec<Item>),
    Dict(Vec<(String,Item)>)
}

impl Item {
    pub fn get_string(&self) -> Option<&String> { if let Item::String(s) = self { Some(s) } else { None } }
    pub fn get_int(&self) -> Option<i64> { if let Item::Int(i) = self { Some(*i) } else { None }}
    pub fn get_float(&self) -> Option<f64> { if let Item::Float(f) = self{ Some(*f) } else { None }}
    pub fn get_list(&self) -> Option<&[Item]> { if let Item::List(l) = self { Some(l.as_slice()) } else { None }}
    pub fn get_dict(&self) -> Option<&[(String,Item)]> { if let Item::Dict(d) = self { Some(d.as_slice()) } else { None }}

    pub fn to_float(&self) -> Option<f64> {
        self.get_float().or_else(|| self.get_int().map(|v|v as f64))
    }
    pub fn get_float_list(&self) -> Option<Vec<f64>> { 
        if let Item::List(l) = self {
            if l.iter().all(|item| match item { Item::Int(_)|Item::Float(_) => true, _ => false } ) {
                Some(l.iter().map(|v| match v { Item::Int(i) => *i as f64, Item::Float(f) => *f, _ => 0.0}).collect())
            }
            else {
                None
            }
        }
        else {
            None
        }
    }
    pub fn get_int_list(&self) -> Option<Vec<i64>> { 
        if let Item::List(l) = self {
            if l.iter().all(|item| match item { Item::Int(_) => true, _ => false } ) {
                Some(l.iter().map(|v| match v { Item::Int(i) => *i, _ => 0}).collect())
            }
            else {
                None
            }
        }
        else {
            None
        }
    }
}

impl Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Item::String(v) => { 
                f.write_char('"')?; 
                for c in v.chars() {
                    match c {
                        '\n' => { f.write_char('\\')?; f.write_char('n')?; },
                        '\t' => { f.write_char('\\')?; f.write_char('t')?; },
                        '\r' => { f.write_char('\\')?; f.write_char('r')?; },
                        '\0' => { f.write_char('\\')?; f.write_char('0')?; },
                        _ => f.write_char(c)?,
                        
                    }
                }
                f.write_char('"')?; 
                Ok(())
            },
            Item::Int(v) => v.fmt(f),
            Item::Float(v) => {
                if *v < f64::INFINITY {
                    if *v > f64::NEG_INFINITY {
                        v.fmt(f)
                    }
                    else {
                        f.write_str("-1e300")
                    }
                } 
                else {
                    f.write_str("1e300")
                }
            }
            Item::Null => f.write_str("null"),
            Item::NaN => f.write_str("nan"),
            Item::List(l) => { 
                f.write_char('[')?;
                if let Some(v) = l.first() {
                    v.fmt(f)?;
                    for v in l[1..].iter() {
                        f.write_char(',')?;
                        v.fmt(f)?;
                    }
                }
                f.write_char(']')?; 
                Ok(())
            },
            Item::Dict(d) => {
                f.write_char('{')?;
                if let Some((k,v))= d.first() {
                    Item::String(k.clone()).fmt(f)?; f.write_char(':')?; v.fmt(f)?;
                    for (k,v) in d[1..].iter() {
                        f.write_char(',')?;
                        Item::String(k.clone()).fmt(f)?; f.write_char(':')?; v.fmt(f)?;
                    }
                }
                f.write_char('}')?; 
                Ok(())
            }
        }
    }
}



pub fn skip_ws<I>(it : &mut std::iter::Peekable<I>) 
    where I : Iterator<Item = (usize,u8)>
{
    while let Some((_,b' ')) = it.peek() { it.next(); } 
}

pub fn expect<I>(it : &mut std::iter::Peekable<I>, c : u8) -> Result<(),String> 
    where I : Iterator<Item = (usize,u8)>
{
    it.next().ok_or_else(|| format!("At EOF: Expected '{}'",c))
        .and_then(|v| if v.1 == c { Ok(()) } else { Err(format!("At {}: Expected '{}', got '{}'",v.0,c as char,v.1 as char)) } )
}

pub fn parse_string_into<I>(it : &mut std::iter::Peekable<I>,res : & mut String) -> Result<(),String>
    where I : Iterator<Item = (usize,u8)>
{
    skip_ws(it);
    expect(it,b'"')?;

    while let Some((_,c)) = it.next() {
        match c {
            b'"' => return Ok(()),
            b'\\' => 
                if let Some((ofs,c)) = it.next() {
                    match c {
                        b'n' => res.push('\n'),
                        b't' => res.push('\t'),
                        b'r' => res.push('\r'),
                        b'0' => res.push('\0'),
                        b'x' => {
                            if let (Some((ofs0,hi)),Some((ofs1,lo))) = (it.next(),it.next()) {
                                let hi = match hi {
                                    b'0'..=b'9' => hi - b'0',
                                    b'a'..=b'f' => hi - b'a' + 10,
                                    b'A'..=b'F' => hi - b'A' + 10,
                                    _ => return Err(format!("At {}: Expected hex char, got '{}'",ofs0,hi as char))
                                };
                                let lo = match lo {
                                    b'0'..=b'9' => hi - b'0',
                                    b'a'..=b'f' => hi - b'a' + 10,
                                    b'A'..=b'F' => hi - b'A' + 10,
                                    _ => return Err(format!("At {}: Expected hex char, got '{}'",ofs1,lo as char))
                                };
                                res.push(((hi << 4) + lo) as char);
                            }
                            else {
                                return Err(format!("At EOF: Invalid string format"))
                            }
                        },
                        _ => return Err(format!("At {}: Invalid char '{}'",ofs,c as char))
                    }
                }
            _ => res.push(c as char),
        }
    }
    Err("Premature end of file".to_string())
}
pub fn parse_string<I>(it : &mut std::iter::Peekable<I>) -> Result<String,String>
    where I : Iterator<Item = (usize,u8)>
{
    let mut res = String::new();
    parse_string_into(it,& mut res)?;
    Ok(res)

}

pub fn parse_key_value<I>(it : & mut std::iter::Peekable<I>) -> Result<(String,Item),String>
    where I : Iterator<Item = (usize,u8)>
{
    skip_ws(it);
    let key = parse_string(it)?;
    skip_ws(it);
    if let Some(&(ofs,c)) = it.peek() {
        if c == b':' {
            it.next();
            skip_ws(it);
            let value = parse_item(it)?;
            Ok((key,value))
        }
        else {
            Err(format!("At {}: Expected a ':', got '{}'",ofs,c as char))
        }
    }
    else {
        Err("At EOF: Expected a ':'".to_string())
    }
}


pub fn parse_number_with<I>(it : & mut std::iter::Peekable<I>, res : &mut String) -> Result<Item,String>
    where I : Iterator<Item = (usize,u8)>
{
    if let Some((ofs,c)) = it.next() {
        // [0-9]+ ( '.' [0-9]+ )? ( [eE] [+-]? [0-9]+)?
        match c {
            b'0'..=b'9' => res.push(c as char),
            b'i' => 
                if let (Some((_,b'n')),Some((_,b'f'))) = (it.next(),it.next()) {
                    if res.chars().nth(0).map(|c| c == '-').unwrap_or(false) {
                        return Ok(Item::Float(f64::NEG_INFINITY));
                    }
                    else {
                        return Ok(Item::Float(f64::INFINITY));
                    }
                },
            _ => return Err(format!("At {}: Expected a number, got '{}'",ofs,c as char))
        }

        while let Some(&(_,c)) = it.peek() {
            match c {
                b'0'..=b'9' => { it.next(); res.push(c as char) },
                _ => break
            }
        }

        let mut is_float = false;
        if let Some((_,b'.')) = it.peek() {
            is_float = true;
            res.push('.'); 
            it.next();
            while let Some(&(_,c)) = it.peek() {
                match c {
                    b'0'..=b'9' => { it.next();  res.push(c as char) },
                    _ => break
                }
            }
        }

        if let Some((_,c)) = it.peek() {
            if *c == b'e' || *c == b'E' {
                is_float = true;
                res.push(*c as char);
                it.next();

                if let Some((_,c)) = it.peek() {
                    if *c == b'+' || *c == b'-' {
                        res.push(*c as char);
                        it.next();
                    }
                    if let Some(&(_,c)) = it.peek() {
                        match c {
                            b'0'..=b'9' => res.push(c as char),
                            _ => return Err("Expected a number".to_string())
                        }
                        it.next(); 
                        while let Some(&(_,c)) = it.peek() {
                            match c {
                                b'0'..=b'9' => res.push(c as char),
                                _ => break
                            }
                            it.next();
                        }
                    }
                }
            }
        }

        if is_float {
            Ok(Item::Float(res.parse().map_err(|_| format!("Invalid float format: '{}'",res))?))
        }
        else {
            Ok(Item::Int(res.parse().map_err(|_| format!("Invalid int format: '{}'",res))?))
        }
     }
    else {
        Err("Expected a number".to_string())
    }
}

pub fn parse_item<I>(it : & mut std::iter::Peekable<I>) -> Result<Item,String>
    where I : Iterator<Item = (usize,u8)>
{
    skip_ws(it);
    
    if let Some(&(ofs,c)) = it.peek() {
        match c {
            b'"' => {
                let mut res = String::new();
                parse_string_into(it, &mut res)?;
                Ok(Item::String(res))
            }
            b'-'|b'+' => {
                it.next();
                let mut res = String::new();
                skip_ws(it);
                res.push(c as char);
                parse_number_with(it,&mut res)
            },
            b'0'..=b'9'|b'i' => {
                let mut res = String::new();
                parse_number_with(it,&mut res)
            },
            b'n' => {
                it.next();
                if let (Some((_,b'u')),Some((_,b'l')),Some((_,b'l'))) = (it.next(),it.next(),it.next()) {
                    Ok(Item::Null)
                }
                else {
                    Err(format!("At {}: Syntax error",ofs))
                }
            },
            b'[' => {
                it.next();
                let mut res = Vec::new();
                skip_ws(it);
                if let Some((_,b']')) = it.peek() {
                    it.next();
                    Ok(Item::List(res))
                }
                else {
                    res.push(parse_item(it)?);
                    skip_ws(it);
                    while let Some((_,b',')) = it.peek() {
                        it.next();
                        skip_ws(it);
                        res.push(parse_item(it)?);
                        skip_ws(it);
                    }
                    if let Some(&(ofs,c)) = it.peek() {
                        if c == b']' {
                            it.next();
                            Ok(Item::List(res))
                        }
                        else {
                            Err(format!("At {}: Expected ']', got '{}'",ofs,c as char))
                        }
                    }
                    else {
                        Err("At EOF: Expected ']'".to_string())
                    }
                }
            },
            b'{' => {
                it.next();
                let mut res = Vec::new();
                skip_ws(it);
                if let Some((_,b'}')) = it.peek() {
                    it.next();
                    Ok(Item::Dict(res))
                } 
                else {
                    res.push(parse_key_value(it)?);
                    skip_ws(it);
                    while let Some((_,b',')) = it.peek() {
                        it.next();
                        skip_ws(it);
                        res.push(parse_key_value(it)?);
                        skip_ws(it);
                    }
                    if let Some(&(ofs,c)) = it.peek() {
                        if c == b'}' {
                            it.next();
                            Ok(Item::Dict(res))
                        }
                        else {
                            Err(format!("At {}: Expected '}}', got '{}'",ofs,c as char))
                        }
                    }
                    else {
                        Err("At EOF: Expected '}}'".to_string())
                    }
                }
                
            },
            _ => Err(format!("At {}: Syntax error in file",ofs))
        }
    }
    else {
        Err("At EOF: Syntax error".to_string())
    }
}

pub fn parse(data : &[u8]) -> Result<Item,String> {
    parse_item(&mut data.iter().cloned().enumerate().peekable())
}

pub trait ToJSON {
    fn to_json(&self) -> Item;
}
impl ToJSON for usize   { fn to_json(&self) -> Item { Item::Int((*self).try_into().unwrap()) } }
impl ToJSON for f64     { fn to_json(&self) -> Item { Item::Float(*self) } }
impl ToJSON for str     { fn to_json(&self) -> Item { Item::String(self.to_string()) } }
impl ToJSON for String  { fn to_json(&self) -> Item { Item::String(self.clone()) } }
impl<T> ToJSON for [T] where T : ToJSON  { fn to_json(&self) -> Item { Item::List(self.iter().map(|v| v.to_json()).collect()) } }
impl<T> ToJSON for [(String,T)] where T : ToJSON { fn to_json(&self) -> Item { Item::Dict(self.iter().map(|(k,v)| (k.clone(),v.to_json())).collect()) } }
impl<T> ToJSON for [(&str,T)] where T : ToJSON { fn to_json(&self) -> Item { Item::Dict(self.iter().map(|(k,v)| (k.to_string(),v.to_json())).collect()) } }

