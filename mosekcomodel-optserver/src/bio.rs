//! Module implementing functionality for formatting and parsing `bdata` files. `bdata` is
//! a general MOSEK format serializing basic array types in a tagged format. 
//!
//! # The B-format:
//! ```text
//! FILE -> BOM ENTRY* LASTENTRY
//!
//! BOM -> b"BASF"|b"FSAB" # depending on endian. If "FSAB", then all entries of size>1 must be byte-swapped.
//! ENTRY -> NAME FORMAT DATAENTRY # the item size and content of DATAENTRY is implied by FORMAT
//! NAME -> size:u8, u8[size]
//! FORMAT-> size:u8, fmt:u8[size]
//! ```
//! The `fmt` is a string that describes the format of DATAENTRY
//! ```text
//! fmt -> FMTENTRY*
//! FMTENTRY -> FMTPRIM
//!          -> FMTARR
//! FMTPRIM  -> 'b' # i8
//!             'B' # u8
//!             'h' # i16
//!             'H' # u16
//!             'i' # i32
//!             'I' # u32
//!             'l' # i64
//!             'L' # u64
//!             'f' # f32
//!             'd' # f64
//! FMTARR -> '[' FMTPRIM # array or stream of primitives
//! ```
//!
//! The `DATAENTRY` for a primitive (`FMTPRIM`) is a single value with type
//! indicated by the format char. For an array `DATAENTRY` has the format
//! ```text
//! DATAENTRY -> {b0:u8 | (b0 >> 5) != 7 }, bx : [u8; b0 >> 5], data : [u8; [b0 & 0x1f, bx] as NBO integer ]
//!           |  {b0:u8 | (b0 >> 5) == 7 }, data : STREAM
//! STREAM -> CHUNK* ENDCHUNK
//! ENDCHUNK -> hdr:u16 = 0
//! CHUNK    -> N:NBO u16, [u8;N]
//! LASTENTRY: 0u16 
//! ```

use std::io::{self, Read, Write};

use itertools::Either;

const BBOM : u32 = 0x42534142;
const REV_BBOM : u32 = 0x42415342;


pub struct Ser<'a,T> where T : Write {
    w : &'a mut T,
    curfmt : Vec<u8>,
    entry_active : bool
}

pub struct SerEntry<'a,'b,T> where T : Write {
    ser : &'b mut Ser<'a,T>,
    fmt : &'b [u8]
}

fn validate_signature(sig : &[u8]) -> bool {
    sig.iter()
        .fold(Some(0),
              |pb,&b| 
                  pb.and_then(|pb|
                      match b {
                          b'b'| b'B'| b'h'| b'H'| 
                          b'i'| b'I'| b'l'| b'L'| 
                          b'f'| b'd' => Ok(b),
                          b'[' => if pb == b'[' { None } else { Some(b) },
                      }))
        .and_then(|b| if *b == b'[' { None } else { Some(b) })
        .is_some()
}

impl<'a,T> Ser<'a,T> where T : Write {
    pub fn new(w : &'a mut T) -> std::io::Result<Self> {
        let bom = [BBOM];
        let cbom : &[u8] = unsafe{ bom.align_to().1 };
        w.write_all(cbom)?;
        Ok(Ser{
            w,
            curfmt: Vec::new(),
            entry_active : false
        })
    }

    pub fn entry<'b>(&'b mut self,name : &[u8], fmt : &[u8]) -> std::io::Result<SerEntry<'b,T>> {
        if self.entry_active {
            Err(std::io::Error::other("Unterminated entry"))
        }
        else if ! validate_signature(fmt) {
            Err(std::io::Error::other("Format error"))
        }
        else {
            self.curfmt.clear();
            self.curfmt.extend_from_slice(fmt);
            self.entry_active = true;

            Ok(SerEntry { ser: self, fmt: self.curfmt.as_slice() })
        }
    }
}


impl<'a,'b,T> SerEntry<'a,'b,T> where T : Write {
    pub fn write_value(&mut self, data : T) -> std::io::Result<()> where T : Serializable {
        let b0 = self.fmt.get(0).ok_or_else(|| std::io::Error::other("Write beyond entry end"))?;
        if b0 == b'[' { return Err(std::io::Error::other("Expected an array in entry")); } 
        self.fmt = &self.fmt[1..];
        
        if T::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }
        let data = [data];
        let bdata = unsafe{ data.align_to().1 };
        self.ser.w.write_all(bdata)?;

        if self.fmt.len() == 0 {
            self.ser.entry_active = false;
        }
        Ok(())
    }

    pub fn write_array(&mut self, data : &[T]) -> std::io::Result<()> where T : Serializable {
        let b0 = self.fmt.get(0).ok_or_else(|| std::io::Error::other("Write beyond entry end"))?;
        if b0 != b'[' { return Err(std::io::Error::other("Expected a single element in entry")); } 
        let b0 = self.fmt[1];
        self.fmt = &self.fmt[2..];
        
        if T::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }
       
        let n = data.len();
        let nb = 
            if      n <= 0x1f { 0 }
            else if n <= 0x1fff { 1 } 
            else if n <= 0x1fffff { 2} 
            else if n <= 0x1fffffff { 3 } 
            else if n <= 0x1fffffffff { 4 } 
            else if n <= 0x1fffffffffff { 5 } 
            else if n <= 0x1fffffffffffff { 6 }
            else  { return Err(std::io::Error::other("Array too large")) };

        let mut size = [0;7];
        size.iter_mut().rev().fold(n,|n,b| { *b = n & 0xff; n >> 8 });
        size[6-nb as usize] |= nb << 5;

        self.ser.w.write_all(&size[6-nb..])?;
        let bdata = unsafe { data.align_to().1 };
        self.ser.w.write_all(bdata)?;
        
        if self.fmt.len() == 0 {
            self.ser.entry_active = false;
        }
        Ok(())
    }
    
    pub fn write_stream<F>(&mut self, cb : F) -> std::io::Result<()> where T : Serializable, F : FnMut(&[T]) -> std::io::Result<()> {
        let b0 = self.fmt.get(0).ok_or_else(|| std::io::Error::other("Write beyond entry end"))?;
        if b0 != b'[' { return Err(std::io::Error::other("Expected a single element in entry")); } 
        let b0 = self.fmt[1];
        self.fmt = &self.fmt[2..];
       
        if T::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }

        let mut hd = [0xe0];
        self.ser.w.write_all(&hd)?;

        cb(&mut SerEntryChunkWriter{sig:b0, ent:self})?;
        let hd = [0;2];
        self.ser.w.write_all(&hd)?;


        if self.fmt.len() == 0 {
            self.ser.entry_active = false;
        }
        Ok(())
    }
}

struct SerEntryChunkWriter<'a,'b,'c,T> where T : Write {
    sig : u8,
    ent : &'c mut SerEntry<'a,'b,T>,
}

impl<'a,'b,'c,T> Write for SerEntryChunkWriter<'a,'b,'c,T> {
    fn write<B>(&mut self, buf: &[B]) -> io::Result<()> where B : Serializable {
        if B::sig() != self.sig { return Err(std::io::Error::other("Invalid data type")) }
        
        let mut bdata : &[u8] = unsafe{ buf.align_to() } ;

        while ! bdata.is_empty() {
            let n = bdata.len().min(0xfff0);
            let mut hd = [(n >> 8) as u8, (n & 0xff) as u8];
            self.ent.ser.write_all(&bdata[..n])?;
            bdata = &bdata[n..];
        }
        Ok(())
    }
    fn flush(&mut self) -> io::Result<()> { Ok(()) }
}

pub trait Serializable { fn sig() -> u8; }

impl Serializable for u8   { fn sig() -> u8 { b'B' } }
impl Serializable for i8   { fn sig() -> u8 { b'b' } }
impl Serializable for u16  { fn sig() -> u8 { b'H' } }
impl Serializable for i16  { fn sig() -> u8 { b'h' } }
impl Serializable for u32  { fn sig() -> u8 { b'U' } }
impl Serializable for i32  { fn sig() -> u8 { b'i' } }
impl Serializable for u64  { fn sig() -> u8 { b'L' } }
impl Serializable for i64  { fn sig() -> u8 { b'l' } }
impl Serializable for f32  { fn sig() -> u8 { b'f' } }
impl Serializable for f64  { fn sig() -> u8 { b'd' } }

//------------------------------------------------------------------
struct Des<'a,R> where R : Read {
    r : &'a mut R,
    curfmt : Vec<u8>,
    curname : Vec<u8>,
    
    byte_swap : bool,

    entry_active : bool,

    end_of_stream : bool
}

impl<'a,R> Des<'a,R> where R : Read {
    pub fn new(r : &'a mut R) -> std::io::Result<Self> { 
        let mut bom = &[0u32];
        let mut bbom = unsafe{ bom.align_to_mut() };
        r.read_all(bbom)?;

        let byte_swap = match bom[0] {
            BBOM => false,
            REV_BBOM => unimplemented!("Byte swapping in data"),
            _ => return Err(std::io::Error::other("Invalid BOM"))
        };

        Ok(Des{ r, curfmt : Vec::new(), curname : Vec::new(), entry_active : false, byte_swap, end_of_stream : false })
    }

    pub fn next_entry<'b>(&'b mut self) -> std::io::Result<Option<DesEntry<'a,'b,R>>> {
        if self.entry_active { return Err(std::io::Error::other("Previous entry not finished")) }
        if self.end_of_stream { return Ok(None); }
        let mut namelen = [0;1];
        self.r.read_exact(&namelen)?;
        if namelen[0] > 0 {
            self.curname.resize(namelen[0] as usize,0);
            self.r.read_exact(self.curname.as_mut_slice())?
        }
        let mut fmtlen = [0;1];
        self.r.read_exact(&fmtlen)?;
        if fmtlen[0] > 0 {
            self.curname.resize(fmtlen[0] as usize,0);
            self.r.read_exact(self.curfmt.as_mut_slice())?;
            if ! validate_signature(self.curfmt.as_slice()) {
                return Err(std::io::Error::other("Invalid signature"));
            }
        }
        
        if namelen[0] == 0 && fmtlen[0] == 0 {
            self.end_of_stream = true;
            Ok(None)
        }
        else {
            Ok(Some(DesEntry{des : self, fmt : &self.fmt.as_slice()}))
        }
    }
}

struct DesEntry<'a,'b,R> where R : Read {
    des : &'b mut Des<'a,R>,
    fmt : &'b[u8]
}

impl<'a,'b,R> DesEntry<'a,'b,R> where R : Read {
    pub fn name(&self) -> &'b[u8] { self.des.curname.as_slice() }
    pub fn fmt(&self) -> &'b[u8] { self.des.curfmt.as_slice() }
   
    pub fn next_value<T>(&mut self) -> std::io::Result<Option<T>> where T : Serializable {
        if self.fmt.len() == 0 { return Ok(None); }
        if T::sig() != self.fmt.get(0).ok_or_else(|| std::io::Error::other("Incorrect field type requested"))?;
        let mut data = [0];
        let mut bdata = unsafe{ data.align_to_mut().1 };
        self.des.r.read_exact(bdata)?;
        self.fmt = &self.fmt[1..];
        Ok(Some(data[0]))
    }
    pub fn next<'c,T>(&'c mut self) -> std::io::Result<Option<DesEntryReader<'a,'b,'c,R,T>>> where T : Serializable { 
        if self.fmt.len() == 0 { return Ok(None); }
        
        let size = 
            match self.fmt.get(0).ok_or_else(|| std::io::Error::other("Incorrect field type requested"))? {
                b'[' => {
                    let b = self.fmt[1];
                    if T::sig() != b {
                        return Err(std::io::Error::other("Incorrect field type requested"));
                    }

                    self.fmt = &self.fmt[2..];
                    let mut buf = [0;7];
                    self.des.r.read_exact(&buf[..1])?;
                    let nb = (buf[0] >> 5) as usize;
                    if nb == 7 {
                        EntryKind::Stream(0)
                    }
                    else {
                        buf[0] = buf[0] & 0x1f;
                        self.des.r.read_exact(&buf[1..nb+1]);
                        EntryKind::Array(buf[..nb+1].iter().fold(0,|v,&b| (v << 8) | (b as usize)))
                    }
                },
                b => {
                    if T::sig() != b {
                        return Err(std::io::Error::other("Incorrect field type requested"));
                    }
                    else {
                        self.fmt = &self.fmt[1..];
                        EntryKind::Value
                    }
                }


            };
        Ok(Some(DesEntryReader{entry : self, size }))
    }
}

#[derive(Copy)]
enum EntryKind {
    Empty,
    Value,
    Array(usize),
    Stream(usize)
}

struct DesEntryReader<'a,'b,'c,R,T> where R : Read {
    entry : &'c DesEntry<'a,'b,R>,
    size : EntryKind
}

impl<'a,'b,'c,R,T> DesEntryReader<'a,'b,'c,R,T> where R : Read {
    fn read(&mut self, buf: &mut [T]) -> io::Result<usize> {
        match self.size {
            EntryKind::Empty => Ok(0),
            EntryKind::Value => {
                if let Some(buf) = buf.get(..1) {
                    self.entry.des.r.read_exact(unsafe{ buf.align_to_mut().1 })?;
                    self.size = EntryKind::Empty;
                    Ok(1)
                }
                else {
                    Ok(0)
                }
            },
            EntryKind::Array(nleft) => {
                let n = nleft.min(buf.len());
                self.entry.des.r.read_exact(unsafe{ buf[..n].align_to_mut().1 })?;
                self.size = if nleft == n { EntryKind::Empty } else { EntryKind::Array(nleft-n) };
                Ok(n)
            },
            EntryKind::Stream(nleft) => {
                let mut buf = buf;
                let chunk_left = nleft;
                let nread = 0;
                while ! buf.is_empty() {
                    if chunk_left == 0 {
                        let mut buf = [0;2];
                        self.entry.des.r.read_exact(&buf)?;
                        chunk_left = ((buf[0] as usize) << 8) | (buf[1] as usize);
                        if chunk_left == 0 {
                            self.size = EntryKind::Empty;
                            break;
                        }

                    }
                    let n = (chunk_left / size_of::<T>).min(buf.len());
                    assert!(n > 0);
                    self.entry.des.r.read_exact(unsafe { buf[..n].align_to_mut() })?;
                    nread += n;
                    chunk_left -= n * size_of::<T>();
                    buf = &mut buf[n..];
                }
                
                Ok(nread)
            }
        }
    }
}









pub struct BAIO<'a> {
    data : &'a[u8],
    pos : usize,

    entry_active : bool,
}

pub struct Entry<'a,'b> {
    io : &'b mut BAIO<'a>,
    name : &'a[u8],
    fmt : &'a[u8],
    fmtpos : usize,
}

pub enum DataType {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
    S64,
    F32,
    F64
}

fn btotype(b : u8) -> Result<(usize,DataType),String> {
    match b {
        b'b' => Ok((1,DataType::U8)),
        b'B' => Ok((1,DataType::S8)),
        b'h' => Ok((2,DataType::U16)),
        b'H' => Ok((2,DataType::S16)),
        b'i' => Ok((4,DataType::U32)),
        b'I' => Ok((4,DataType::S32)),
        b'l' => Ok((8,DataType::U64)),
        b'L' => Ok((8,DataType::S64)),
        b'f' => Ok((4,DataType::F32)),
        b'd' => Ok((8,DataType::F64)),
        _ => Err("Invalid bdata format".to_string())
    }
}



impl<'a,'b> Drop for Entry<'a,'b> { 
    fn drop(&mut self) {
        if self.io.entry_active {
            panic!("Usage error: Unfinished entry in bdata");
        }
    }
}

impl<'a> BAIO<'a> {
    pub fn new(data : &'a[u8]) -> BAIO<'a> { BAIO{ data, pos : 0, entry_active : false } }

    fn peek_internal(&self) -> Result<Option<(&'a[u8],&'a[u8],usize)>,String> {
        if self.entry_active { panic!("Invalid use of BAIO") }
        if self.pos == self.data.len() { return Ok(None); }
        let mut pos = self.pos;

        let name_len = (*self.data.get(pos).ok_or_else(|| "Invalid bdata format".to_string())?) as usize;
        let name = self.data.get(pos+1..pos+1+name_len).ok_or_else(|| "Invalid bdata format".to_string())?;
        pos += name_len+1;

        let fmt_len = (*self.data.get(pos).ok_or_else(|| "Invalid bdata format".to_string())?) as usize;
        let fmt = self.data.get(pos+1..pos+1+name_len).ok_or_else(|| "Invalid bdata format".to_string())?;
        pos += fmt_len+1;

        Ok(Some((name,fmt,pos)))

    }
    pub fn next<'b>(&'b mut self) -> Result<Option<Entry<'a,'b>>,String> {
        if let Some((name,fmt,pos)) = self.peek_internal()? {
            self.pos = pos;

            Ok(Some(Entry{
                io : self,
                fmt,
                name,
                fmtpos : 0
            }))
        }
        else {
            Ok(None)
        }
    }

    pub fn expect<'b>(&'b mut self) -> Result<Entry<'a,'b>,String> {
        self.next().and_then(|e| e.ok_or_else(|| "Expected entry bdata".to_string()))
    }

    pub fn peek(&self) -> Result<Option<(&'a[u8],&'a[u8])>,String> {
        if let Some((name,fmt,_)) = self.peek_internal()? {
            Ok(Some((name,fmt)))
        }
        else {
            Ok(None)
        }
    }
}

impl<'a,'b> Entry<'a,'b> {
    pub fn name(&self) -> &'a[u8] { self.name }
    pub fn fmt(&self) -> &'a[u8] { self.fmt }
    pub fn next(&mut self) -> Result<Option<(DataType,&'a[u8])>,String> {
        if self.fmtpos == self.fmt.len() { return Ok(None); }
        
        let b = self.fmt.get(self.fmtpos).ok_or_else(|| "Invalid solution format".to_string())?; self.fmtpos += 1;
        if *b == b'[' {
            let (sz,tp) = btotype(*self.fmt.get(self.fmtpos).ok_or_else(|| "Invalid solution format".to_string())?)?; self.fmtpos += 1;
            let b0 = self.io.data.get(self.io.pos).ok_or_else(|| "Invalid solution format".to_string())?; self.io.pos += 1;
            match b0 >> 5 {
                7 => Err("Invalid solution format".to_string()), 
                nb => {
                    let mut numbytes = (b0 & 0x1f) as usize;
                    for &b in self.io.data.get(self.io.pos+1..self.io.pos+1+nb as usize).ok_or_else(|| "Invalid solution format".to_string())?.iter() {
                        numbytes = numbytes << 8 | b as usize;
                    }
                    numbytes *= sz;
                    self.io.pos += nb as usize+1;
                    let res = &self.io.data[self.io.pos..self.io.pos+numbytes];
                    self.io.pos += numbytes;
                    if self.io.pos == self.io.data.len() { self.io.entry_active = false; }
                    Ok(Some((tp,res)))
                }
            }
        }
        else {
            let (sz,tp) = btotype(*b)?;
            let res = &self.io.data[self.io.pos..self.io.pos+sz];
            self.io.pos += sz;
            if self.io.pos == self.io.data.len() { self.io.entry_active = false; }
            Ok(Some((tp,res)))
        }
    }
    pub fn finalize(&mut self) -> Result<(),String> {
        while self.next()?.is_some() {
        }
        Ok(())
    }
}
