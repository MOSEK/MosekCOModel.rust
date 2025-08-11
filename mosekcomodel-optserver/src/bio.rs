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
use std::marker::PhantomData;

const BBOM : u32 = 0x42534142;
const REV_BBOM : u32 = 0x42415342;



/// Structure for serializing arrays. The b-stream consists of a series of entries, each entry a
/// name, a format and a series of fields. Each field is either a single value, a fixed-size array
/// of values or a stream og values.
pub struct Ser<'a,T> where T : Write {
    /// Unserlying writer
    w : &'a mut T,
    /// Buffer holding the current entry format
    curfmt : Vec<u8>,
    /// Indicates is an entry is currently active
    entry_active : bool,
}

/// Writer for a single b-stream entry.
pub struct SerEntry<'a,'b,T> where T : Write {
    ser : &'b mut Ser<'a,T>,
    ready : bool,
    fmtpos : usize,
}

/// Writer for a single b-stream entry field stream.
pub struct SerEntryChunkWriter<'a,'b,'c,T,E> where T : Write, E : Serializable {
    ent : &'c mut SerEntry<'a,'b,T>,
    ready : bool,
    _t : PhantomData<E>
}

fn validate_signature(sig : &[u8]) -> bool {
    sig.iter()
        .fold(Some(0),
              |pb,&b| 
                  pb.and_then(|pb|
                      match b {
                          b'b'| b'B'| b'h'| b'H'| 
                          b'i'| b'I'| b'l'| b'L'| 
                          b'f'| b'd' => Some(b),
                          b'[' => if pb == b'[' { None } else { Some(b) },
                          _ => None,
                      }))
        .and_then(|b| if b == b'[' { None } else { Some(b) })
        .is_some()
}

impl<'a,T> Ser<'a,T> where T : Write {
    /// Create a new b-stream from a writer object.
    pub fn new(w : &'a mut T) -> std::io::Result<Self> {
        let bom = [BBOM];
        let cbom : &[u8] = unsafe{ bom.align_to().1 };
        w.write_all(cbom)?;
        Ok(Ser{
            w,
            curfmt: Vec::new(),
            entry_active : false,
        })
    }

    /// Create new entry 
    ///
    /// # Arguments
    /// - `name` Name of the entry
    /// - `fmt` Format of the entry
    /// # Returns
    /// An entry writer.
    pub fn entry<'b>(&'b mut self,name : &[u8], fmt : &[u8]) -> std::io::Result<SerEntry<'a,'b,T>> {
        if self.entry_active {
            Err(std::io::Error::other("Unterminated entry"))
        }
        else if ! validate_signature(fmt) {
            Err(std::io::Error::other("Format error"))
        }
        else if name.len() > 255 {
            Err(std::io::Error::other("Name length limit of 255 exceeded"))
        }
        else if name.len() > 255 {
            Err(std::io::Error::other("Format length limit of 255 exceeded"))
        }
        else {
            self.w.write_all(&[name.len() as u8])?;
            self.w.write_all(name)?;
            self.w.write_all(&[fmt.len() as u8])?;
            self.w.write_all(fmt)?;

            self.curfmt.clear();
            self.curfmt.extend_from_slice(fmt);
            self.entry_active = true;

            Ok(SerEntry { 
                ser: self,
                ready : true,
                fmtpos: 0, 
            })
        }
    }

    /// Write the final b-stream block. Nothing can be subsequently added to the stream.
    pub fn finalize(&mut self) -> std::io::Result<()> {
        if self.entry_active {
            Err(std::io::Error::other("Entry still active"))
        }
        else if self.ready {
            self.ready = false;
            self.w.write_all(&[0,0])
        }
        else {
            Ok(())
        }
    }
}

impl<'a,'b,T> SerEntry<'a,'b,T> where T : Write {
    /// Write a single value of the given type. 
    ///
    /// # Arguments
    /// - `data` The value to write. The format must match the value type.
    pub fn write_value<E>(&mut self, data : E) -> std::io::Result<&mut Self> where E : Serializable {
        if ! self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.ser.curfmt.len() { return Err(std::io::Error::other("Write beyond entry end")); }
        let b0 = self.ser.curfmt[self.fmtpos];
        if b0 == b'[' { return Err(std::io::Error::other("Expected an array in entry")); } 

        self.fmtpos += 1;
        
        if E::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }
        let data = [data];
        let bdata = unsafe{ data.align_to().1 };
        self.ser.w.write_all(bdata)?;

        if self.fmtpos == self.ser.curfmt.len() {
            self.ser.entry_active = false;
        }
        Ok(self)
    }

    /// Write a fixed-length array to b-stream.
    ///
    /// # Arguments
    /// - `data` The value array to write. The format must match the value type.
    pub fn write_array<E>(&mut self, data : &[E]) -> std::io::Result<&mut Self> where E : Serializable {
        if ! self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.ser.curfmt.len() { return Err(std::io::Error::other("Write beyond entry end")); }
        let b0 = self.ser.curfmt[0];
        if b0 != b'[' { return Err(std::io::Error::other("Expected a single element in entry")); } 
        let b0 = self.ser.curfmt[self.fmtpos+1];
        self.fmtpos += 2;
        
        if E::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }
       
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
        size.iter_mut().rev().fold(n,|n,b| { *b = (n & 0xff) as u8; n >> 8 });
        size[6-nb as usize] |= (nb << 5) as u8;

        self.ser.w.write_all(&size[6-nb..])?;
        let bdata = unsafe { data.align_to().1 };
        self.ser.w.write_all(bdata)?;
        
        if self.fmtpos == self.ser.curfmt.len() {
            self.ser.entry_active = false;
        }
        Ok(self)
    }
    
    /// Create a value stream writer for the b-stream. 
    ///
    /// Note that the value type may not always be deducable, in which case it is necessary to
    /// specify he type explicitly.
    pub fn stream_writer<'c,E>(&'c mut self) -> std::io::Result<SerEntryChunkWriter<'a,'b,'c,T,E>> where E : Serializable
    {
        if ! self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.ser.curfmt.len() { return Err(std::io::Error::other("Write beyond entry end")); }
        let b0 = self.ser.curfmt[self.fmtpos];
        if b0 != b'[' { return Err(std::io::Error::other("Expected a single element in entry")); } 
        let b0 = self.ser.curfmt[self.fmtpos+1];
        self.fmtpos += 2;
       
        if E::sig() != b0 { return Err(std::io::Error::other("Incorrect type in entry")); }

        self.ready = false;
        self.ser.w.write_all(&[0xe0])?;

        Ok(SerEntryChunkWriter{ent:self,ready: true, _t : PhantomData::<E>::default()})
    }
}


impl<'a,'b,'c,T,E> SerEntryChunkWriter<'a,'b,'c,T,E> where T : Write, E : Serializable, T : Write {
    /// Terminate the stream. No subsequent writes are allowed.
    pub fn close(&mut self) -> std::io::Result<()> {
        self.ent.ser.w.write_all(&[0x80,0])?;
        self.ready = false;
        self.ent.ready = true;
        Ok(())            
    }

    /// Write values to stream.
    ///
    /// # Arguments
    /// - `buf` Values to write
    pub fn write(&mut self, buf: &[E]) -> io::Result<()> {
        if ! self.ready { return Err(std::io::Error::other("Chunk writer closed")); }
        let mut bdata : &[u8] = unsafe{ buf.align_to().1 } ;

        while ! bdata.is_empty() {
            let n = bdata.len().min(0x7ff8);
            self.ent.ser.w.write_all(&[(n >> 8) as u8, (n & 0xff) as u8])?;
            self.ent.ser.w.write_all(&bdata[..n])?;
            bdata = &bdata[n..];
        }
        Ok(())
    }
}

/// Trait defining anything that a type are can be written to a b-stream.
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




/// Structure for parsing a b-stream.
pub struct Des<'a,R> where R : Read {
    r : &'a mut R,
    
    byte_swap : bool,

    entry_active : bool,

    end_of_stream : bool
}

/// Structure for reading a single entry from a b-stream
pub struct DesEntry<'a,'b,R> where R : Read {
    fmt : [u8;256],
    name : [u8;256],
    fmtpos : usize,
    des : &'b mut Des<'a,R>,
    ready : bool,
}

#[derive(Copy,Clone)]
enum EntryKind {
    Empty,
    Value,
    Array(usize),
    Stream(usize)
}

/// Structure for eading a single element in a b-stream entry.
pub struct DesEntryReader<'a,'b,'c,R,E> 
    where 
        R : Read, 
        E : Serializable 
{
    entry : & 'c mut DesEntry<'a,'b,R>,
    _t    : PhantomData<E>,
    kind  : EntryKind,
    ready : bool,
}

impl<'a,R> Des<'a,R> where R : Read {
    /// Create a new b-stream deserializer.
    pub fn new(r : &'a mut R) -> std::io::Result<Self> { 
        let mut bom = [0u32];
        r.read_exact(unsafe{ bom.align_to_mut().1 })?;

        let byte_swap = match bom[0] {
            BBOM => false,
            REV_BBOM => unimplemented!("Byte swapping in data"),
            _ => return Err(std::io::Error::other("Invalid BOM"))
        };

        Ok(Des{ r, entry_active : false, byte_swap, end_of_stream : false })
    }

    /// Get the next entry.
    ///
    /// # Returns
    /// At the end of stream, return `None`, otherwise return an entry reader.
    pub fn next_entry<'b>(&'b mut self) -> std::io::Result<Option<DesEntry<'a,'b,R>>> {
        if self.entry_active { return Err(std::io::Error::other("Previous entry not finished")) }
        if self.end_of_stream { return Ok(None); }
        let mut name = [0u8; 256];
        let mut fmt  = [0u8; 256];
        self.r.read_exact(&mut name[..1])?;
        if name[0] > 0 {
            let len = name[0] as usize;
            self.r.read_exact(&mut name[1..1+len])?;
        }
        self.r.read_exact(&mut fmt[..1])?;
        if fmt[0] > 0 {
            let len = fmt[0] as usize;
            self.r.read_exact(&mut fmt[1..1+len])?;
            if ! validate_signature(&fmt[1..1+fmt[0] as usize]) {
                return Err(std::io::Error::other("Invalid signature"));
            }
        }
        
        if name[0] == 0 && fmt[0] == 0 {
            self.end_of_stream = true;
            Ok(None)
        }
        else {
            Ok(Some(DesEntry{des : self, fmt, name, fmtpos : 0, ready : true }))
        }
    }

    pub fn expect<'b>(&'b mut self, name : &[u8]) -> std::io::Result<DesEntry<'a,'b,R>> {
        match self.next_entry()? {
            None => Err(std::io::Error::other("Expected a entry, got end-of-file")),
            Some(v) => if v.name() == name { Ok(v) } else { Err(std::io::Error::other(format!("Expected a entry '{}'", std::str::from_utf8(name).unwrap_or("<invalid utf-8>")))) },
        }
    }
}

impl<'a,'b,R> DesEntry<'a,'b,R> where R : Read {
    /// Return the entry name
    pub fn name(&self) -> &[u8] { &self.name[1..1+self.name[0] as usize] }
    /// Return the fmt name
    pub fn fmt(&self)  -> &[u8] { &self.fmt[1..1+self.fmt[0] as usize] }
    pub fn check_fmt(self, fmt : &[u8]) -> std::io::Result<Self> {
        if self.fmt != fmt { Err(std::io::Error::other(format!("Expected entry in format '{}'",std::str::from_utf8(fmt).unwrap_or("<invalid utf-8>")))) }
        else { Ok(self) }
    }
   
    /// Get next field as a value of specific type. The type must match the signture.
    pub fn next_value<E>(&mut self) -> std::io::Result<Option<E>> where E : Serializable+Default+Copy {
        if !self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.fmt[0] as usize { return Ok(None); }
        if E::sig() != self.fmt[1+self.fmtpos] { return Err(std::io::Error::other("Incorrect field type requested")); }
        let mut data = [ E::default() ];
        self.des.r.read_exact(unsafe{ data.align_to_mut().1 })?;
        self.fmtpos += 1;
        Ok(Some(data[0]))
    }

    pub fn read<E>(&mut self) -> std::io::Result<Vec<E>>
        where 
            E : Serializable+Default+Copy 
    { 
        if let Some(mut r) = self.next::<E>()? {
            let mut res : Vec<E> = Vec::new();
            loop {
                let base = res.len(); res.resize(res.len()+4096,E::default());
                let n = r.read(&mut res[base..])?;
                res.truncate(base+n);
                if n == 0 { break; }
            }
            Ok(res)
        }
        else {
            Err(std::io::Error::other("Read beyond end of entry"))
        }
    }
    

    /// Get a reader for the next field. It may be necessary to specify the type of the field,
    /// which must match the signature.
    pub fn next<'c,E>(&'c mut self) -> std::io::Result<Option<DesEntryReader<'a,'b,'c,R,E>>> where E : Serializable+Default+Copy { 
        if !self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.fmt[0] as usize { return Ok(None); }
        
        let kind = 
            match self.fmt[1+self.fmtpos] {
                b'[' => {
                    let b = self.fmt[self.fmtpos+2];
                    if E::sig() != b { return Err(std::io::Error::other("Incorrect field type requested")); }

                    self.fmtpos += 2;
                    self.ready = false;
                    let mut buf = [0;7];
                    self.des.r.read_exact(&mut buf[..1])?;
                    let nb = (buf[0] >> 5) as usize;
                    if nb == 7 {
                        EntryKind::Stream(0)
                    }
                    else {
                        buf[0] &= 0x1f;
                        self.des.r.read_exact(&mut buf[1..nb+1]);
                        EntryKind::Array(buf[..nb+1].iter().fold(0,|v,&b| (v << 8) | (b as usize)))
                    }
                },
                b => {
                    if E::sig() != b {
                        return Err(std::io::Error::other("Incorrect field type requested"));
                    }
                    else {
                        self.fmtpos += 1;
                        EntryKind::Value
                    }
                }
            };
        Ok(Some(DesEntryReader{
            entry : self,
            kind,
            _t : Default::default(),
            ready : true }))
    }
}

impl<'a,'b,'c,R,E> DesEntryReader<'a,'b,'c,R,E> where R : Read, E : Serializable+Default+Copy {
    /// Read from the net field.
    fn read(&mut self, buf: &mut [E]) -> io::Result<usize> {
        match self.kind {
            EntryKind::Empty => Ok(0),
            EntryKind::Value => {
                if let Some(buf) = buf.get_mut(..1) {
                    self.entry.des.r.read_exact(unsafe{ buf.align_to_mut().1 })?;
                    self.kind = EntryKind::Empty;
                    Ok(1)
                }
                else {
                    Ok(0)
                }
            },
            EntryKind::Array(nleft) => {
                let n = nleft.min(buf.len());
                self.entry.des.r.read_exact(unsafe{ buf[..n].align_to_mut().1 })?;
                self.kind = if nleft == n { EntryKind::Empty } else { EntryKind::Array(nleft-n) };
                Ok(n)
            },
            EntryKind::Stream(nleft) => {
                let mut buf = buf;
                let mut chunk_left = nleft;
                let mut nread = 0;
                while ! buf.is_empty() {
                    if chunk_left == 0 {
                        let mut buf = [0;2];
                        self.entry.des.r.read_exact(&mut buf)?;
                        chunk_left = ((buf[0] as usize) << 8) | (buf[1] as usize);
                        if chunk_left == 0 {
                            self.kind = EntryKind::Empty;
                            break;
                        }

                    }
                    let n = (chunk_left / size_of::<E>()).min(buf.len());
                    assert!(n > 0);
                    self.entry.des.r.read_exact(unsafe { buf[..n].align_to_mut().1 })?;
                    nread += n;
                    chunk_left -= n * size_of::<E>();
                    buf = &mut buf[n..];
                }
                
                Ok(nread)
            }
        }
    }
}




















//
//pub struct BAIO<'a> {
//    data : &'a[u8],
//    pos : usize,
//
//    entry_active : bool,
//}
//
//pub struct Entry<'a,'b> {
//    io : &'b mut BAIO<'a>,
//    name : &'a[u8],
//    fmt : &'a[u8],
//    fmtpos : usize,
//}
//
//pub enum DataType {
//    U8,
//    S8,
//    U16,
//    S16,
//    U32,
//    S32,
//    U64,
//    S64,
//    F32,
//    F64
//}
//
//fn btotype(b : u8) -> Result<(usize,DataType),String> {
//    match b {
//        b'b' => Ok((1,DataType::U8)),
//        b'B' => Ok((1,DataType::S8)),
//        b'h' => Ok((2,DataType::U16)),
//        b'H' => Ok((2,DataType::S16)),
//        b'i' => Ok((4,DataType::U32)),
//        b'I' => Ok((4,DataType::S32)),
//        b'l' => Ok((8,DataType::U64)),
//        b'L' => Ok((8,DataType::S64)),
//        b'f' => Ok((4,DataType::F32)),
//        b'd' => Ok((8,DataType::F64)),
//        _ => Err("Invalid bdata format".to_string())
//    }
//}
//
//
//
//impl<'a,'b> Drop for Entry<'a,'b> { 
//    fn drop(&mut self) {
//        if self.io.entry_active {
//            panic!("Usage error: Unfinished entry in bdata");
//        }
//    }
//}
//
//impl<'a> BAIO<'a> {
//    pub fn new(data : &'a[u8]) -> BAIO<'a> { BAIO{ data, pos : 0, entry_active : false } }
//
//    fn peek_internal(&self) -> Result<Option<(&'a[u8],&'a[u8],usize)>,String> {
//        if self.entry_active { panic!("Invalid use of BAIO") }
//        if self.pos == self.data.len() { return Ok(None); }
//        let mut pos = self.pos;
//
//        let name_len = (*self.data.get(pos).ok_or_else(|| "Invalid bdata format".to_string())?) as usize;
//        let name = self.data.get(pos+1..pos+1+name_len).ok_or_else(|| "Invalid bdata format".to_string())?;
//        pos += name_len+1;
//
//        let fmt_len = (*self.data.get(pos).ok_or_else(|| "Invalid bdata format".to_string())?) as usize;
//        let fmt = self.data.get(pos+1..pos+1+name_len).ok_or_else(|| "Invalid bdata format".to_string())?;
//        pos += fmt_len+1;
//
//        Ok(Some((name,fmt,pos)))
//
//    }
//    pub fn next<'b>(&'b mut self) -> Result<Option<Entry<'a,'b>>,String> {
//        if let Some((name,fmt,pos)) = self.peek_internal()? {
//            self.pos = pos;
//
//            Ok(Some(Entry{
//                io : self,
//                fmt,
//                name,
//                fmtpos : 0
//            }))
//        }
//        else {
//            Ok(None)
//        }
//    }
//
//    pub fn expect<'b>(&'b mut self) -> Result<Entry<'a,'b>,String> {
//        self.next().and_then(|e| e.ok_or_else(|| "Expected entry bdata".to_string()))
//    }
//
//    pub fn peek(&self) -> Result<Option<(&'a[u8],&'a[u8])>,String> {
//        if let Some((name,fmt,_)) = self.peek_internal()? {
//            Ok(Some((name,fmt)))
//        }
//        else {
//            Ok(None)
//        }
//    }
//}
//
//impl<'a,'b> Entry<'a,'b> {
//    pub fn name(&self) -> &'a[u8] { self.name }
//    pub fn fmt(&self) -> &'a[u8] { self.fmt }
//    pub fn next(&mut self) -> Result<Option<(DataType,&'a[u8])>,String> {
//        if self.fmtpos == self.fmt.len() { return Ok(None); }
//        
//        let b = self.fmt.get(self.fmtpos).ok_or_else(|| "Invalid solution format".to_string())?; self.fmtpos += 1;
//        if *b == b'[' {
//            let (sz,tp) = btotype(*self.fmt.get(self.fmtpos).ok_or_else(|| "Invalid solution format".to_string())?)?; self.fmtpos += 1;
//            let b0 = self.io.data.get(self.io.pos).ok_or_else(|| "Invalid solution format".to_string())?; self.io.pos += 1;
//            match b0 >> 5 {
//                7 => Err("Invalid solution format".to_string()), 
//                nb => {
//                    let mut numbytes = (b0 & 0x1f) as usize;
//                    for &b in self.io.data.get(self.io.pos+1..self.io.pos+1+nb as usize).ok_or_else(|| "Invalid solution format".to_string())?.iter() {
//                        numbytes = numbytes << 8 | b as usize;
//                    }
//                    numbytes *= sz;
//                    self.io.pos += nb as usize+1;
//                    let res = &self.io.data[self.io.pos..self.io.pos+numbytes];
//                    self.io.pos += numbytes;
//                    if self.io.pos == self.io.data.len() { self.io.entry_active = false; }
//                    Ok(Some((tp,res)))
//                }
//            }
//        }
//        else {
//            let (sz,tp) = btotype(*b)?;
//            let res = &self.io.data[self.io.pos..self.io.pos+sz];
//            self.io.pos += sz;
//            if self.io.pos == self.io.data.len() { self.io.entry_active = false; }
//            Ok(Some((tp,res)))
//        }
//    }
//    pub fn finalize(&mut self) -> Result<(),String> {
//        while self.next()?.is_some() {
//        }
//        Ok(())
//    }
//}
