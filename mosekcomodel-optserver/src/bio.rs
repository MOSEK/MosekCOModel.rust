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

use std::default;
use std::fmt::Pointer;
use std::io::{self, Read, Write};
use std::marker::PhantomData;

const BBOM     : u32 = 0x424b534d;
const REV_BBOM : u32 = 0x4d534b42;


#[derive(Debug)]
pub enum FieldElementType {
    U8,I8,
    U16,I16,
    U32,I32,
    U64,I64,
    F32,F64
}

impl TryFrom<&u8> for FieldElementType {
    type Error = std::io::Error;
    fn try_from(value: &u8) -> Result<Self, Self::Error> {
        FieldElementType::try_from(*value)
    }
}
impl TryFrom<u8> for FieldElementType {
    type Error = std::io::Error;
    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value { 
            b'b' => Ok(FieldElementType::I8),
            b'B' => Ok(FieldElementType::U8),
            b'h' => Ok(FieldElementType::I16),
            b'H' => Ok(FieldElementType::U16),
            b'i' => Ok(FieldElementType::I32),
            b'I' => Ok(FieldElementType::U32),
            b'l' => Ok(FieldElementType::I64),
            b'L' => Ok(FieldElementType::U64),
            b'f' => Ok(FieldElementType::F32),
            b'd' => Ok(FieldElementType::F64),
            _ => Err(std::io::Error::other(format!("Invalid byte in format: '{}'",value as char)))
        }
        
    }
}

impl FieldElementType {
    fn size_of(&self) -> usize {
        use FieldElementType::*;
        match self {
            U8 => 1,
            I8 => 1,
            U16 => 2,
            I16 => 2,
            U32 => 4,
            I32 => 4,
            U64 => 8,
            I64 => 8,
            F32 => 4,
            F64 => 8,
        }
    }
}

#[derive(Debug)]
pub enum FieldType {
    Value,
    Array
}

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
    ready : bool,
}

/// Writer for a single b-stream entry.
pub struct SerEntry<'a,'b,T> where T : Write {
    /// Underlying serializer object
    ser : &'b mut Ser<'a,T>,
    /// Indicates if the entry is ready for writing 
    ready : bool,
    /// Position in current format of the next field to write.
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
        .try_fold(0,
              |pb,&b| 
                  match b {
                      b'b'| b'B'| b'h'| b'H'| 
                      b'i'| b'I'| b'l'| b'L'| 
                      b'f'| b'd' => Some(b),
                      b'[' => if pb == b'[' { None } else { Some(b) },
                      _ => None,
                  })
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
            ready : true,
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

impl<'a,T> Drop for Ser<'a,T> where T : Write {
    fn drop(&mut self) {
        self.finalize().unwrap();
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
            self.end_field();
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
       
        let n = data.len()*size_of::<E>();
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
            self.end_field();
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

    fn end_field(&mut self) {
        self.ready = true; 
        self.ser.entry_active = false;
        if self.fmtpos == self.ser.curfmt.len() {
            self.ready = false;
            self.ser.ready = true;
        }
    }
} 


impl<'a,'b,'c,T,E> SerEntryChunkWriter<'a,'b,'c,T,E> where T : Write, E : Serializable, T : Write {
    /// Terminate the stream. No subsequent writes are allowed.
    pub fn close(&mut self) -> std::io::Result<()> {
        if self.ready {
            self.ent.ser.w.write_all(&[0,0])?;
            self.ent.end_field();
            self.ready = false;
        }
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
            println!("Write chunk size: {} ({}) ",n,E::sig() as char);
            bdata = &bdata[n..];
        }
        Ok(())
    }
}

impl<'a,'b,'c,T,E> Drop for SerEntryChunkWriter<'a,'b,'c,T,E> where T : Write, E : Serializable, T : Write {
    fn drop(&mut self) {
        self.close().unwrap();
    }
}

/// Trait defining anything that a type are can be written to a b-stream.
pub trait Serializable { fn sig() -> u8; }

impl Serializable for u8   { fn sig() -> u8 { b'B' } }
impl Serializable for i8   { fn sig() -> u8 { b'b' } }
impl Serializable for u16  { fn sig() -> u8 { b'H' } }
impl Serializable for i16  { fn sig() -> u8 { b'h' } }
impl Serializable for u32  { fn sig() -> u8 { b'I' } }
impl Serializable for i32  { fn sig() -> u8 { b'i' } }
impl Serializable for u64  { fn sig() -> u8 { b'L' } }
impl Serializable for i64  { fn sig() -> u8 { b'l' } }
impl Serializable for f32  { fn sig() -> u8 { b'f' } }
impl Serializable for f64  { fn sig() -> u8 { b'd' } }




//------------------------------------------------------------------




/// Structure for parsing a b-stream.
pub struct Des<'a,R> where R : Read {
    /// Underlying stream reader
    r : &'a mut R,
   
    /// Indicates if data entries are in native endian or should be byte-swapped. We currently
    /// ignore it and hope everyone are using little-endian.
    #[allow(unused)]
    byte_swap : bool,

    /// Indicates if an entry is currently partially read.
    entry_active : bool,

    /// Indicates EOF
    end_of_stream : bool
}

/// Structure for reading a single entry from a b-stream
pub struct DesEntry<'a,'b,R> where R : Read {
    /// Current entry format
    fmt : [u8;256],
    /// Current entry name
    name : [u8;256],
    /// Current opsition in fmt: Index of the format of the next field to be read.
    fmtpos : usize,
    /// Underlying deserializer object
    des : &'b mut Des<'a,R>,
    /// Entry is ready for reading next field.
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
    /// Current entry being read.
    entry : & 'c mut DesEntry<'a,'b,R>,
    /// Necessary because we need to use the `E` type in the struct or the compiler will complain.
    _t    : PhantomData<E>,
    /// Field kind
    kind  : EntryKind,
}

impl<'a,R> Des<'a,R> where R : Read {
    /// Create a new b-stream deserializer.
    pub fn new(r : &'a mut R) -> std::io::Result<Self> { 
        let mut bom = [0u32];
        r.read_exact(unsafe{ bom.align_to_mut().1 })?;

        let byte_swap = match bom[0] {
            BBOM => false,
            REV_BBOM => unimplemented!("Byte swapping in data"),
            _ => return Err(std::io::Error::other(format!("Invalid BOM 0x{:08x}",bom[0])))
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
            println!("Des::next_entry() name = '{}'",asciistr(&name[1..len+1]));
        }
        self.r.read_exact(&mut fmt[..1])?;
        if fmt[0] > 0 {
            let len = fmt[0] as usize;
            self.r.read_exact(&mut fmt[1..1+len])?;
            println!("Des::next_entry() sig = '{}'",asciistr(&fmt[1..len+1]));
            if ! validate_signature(&fmt[1..1+fmt[0] as usize]) {
                
                return std::str::from_utf8(&fmt[1..1+fmt[0] as usize])
                    .map_err(|_| std::io::Error::other(format!("Invalid signature: {}", std::str::from_utf8(fmt[1..1+fmt[0] as usize].iter().map(|&b| if (32..128).contains(&b) { b } else { b'?' }).collect::<Vec<u8>>().as_slice()).unwrap())))
                    .and_then(|s| Err(std::io::Error::other(format!("Invalid signature: {}",std::str::from_utf8(&fmt[1..1+fmt[0] as usize]).unwrap_or("<?>")))));
            }
        }
        
        if name[0] == 0 && fmt[0] == 0 {
            self.end_of_stream = true;
            Ok(None)
        }
        else {
            Ok(Some(DesEntry{des : self, fmt, name, fmtpos : 1, ready : true }))
        }
    }

    pub fn expect<'b>(&'b mut self, name : &[u8]) -> std::io::Result<DesEntry<'a,'b,R>> {
        match self.next_entry()? {
            None => Err(std::io::Error::other("Expected a entry, got end-of-file")),
            Some(v) => if v.name() == name { 
                Ok(v) 
            } else {
                Err(std::io::Error::other(format!("Expected a entry '{}'", std::str::from_utf8(name).unwrap_or("<invalid utf-8>")))) 
            },
        }
    }

    fn read_array_length(&mut self) -> std::io::Result<Option<usize>> {
        let mut buf = [0;7];
        self.r.read_exact(&mut buf[..1])?;
        let nb = (buf[0] >> 5) as usize;
        if nb == 7 {
            Ok(None)
        }
        else {
            buf[0] &= 0x1f;
            self.r.read_exact(&mut buf[1..nb+1])?;
            Ok(Some(buf[..nb+1].iter().fold(0,|v,&b| (v << 8) | (b as usize))))
        }
    }
}

impl<'a,'b,R> DesEntry<'a,'b,R> where R : Read {
    /// Return the entry name
    pub fn name(&self) -> &[u8] { &self.name[1..1+self.name[0] as usize] }
    /// Return the fmt name
    #[allow(unused)]
    pub fn fmt(&self)  -> &[u8] { &self.fmt[1..1+self.fmt[0] as usize] }
    pub fn check_fmt(self, fmt : &[u8]) -> std::io::Result<Self> {
        if self.fmt() != fmt { Err(std::io::Error::other(format!("Expected entry in format '{}', got '{}'",
                                                               std::str::from_utf8(fmt).unwrap_or("<invalid utf-8>"), 
                                                               std::str::from_utf8(self.fmt()).unwrap_or("<?>")))) }
        else { Ok(self) }
    }
 
    pub fn skip_field(&mut self) -> std::io::Result<()> { 
        let mut readbuf = [0;4096];
        if !self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmt[self.fmtpos+1] == b'[' {
            if self.fmtpos+1 >= self.fmt[0] as usize  {
                return Err(std::io::Error::other("Invalid format"));
            }
            self.fmtpos += 2;
            if let Some(size) = self.des.read_array_length()? {
                let mut size = size;
                while size > 0 {
                    let n = size.min(readbuf.len());
                    self.des.r.read_exact(&mut readbuf[..n])?;
                    size -= n;
                }
            }
            else {
                let mut chunksize = 0;
                loop {
                    if chunksize == 0 {
                        self.des.r.read_exact(&mut readbuf[..2])?;
                        chunksize = ((readbuf[0] as usize) << 8) + readbuf[1] as usize;
                        if chunksize == 0 { break; }
                    }
                    else {
                        let n = chunksize.min(readbuf.len());
                        self.des.r.read_exact(&mut readbuf[..n])?;
                        chunksize -= n;
                    }
                }
            }
        }       
        else {
            let n = FieldElementType::try_from(self.fmt[self.fmtpos+1])?.size_of();
            self.fmtpos += 1;
            self.des.r.read(&mut readbuf[..n])?;
        }
        Ok(())
    }
    pub fn skip_all(&mut self) -> std::io::Result<()> {
        while self.fmtpos < self.fmt[0] as usize {
            self.skip_field()?;
        }
        Ok(())
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

    pub fn field_type(&self) -> std::io::Result<Option<(FieldType,FieldElementType)>> {
        if !self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == self.fmt[0] as usize + 1 { return Ok(None); }

        //println!("DesEntry::field_type() fmt = '{}'",std::str::from_utf8(&self.fmt[self.fmtpos..self.fmt[0] as usize + 1]).unwrap_or("<?>"));
    
        if self.fmt[self.fmtpos] == b'[' {
            let fet = self.fmt.get(self.fmtpos+1)
                .ok_or_else(|| std::io::Error::other("Invalid fmt string"))
                .and_then(|c| c.try_into())?;
            Ok(Some((FieldType::Array,fet)))
        }
        else {
            Ok(Some((FieldType::Value,self.fmt[self.fmtpos].try_into()?)))
        }

        
    }

    pub fn read_into<E>(&mut self, res : &mut Vec<E>) -> std::io::Result<usize> 
        where 
            E : Serializable+Default+Copy 
    {
        if let Some(mut r) = self.next::<E>()? {
            let mut initial_length = res.len();
            loop {
                let base = res.len(); res.resize(res.len()+4096,E::default());
                let n = r.read(&mut res[base..])?;
                res.truncate(base+n);
                if n == 0 { break; }
            }
            Ok(res.len()-initial_length)
        }
        else {
            Err(std::io::Error::other("Read beyond end of entry"))
        }
    }
    pub fn read<E>(&mut self) -> std::io::Result<Vec<E>>
        where 
            E : Serializable+Default+Copy 
    {
        let mut res = Vec::new();
        self.read_into(&mut res).and_then(|_| Ok(res))
    }
    

    /// Get a reader for the next field. It may be necessary to specify the type of the field,
    /// which must match the signature.
    pub fn next<'c,E>(&'c mut self) -> std::io::Result<Option<DesEntryReader<'a,'b,'c,R,E>>> where E : Serializable+Default+Copy { 
        if !self.ready { return Err(std::io::Error::other("Entry not ready")); }
        if self.fmtpos == 1+self.fmt[0] as usize { return Ok(None); }
        
        let kind = 
            match self.fmt[self.fmtpos] {
                b'[' => {
                    let b = self.fmt[self.fmtpos+1];
                    if E::sig() != b { return Err(std::io::Error::other(format!("Incorrect field type requested: {:?}, expected {:?}",E::sig(), b))); }

                    self.fmtpos += 2;
                    self.ready = false;

                    if let Some(size) = self.des.read_array_length()? {
                        EntryKind::Array(size)
                    }
                    else {
                        EntryKind::Stream(0)
                    }
                },
                b => {
                    if E::sig() != b {
                        return Err(std::io::Error::other(format!("Incorrect field type requested: {:?}, expected {:?}",E::sig() as char, b as char)));
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
            _t : Default::default() }))
    }
}

impl<'a,'b,'c,R,E> DesEntryReader<'a,'b,'c,R,E> where R : Read, E : Serializable+Default+Copy {
    /// Read from the net field.
    pub fn read(&mut self, buf: &mut [E]) -> io::Result<usize> {
        match self.kind {
            EntryKind::Empty => Ok(0),
            EntryKind::Value => {
                //println!("DesEntryReader::read() value");
                if let Some(buf) = buf.get_mut(..1) {
                    self.entry.des.r.read_exact(unsafe{ buf.align_to_mut().1 })?;
                    self.kind = EntryKind::Empty;
                    self.entry.ready = true;
                    Ok(1)
                }
                else {
                    Ok(0)
                }
            },
            EntryKind::Array(nleft) => {
                //println!("DesEntryReader::read() array[{}]",nleft/size_of::<E>());
                let nelmleft = nleft/size_of::<E>();
                let n = nelmleft.min(buf.len());
                self.entry.des.r.read_exact(unsafe{ buf[..n].align_to_mut().1 })?;

                if nleft == n*size_of::<E>() {
                    self.entry.ready = true; 
                    self.kind = EntryKind::Empty 
                } 
                else { 
                    self.kind = EntryKind::Array(nleft-n) 
                };
                //println!("DesEntryReader::read() -> {}/{} elements", n/size_of::<E>(),buf.len());
                Ok(n)
            },
            EntryKind::Stream(nleft) => {
                println!("DesEntryReader::read() stream, cur chunk : {}",nleft);
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
                            self.entry.ready = true;
                            break;
                        }
                        println!("DesEntryReader::read() stream loop: next chunk : {}",chunk_left);
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

    pub fn read_vec(&mut self) -> io::Result<Vec<E>> {
        let mut res = Vec::new();
        self.read_all(&mut res)?;
        Ok(res)
    }
    pub fn read_all(&mut self, buf: &mut Vec<E>) -> io::Result<usize> {
        let len0 = buf.len();
        loop {
            let base = buf.len();
            buf.resize(base+4096/size_of::<E>(),Default::default());
            let n = self.read(&mut buf[base..])?;
            buf.truncate(base+n);
            if n == 0 { break; }
        }
        Ok(buf.len()-len0)
    }
}


fn asciistr(bs : &[u8]) -> String {
    let mut res = String::new();
    for b in bs.iter() {
        res.push(
            if (32..128).contains(b) { *b as char }
            else { '.' });
    }
    res
}


#[cfg(test)]
mod test {
    use std::fs::File;

    use super::*;

    #[test]
    fn test_ser_des_1() {
        use FieldElementType::*;
        let mut f = File::open("tests/lo1-sol.b").unwrap();
        let mut d = Des::new(&mut f).unwrap();

        while let Some(mut entry) = d.next_entry().unwrap() {
            println!("Entry: {} ({})",
                     std::str::from_utf8(entry.name()).unwrap(),
                     std::str::from_utf8(entry.fmt()).unwrap());
            
            while let Some((ft,et)) = entry.field_type().unwrap() {
                println!("  Field type: {:?} of {:?}",ft,et);
                match et {
                    U8  => { let mut buf : Vec<u8>  = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{}'", std::str::from_utf8(buf.as_slice()).unwrap_or("<?>")); },
                    I8  => { let mut buf : Vec<i8>  = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    U16 => { let mut buf : Vec<u16> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    I16 => { let mut buf : Vec<i16> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    U32 => { let mut buf : Vec<u32> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    I32 => { let mut buf : Vec<i32> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    U64 => { let mut buf : Vec<u64> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    I64 => { let mut buf : Vec<i64> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    F32 => { let mut buf : Vec<f32> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                    F64 => { let mut buf : Vec<f64> = Vec::new(); entry.next().unwrap().unwrap().read_all(& mut buf).unwrap(); println!("  Field data: '{:?}'",buf); },
                }
            }
        }
        println!("Deserialization done")
    }
    
    #[test]
    fn test_ser_des_2() {
        use FieldElementType::*;
        let mut f = File::open("tests/lo1-sol.b").unwrap();
        let mut d = Des::new(&mut f).unwrap();

        while let Some(mut entry) = d.next_entry().unwrap() {
            println!("Field: {} ({})",
                     std::str::from_utf8(entry.name()).unwrap(),
                     std::str::from_utf8(entry.fmt()).unwrap());
            
            while let Some((_ft,et)) = entry.field_type().unwrap() {
                match et {
                    U8  => _ = entry.read::<u8> ().unwrap(), 
                    I8  => _ = entry.read::<i8> ().unwrap(),
                    U16 => _ = entry.read::<u16>().unwrap(),
                    I16 => _ = entry.read::<i16>().unwrap(),
                    U32 => _ = entry.read::<u32>().unwrap(),
                    I32 => _ = entry.read::<i32>().unwrap(),
                    U64 => _ = entry.read::<u64>().unwrap(),
                    I64 => _ = entry.read::<i64>().unwrap(),
                    F32 => _ = entry.read::<f32>().unwrap(),
                    F64 => _ = entry.read::<f64>().unwrap(),
                }
            }
        }
        println!("Deserialization done")
    }

    #[test]
    fn test_ser_des_3() {
        let mut data : Vec<u8> = Vec::new();
        {
            let mut s = Ser::new(&mut data).unwrap();
            s.entry(b"INFO", b"[B").unwrap().write_array(b"This is a test!").unwrap();
            s.entry(b"SomeData", b"[B[i[d").unwrap()
                .write_array(b"Blablabla blabla bla").unwrap()
                .write_array::<i32>(&[1,2,3,4,5,6,7,8,9]).unwrap()
                .write_array(&[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
            {
                let mut e = s.entry(b"StreamTest1",b"[d").unwrap();
                { 
                    let mut w = e.stream_writer::<f64>().unwrap();
                    w.write(&[1.1,1.2,1.3,1.4,1.5,1.6]).unwrap();
                    w.write(&[2.1,2.2,2.3,2.4,2.5,2.6]).unwrap();
                    w.write(&[3.1,3.2,3.3,3.4,3.5,3.6]).unwrap();
                    w.close().unwrap();
                }            
            }
            {
                let mut e = s.entry(b"StreamTest2",b"[B[d").unwrap();
                { 
                    let mut w = e.stream_writer::<u8>().unwrap();
                    w.write(b"asdsfdfasdfdasfdsf").unwrap();
                    w.write(b"qwerqewrqewrqewrqe").unwrap();
                    w.write(b"213423421342134213").unwrap();
                    w.close().unwrap();
                }
                { 
                    let mut w = e.stream_writer::<f64>().unwrap();
                    w.write(&[1.1,1.2,1.3,1.4,1.5,1.6]).unwrap();
                    w.write(&[2.1,2.2,2.3,2.4,2.5,2.6]).unwrap();
                    w.write(&[3.1,3.2,3.3,3.4,3.5,3.6]).unwrap();
                    w.close().unwrap();
                }            
            }
            s.finalize().unwrap();
        }
        {
            let mut r = std::io::Cursor::new(data);
            let mut d = Des::new(&mut r).unwrap();
            {
                let mut entry = d.expect(b"INFO").unwrap().check_fmt(b"[B").unwrap();
                let data = entry.next::<u8>().unwrap().unwrap().read_vec().unwrap();
                println!(" INFO data = {}",asciistr(data.as_slice()));
            }
            {
                let mut entry = d.expect(b"SomeData").unwrap().check_fmt(b"[B[i[d").unwrap();

                let data1 = entry.next::<u8>().unwrap().unwrap().read_vec().unwrap();
                let data2 = entry.next::<i32>().unwrap().unwrap().read_vec().unwrap();
                let data3 = entry.next::<f64>().unwrap().unwrap().read_vec().unwrap();
                println!(" SomeData\n\tdata1 = {},\n\tdata2 = {:?}\n\tdata3 = {:?}",asciistr(data1.as_slice()),data2,data3);
            }

            {
                let mut entry = d.expect(b"StreamTest1").unwrap().check_fmt(b"[d").unwrap();

                let data2 = entry.next::<f64>().unwrap().unwrap().read_vec().unwrap();
                println!(" StreamTest\n\tdata2 = {:?}",data2);
            }
            {
                let mut entry = d.expect(b"StreamTest2").unwrap().check_fmt(b"[B[d").unwrap();

                let data1 = entry.next::<u8>().unwrap().unwrap().read_vec().unwrap();
                let data2 = entry.next::<f64>().unwrap().unwrap().read_vec().unwrap();
                println!(" StreamTest\n\tdata1 = {},\n\tdata2 = {:?}",asciistr(data1.as_slice()),data2);
            }
        }
    }
}
