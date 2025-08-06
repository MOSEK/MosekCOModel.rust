

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
