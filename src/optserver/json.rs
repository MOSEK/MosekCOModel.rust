pub struct JSONStream<'a,T> where T : std::io::Write {
    s : &'a mut T,
    buf : [u8; 4096],
    pos : usize
}
pub trait JSONWritable<T> where T : std::io::Write { fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()>; }

impl<T> std::io::Write for JSONStream<'_,T> where T : std::io::Write {
    fn write(&mut self, data : &[u8]) -> std::io::Result<usize> {
        let nleft = self.buf.len()-self.pos;

        if nleft >= data.len() {
            self.buf[self.pos..self.pos+data.len()].copy_from_slice(data);
            return Ok(data.len())
        }
        
        let mut datapos = 0;
        if self.pos > 0 {
            self.buf[self.pos..].copy_from_slice(&data[..nleft])
        }
        self.flush()?;
        if self.buf.len()+nleft < data.len() {
            self.s.write_all(&data[nleft..])?;
        }
        else {
            self.buf[..data.len()-nleft].copy_from_slice(&data[nleft..]);
        }
       Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.flush()
    }

}

impl<'a,T> JSONStream<'a,T> where T : std::io::Write {
    pub fn new(s : &'a mut T) -> JSONStream<'a,T> {
        JSONStream{
            s,
            buf : [0u8;4096],
            pos : 0
        }
    }
    pub fn flush(&mut self) -> std::io::Result<()> {
        if self.pos > 0 {
            self.s.write_all(&self.buf[self.pos..])?;
            self.pos = 0;
        }
        Ok(())
    }

    pub fn putc(&mut self, c : u8) -> std::io::Result<()> {
        if self.pos == self.buf.len() { self.flush()?; }
        self.buf[self.pos] = c;
        self.pos += 1;
        Ok(())
    }

    pub fn write<D>(&mut self, data : D) -> std::io::Result<()> where D : JSONWritable<T> {
        data.write(self)
    }

    pub fn write_list<I,E>(&mut self, data : I) -> std::io::Result<()> where I : IntoIterator<Item=E>, E:JSONWritable<T> {
        self.putc(b'[')?;
        let mut it = data.into_iter();
        if let Some(v) = it.next() { v.write(self)?; }
        while let Some(v) = it.next() { self.putc(b',')?; v.write(self)?; }
        self.putc(b']')
    }

    pub fn with_dict<F>(&mut self, f : F)  -> std::io::Result<()> where F : FnOnce(&mut Self) -> std::io::Result<()> {
        self.putc(b'{')?;
        f(self)?;
        self.putc(b'}')
    }

}

fn hex(b : u8) -> (u8,u8) {
    let l = match b & 0xf {
        b @ (0..=9) => b'0'+b,
        b => b'a'+b-10
    };
    let u = match b >> 4 {
        b @ (0..=9) => b'0'+b,
        b => b'a'+b-10
    };
    (l,u)
}
impl<T> JSONWritable<T> for &str where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> {
        s.putc(b'"')?;
        for b in self.as_bytes().iter() {
            match b {
                b'\\' => { s.putc(b'\\')?; s.putc(b'\\')?; },
                b'"'  => { s.putc(b'\\')?; s.putc(b'"')?; },
                b'\n' => { s.putc(b'\\')?; s.putc(b'n')?; },
                b'\t' => { s.putc(b'\t')?; },
                b'\r' => { s.putc(b'\\')?; s.putc(b'r')?; },
                b'\r' => { s.putc(b'\\')?; s.putc(b'r')?; },
                0..31 => { s.putc(b'\\')?; s.putc(b'x')?; let (l,u) = hex(*b); s.putc(l)?; s.putc(u)?; },
                _ => s.putc(*b)?
            }
        }
        s.putc(b'"')?;
        Ok(())
    }
}

impl<T,D> JSONWritable<T> for &[D] where D : JSONWritable<T>+Copy, T : std::io::Write {
    fn write(self,s : &mut JSONStream<T>) -> std::io::Result<()> {
        s.putc(b'[')?;
        let mut it = self.iter();
        if let Some(v) = it.next() { (*v).write(s)?; }
        while let Some(v) = it.next() {
            s.putc(b',')?;
            (*v).write(s)?;
        }
        s.putc(b']')?;
        Ok(())
    }
}

impl<T> JSONWritable<T> for &f64 where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { std::io::Write::write_fmt(s,format_args!("{}",self)) }
}
impl<T> JSONWritable<T> for &usize where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { std::io::Write::write_fmt(s,format_args!("{}",self)) }
}
impl<T> JSONWritable<T> for &i32 where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { std::io::Write::write_fmt(s,format_args!("{}",self)) }
}

impl<T> JSONWritable<T> for f64 where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { (&self).write(s) }
}
impl<T> JSONWritable<T> for usize where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { (&self).write(s) }
}
impl<T> JSONWritable<T> for i32 where T : std::io::Write {
    fn write(self,s : &mut JSONStream<'_,T>) -> std::io::Result<()> { (&self).write(s) }
}
impl<T,D> JSONWritable<T> for (&str,D) where D : JSONWritable<T>, T : std::io::Write {
    fn write(self,s : &mut JSONStream<T>) -> std::io::Result<()> {
        self.0.write(s)?;
        s.putc(b':')?;
        self.1.write(s)?;
        Ok(())
    }
}
