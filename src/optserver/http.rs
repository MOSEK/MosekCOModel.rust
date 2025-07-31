//! A very basic http module, only implements just enough for simple communication with an
//! optserver.
//!
//!

use std::io::{Read, Write};

/// HTTP 1.1 request. 
///
///
#[derive(Clone)]
pub struct Request {
    header : Vec<u8>,
    content_length : Option<usize>,
    transfer_coding_chunked : bool,
}

pub struct Response<'a,T> where T : Read {
    proto_ver : (u8,u8),
    code : u16,
    reason : String,
    header : Vec<u8>,
    headers : Vec<(usize,usize,usize,usize)>,

    chunked : bool,
    chunk_remains : usize,
    content_length : Option<usize>,
    readbuf : Vec<u8>,
    pos : usize,
    eof : bool,

    s : & 'a mut T,
}


pub struct MsgWriter<'a,T> where T : std::io::Write{ 
    chunked : bool,
    remains : usize,
    pos    : usize,
    buffer : [u8;0x1000], // TODO: reserve bytes at beginning for SIZE\r\n, and 4 at the end for
                          // \r\n\r\n
    s : & 'a mut T,
}

impl<'a,T> MsgWriter<'a,T> where T : std::io::Write {
    fn new(nbytes : Option<usize>, s : &'a mut T) -> MsgWriter<'a,T> 
    {
        MsgWriter { chunked: nbytes.is_none(), remains: nbytes.unwrap_or(0), pos: 6, buffer: [0;0x1000], s }
    }
    pub fn write(&mut self, data : &[u8]) -> std::io::Result<()> {
        if !self.chunked && data.len() > self.remains {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Write beyond Content-Length".to_string()));
        }

        if self.pos+data.len() < self.buffer.len() {
            self.buffer[self.pos..self.pos+data.len()].copy_from_slice(data);
            self.pos += data.len();
        }
        else {
            let mut datap = 0;
            while datap < data.len() {
                if self.pos == self.buffer.len() {
                    self.flush()?;
                }

                let nw = (data.len()-datap).min(self.buffer.len()-self.pos);
                self.buffer[self.pos..self.pos+nw].copy_from_slice(&data[datap..datap+nw]);
                datap += nw;
                self.pos += nw;
            }
        }
        if ! self.chunked {
            self.remains -= data.len();
        }
        Ok(())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self.pos > 0 {
            if self.chunked {
                let (b0,b1) = hex_byte(((self.pos-6) & 0xff) as u8);
                let (b2,b3) = hex_byte(((self.pos-6) >> 8) as u8);
                self.buffer[0] = b3;
                self.buffer[1] = b2;
                self.buffer[2] = b1;
                self.buffer[3] = b0;
                self.buffer[4] = b'\r';
                self.buffer[5] = b'\n';
                self.s.write_all(&self.buffer[..self.pos])?;
            }
            else {
                self.s.write_all(&self.buffer[6..self.pos])?;
            }
            self.pos = 6;
        }
        Ok(())
    }

    fn finalize(&mut self) -> Result<(),String> {
        self.flush().map_err(|e| e.to_string())?;
        if !self.chunked {
            if self.remains > 0 {
                return Err("Incomplete message body".to_string());                
            }
        }            
        else {
            self.s.write_all(b"0\r\n\r\n").map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

impl<'a,T> Write for MsgWriter<'a,T> where T : Write {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.write(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.flush()
    }
}


impl Request {
    fn new(method : &str, path : &str) -> Request { 
        let mut r = Request{ header : Vec::new(),content_length : None, transfer_coding_chunked : false };
        r.header.extend_from_slice(method.as_bytes());
        r.header.push(b' ');
        r.header.extend_from_slice(path.as_bytes());
        r.header.push(b' ');
        r.header.extend_from_slice(b"HTTP/1.1\r\n");
        r
    }

    pub fn get(path : &str)  -> Request { Request::new("GET",path) }
    pub fn put(path : &str)  -> Request { Request::new("PUT",path) }
    pub fn post(path : &str) -> Request { Request::new("POST", path) }
    pub fn head(path : &str) -> Request { Request::new("PUT", path) }

    pub fn content_length(mut self, size : usize) -> Self{
        self.content_length = Some(size);
        self
    }

    pub fn add_header(mut self, key : &str, value : &str) -> Self{
        if key.eq_ignore_ascii_case("content-length") {
            if let Ok(v) = value.parse::<usize>() {
                self.content_length = Some(v);
            }
        }
        else if key.eq_ignore_ascii_case("transfer-coding") {
            if value.eq_ignore_ascii_case("chunked") {
                self.transfer_coding_chunked = true;
            }
        }
        else {
            self.header.extend_from_slice(key.as_bytes());
            self.header.extend_from_slice(b": ");
            self.header.extend_from_slice(value.as_bytes());
            self.header.extend_from_slice(b"\r\n");
        }
        self
    }
    
    pub fn submit_empty<'a,T>(mut self, s : &'a mut T) -> Result<Response<'a,T>,String> 
        where T : Read+Write
    {       
        self.header.extend_from_slice(b"Content-Length: 0\r\n");
        self.header.extend_from_slice(b"\r\n");

        s.write_all(self.header.as_slice()).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        self.wait_response(s)
    }


    pub fn submit_data<'a,T>(mut self, s : &'a mut T, data : &[u8]) -> Result<Response<'a,T>,String> 
        where T : Read+Write
    {
        if let Some(len) = self.content_length {
            if data.len() != len {
                return Err("Mismatching Content-Length and data".to_string());
            }
            else {
                write!(self.header,"Content-Length: {}\r\n",len).map_err(|e| e.to_string())?;
            }
        }
        else {
            write!(self.header,"Content-Length: {}\r\n",data.len()).map_err(|e| e.to_string())?;
        }
        self.header.extend_from_slice(b"\r\n");

        s.write_all(self.header.as_slice()).map_err(|e| e.to_string()).map_err(|e| e.to_string())?;
        s.write_all(data).map_err(|e| e.to_string()).map_err(|e| e.to_string())?;
        s.flush().map_err(|e| e.to_string())?;

        self.wait_response(s)
    }

    pub fn submit_with_writer<'a,T,W>(mut self, s : & 'a mut T, body_writer : W) -> Result<(),String>
        where W : FnOnce(&mut MsgWriter<'a,T>) -> Result<(),String>,
              T : Write+Read
    {
        let mut msgw = MsgWriter::new(self.content_length, s);
        if let Some(len) = self.content_length {
            write!(self.header,"Content-Length: {}\r\n",len);
        }
        else {
            self.header.extend_from_slice(b"Transfer-Encoding: chunked\r\n"); 
        }

        body_writer(&mut msgw)
    }

    fn wait_response<'a,T>(&mut self, s : &'a mut T) -> Result<Response<'a,T>,String> 
        where T : Read+Write
    {
        let mut lastcrlfpos = 0usize;
        let buf = {
            let mut buf = Vec::new();
           
            // locate CR LF CR LF marking the end of the header
            'outer: loop {
                let top = buf.len();
                buf.resize(top+4096,0);
                let rn = s.read(&mut buf[top..]).map_err(|e| e.to_string())?;
                buf.resize(top+rn,0);
                while let Some(p) = buf[lastcrlfpos..].iter().enumerate().find(|(i,&b)| b == b'\n').map(|(i,_)| i) {
                    lastcrlfpos += p+1;

                    if p == 1 {
                        break 'outer;
                    }
                }
            }
            buf
        };
        let headers_begin = buf.iter().enumerate().find(|c| *c.1 == b'\n').unwrap().1+1;
        let mut headers = Vec::new();
        let mut content_length = None;
        let mut chunked = false;
       
        let mut line_iter = buf.chunk_by(|c,_| *c != b'\n').scan(0,|pos,line| { let start = *pos; *pos += line.len(); if line.len() == 2 { None } else { Some((start, line)) } });
        let (code,proto_ver,reason) = {            
            let (_,line) = line_iter.next().ok_or_else(|| "Invalid HTTP{ status line".to_string())?;
            println!("line = '{}'",std::str::from_utf8(line).unwrap());
            let mut code = 0u16;
            // read first line in the format "'HTTP/' [0-9] '.' [0-9]" [ ]+ [0-9]+ [ ]+ .*
            if !line.starts_with(b"HTTP/") { return Err("Invalid HTTP status line".to_string()); }
            let mut it = line[5..].iter().enumerate().peekable();
            let proto_v1 = it.next().and_then(|item| match item.1 { b@(b'0'..=b'9') => Some(b-b'0'), _ => None }).ok_or("Invalid HTTP status line".to_string())?;
            if b'.' != *it.next().ok_or_else(|| "Invalid HTTP status line".to_string())?.1 { return Err("Invalid HTTP status line".to_string()); }
            let proto_v2 = it.next().and_then(|item| match item.1 { b@(b'0'..=b'9') => Some(b-b'0'), _ => None }).ok_or("Invalid HTTP status line".to_string())?;
            if b' ' != *it.next().ok_or_else(|| "Invalid HTTP status line".to_string())?.1 { return Err("Invalid HTTP status line".to_string()); }
            if it.peek().is_some() {
                loop {
                    match it.next() {
                        Some((_,b @ (b'0'..=b'9'))) => code = code * 10 + (*b as u16),
                        Some((_,b' ')) => break,
                        _ => return Err("Invalid HTTP status line".to_string()),
                    }
                }
            }
            else {
                return Err("Invalid HTTP status line".to_string());       
            }
            
            let message_start = it.peek().unwrap().0+5;
            let message_end = it.find(|c| *c.1 == b'\n').unwrap().0+5+1;
            
            let reason = std::str::from_utf8(line[message_start..message_end].trim_ascii()).map_err(|_| "Invalid HTTP status line".to_string())?.to_string();
    

            (code,(proto_v1,proto_v2),reason)
        };

        for (start,line) in line_iter {
            let colon_pos : usize = line.iter().enumerate().find(|&item| *item.1 == b':').ok_or_else(|| "Invalid HTTP header".to_string())?.0;

            let key   = &line[..colon_pos];
            let value = &line[colon_pos+1..].trim_ascii();
            if key.eq_ignore_ascii_case(b"content-length") { 
                if let Some(v) = std::str::from_utf8(value).ok().and_then(|s| s.parse::<usize>().ok()) {
                    content_length = Some(v); 
                }
                else {
                    return Err("Invalid Content-Length value".to_string());
                }
            }
            else if key.eq_ignore_ascii_case(b"transfer-coding") { chunked = value.eq_ignore_ascii_case(b"chunked"); }

            headers.push((start,start+colon_pos,start+colon_pos+1,start+line.len()))                
        }

        let readbuf = buf[lastcrlfpos..].to_vec();

        Ok(Response{
            proto_ver,
            code,
            reason,
            header : buf,
            headers,

            content_length,
            chunked, 
            chunk_remains : 0,
            readbuf,
            pos : 0,
            eof : false,

            s
        })
    }
}

impl<'a,T> Response<'a,T> where T : Read 
{
    fn headers<'b>(&'b self) -> impl Iterator<Item=(&'b[u8],&'b[u8])> {
        self.headers.iter().map(|&(k0,k1,v0,v1)| (&self.header[k0..k1],self.header[v0..v1].trim_ascii()))
    }
}

impl<'a,T> Read for Response<'a,T> where T : Read {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.chunked {
            if self.eof { return Ok(0); }
            else if self.pos < self.readbuf.len() {
                // do nothing
            }
            else if self.chunk_remains == 0 {
                // read until CRLF
                let mut sizebuf = [0u8;2];
                let mut chunksize = 0usize;
                loop {
                    self.s.read_exact(&mut sizebuf[0..1])?;
                    match sizebuf[0] {
                        c @ b'0'..=b'9' => chunksize = (chunksize << 4) + (c-b'0') as usize,
                        c @ b'a'..=b'f' => chunksize = (chunksize << 4) + (c-b'a'+10) as usize,
                        c @ b'A'..=b'F' => chunksize = (chunksize << 4) + (c-b'A'+10) as usize,
                        b'\r' => break,
                        _ => return Err(std::io::Error::other("Invalid chunk header"))
                    }
                }
                self.s.read_exact(&mut sizebuf[0..1])?; 
                if sizebuf[0] != b'\n' { return Err(std::io::Error::other("Invalid chunk header")); }

                if chunksize == 0 {
                    self.s.read_exact(&mut sizebuf)?;
                    if &sizebuf != b"\r\n" { return Err(std::io::Error::other("Invalid chunk header")); }
                    self.eof = true;
                }
                self.chunk_remains = chunksize;
            }

            if self.pos == self.readbuf.len() {
                // read into buffer
                self.pos = 0;
                self.readbuf.resize(4096,0);
                let n = self.s.read(self.readbuf.as_mut_slice())?;
                self.readbuf.truncate(n);
                self.chunk_remains -= n;

                // read trailing CRLF
                if self.chunk_remains == 0 {
                    let mut sizebuf = [0u8;2];
                    self.s.read_exact(&mut sizebuf)?;
                    if &sizebuf != b"\r\n" { return Err(std::io::Error::other("Invalid chunk header")); }
                }
            }

            let n = buf.len().min(self.readbuf.len()-self.pos);
            buf[..n].copy_from_slice(&self.readbuf[self.pos..self.pos+n]);
            self.pos += n;
            Ok(n)
        }
        else {
            if self.pos == self.readbuf.len() {
                self.pos = 0;
                self.readbuf.resize(4096,0);
                let n = self.s.read(self.readbuf.as_mut_slice())?;
                self.readbuf.truncate(n);
            }

            if self.pos < self.readbuf.len() {
                let n = buf.len().min(self.readbuf.len()-self.pos);
                buf[..n].copy_from_slice(&self.readbuf[self.pos..self.pos+n]);
                self.pos += n;
                Ok(n)
            }
            else {
                Ok(0)
            }
        }
    }
}

//
// Utilities
//
fn hex_byte(b : u8) -> (u8,u8) {
    (match b & 0xf {
        b @ 0..=9 => b'0' + b,
        b => b'a' + b
    },
    match b >> 4 {
        b @ 0..=9 => b'0' + b,
        b => b'a' + b
    })
}

#[cfg(test)]
mod test {
    use super::*;

    struct RWBuf {
        outbuf : Vec<u8>,
        inbuf  : Vec<u8>,
        pos : usize,
    }

    impl RWBuf {
        fn new(readdata : &[u8]) -> RWBuf {
            RWBuf { outbuf: Vec::new(), inbuf: readdata.to_vec(), pos: 0 }
        }
    }

    impl Write for RWBuf {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.outbuf.write(buf)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Read for RWBuf {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.inbuf.len()-self.pos);
            buf[..n].copy_from_slice(&self.inbuf[self.pos..self.pos+n]);
            self.pos += n;
            Ok(n)
        }
    }

    #[test]
    fn test_msg_writer() {
        {
            let mut buf = Vec::new();
            let mut w = MsgWriter::new(Some(100),&mut buf);
            w.write(b"0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789").unwrap();
            w.finalize().unwrap();
            assert_eq!(buf.len(),100);
        }
        {
            let mut buf = Vec::new();
            let mut w = MsgWriter::new(Some(100),&mut buf);
            w.write(b"01234567890123456789012345678901234567890123456789").unwrap();
            assert!(w.finalize().is_err());
        }
        {
            let mut buf = Vec::new();
            let mut w = MsgWriter::new(Some(10),&mut buf);
            assert!(w.write(b"01234567890123456789012345678901234567890123456789").is_err());
        }
        {
            let mut buf = Vec::new();
            let mut w = MsgWriter::new(None,&mut buf);

            w.write(b"0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789").unwrap();
            w.flush().unwrap();
            w.write(b"0123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789").unwrap();
            w.finalize().unwrap();
            assert_eq!(buf.len(),212+5);
            assert_eq!(&buf[0..6],b"0064\r\n");
            assert_eq!(&buf[106..112],b"0064\r\n");
        }
        {
            let mut buf = RWBuf::new(b"HTTP/1.1 200 Ok\r\nContent-Length: 12\r\nX-Mosek-Token: SECRET\r\n\r\nabcdefghi\n\n");
            let mut resp = Request::post("/path/to/something")
                .content_length(12)
                .add_header("Secret-Data", "SUPER SECRET")
                .submit_data(&mut buf, b"MESSAGE BODY")
                .unwrap();
            for ((k,v),(kx,vx)) in resp.headers().zip([("Content-Length","12"),("X-Mosek-Token","SECRET")].iter()) {
                assert_eq!(std::str::from_utf8(k).unwrap(),*kx);
                assert_eq!(std::str::from_utf8(v).unwrap(),*vx);
            }

            let mut data = Vec::new();
            let _= resp.read_to_end(&mut data).map_err(|e| e.to_string()).unwrap();
            println!("data = '{}'",std::str::from_utf8(data.as_slice()).unwrap());
            assert_eq!(data.as_slice(),b"abcdefghi\n\n");

        }
    }
}
