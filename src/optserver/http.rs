//! A very basic http module, only implements just enough for simple communication with an
//! optserver.
//!

use std::net::TcpStream;
use std::io::{Read, Write};
use std::ops::Range;
use std::usize;


enum Method {
    GET,
    PUT,
    POST,
    HEAD
}

/// HTTP 1.1 request. 
///
/// # Example
/// ```
/// fn test(s : &mut TcpStream) {
///   Request::get("/")
///     .add_header("X-Mosek-Access","SECRET TOKEN")
///     .submit(s)
///     .
/// }
/// ```
/// 
///
#[derive(Clone)]
struct Request {
    header : Vec<u8>,

    content_length : Option<usize>,
    transfer_coding_chunked : bool,
}

struct Response<'a,T> where T : Read {
    proto_ver : (u8,u8),
    code : u16,
    reason : Range<usize>,
    header : Vec<u8>,
    header_start : usize,
    header_end : usize,

    chunked : bool,
    content_length : Option<usize>,

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
        MsgWriter { chunked: nbytes.is_none(), remains: nbytes.unwrap_or(0), pos: 4, buffer: [0;0x1000], s }
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
                let (b0,b1) = hex_byte(((self.pos-4) & 0xff) as u8);
                let (b2,b3) = hex_byte(((self.pos-4) >> 8) as u8);
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

    pub fn content_length(&mut self, size : usize) -> &mut Self{
        self.content_length = Some(size);
        self
    }

    pub fn add_header(&mut self, key : &str, value : &str) -> &mut Self{
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
                write!(self.header,"Content-Length: {}\r\n",len);
            }
        }
        else {
            write!(self.header,"Content-Length: {}\r\n",data.len());
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

        let (code,proto_ver,msg_range,hdrs_start) = {
            let mut code = 0u16;
            // read first line in the format "'HTTP/' [0-9] '.' [0-9]" [ ]+ [0-9]+ [ ]+ .*
            if !buf.starts_with(b"HTTP/") { return Err("Invalid HTTP status line".to_string()); }
            let mut it = buf[5..].iter().enumerate().peekable();
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
            let hdrs_start = message_end+1;

            (code,(proto_v1,proto_v2),message_start..message_end-1,hdrs_start)
        };

        let mut resp = Response{
            proto_ver,
            code,
            reason : msg_range,
            header : buf,
            header_start : hdrs_start,
            header_end : lastcrlfpos-2,

            content_length : None,
            chunked : false,

            s
        };

        for line in resp.header.chunk_by(|c,_| *c == b'\n') {
            let pos = line.iter().enumerate().find(|c| *c.1 == b':').ok_or_else(|| "Invalid HTTP header".to_string())?.0;
            let key = &line[..pos];
            let value = line[pos..].trim_ascii();

            if      key.eq_ignore_ascii_case(b"content-length")  { 
                if let Some(v) = std::str::from_utf8(value).ok().and_then(|s| s.parse::<usize>().ok()) {
                    resp.content_length = Some(v); 
                }
                else {
                    return Err("Invalid Content-Length value".to_string());
                }
            }
            else if key.eq_ignore_ascii_case(b"transfer-coding") { resp.chunked = value.eq_ignore_ascii_case(b"chunked"); }
        }

        Ok(resp)
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
            assert_eq!(buf.len(),213);
            assert_eq!(&buf[0..6],b"0064\r\n");
            assert_eq!(&buf[104..108],b"64\r\n");
        }
    }
}
