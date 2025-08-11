//! This module implements a pipe-like object that allows us to "write" to a reqwest request and
//! "read" a response as if they were streams. There must certainly be a better way to do this.
//! maybe something async?
//!
//! Implements a very simple pipe using shared memory, [std::sync::Arc], [std::sync::Condvar] and
//! [std::sync::Mutex]
//! .
use std::io::{Read,Write};
use std::sync::{Arc, Condvar, Mutex};

pub struct IOBuffer{
    data   : [u8;4096], 
    base   : usize,
    top    : usize,
    eof    : bool,
}

pub struct IOPipe {
    filled : Condvar,
    empty  : Condvar,
    buf    : Mutex<IOBuffer>,
}

pub struct IOPipeReader {
    pipe : Arc<IOPipe>
}

pub struct IOPipeWriter {
    pipe : Arc<IOPipe>
}

impl IOPipeReader {
    fn close(&mut self) {
        let mut buf = self.pipe.buf.lock().unwrap();
        buf.eof = true;
        self.pipe.filled.notify_one();
        self.pipe.empty.notify_one();
    }
}


impl IOPipeWriter {
    fn close(&mut self) {
        let mut buf = self.pipe.buf.lock().unwrap();
        buf.eof = true;
        self.pipe.filled.notify_one();
        self.pipe.empty.notify_one();
    }
}

impl Drop for IOPipeReader {
    fn drop(&mut self) {
        self.close();
    }
}
impl Drop for IOPipeWriter {
    fn drop(&mut self) {
        self.close();
    }
}

impl Read for IOPipeReader {
    fn read(&mut self, dst: &mut [u8]) -> std::io::Result<usize> {
        let p = &*(self.pipe);        
        let mut buf = p.buf.lock().unwrap(); 

        if buf.base < buf.top {
            let n = dst.len().min(buf.top-buf.base);
            dst.copy_from_slice(&buf.data[..n]);
            buf.base += n;
            if buf.base == buf.top { 
                buf.base = 0; 
                buf.top = 0; 
                p.empty.notify_all();
            }
            Ok(n)
        }
        else if buf.eof { 
            Ok(0)
        }
        else 
        {
            let mut buf = p.filled.wait(buf).unwrap();
            let n = dst.len().min(buf.top-buf.base);
            dst.copy_from_slice(&buf.data[..n]);
            buf.base += n;
            if buf.base == buf.top { buf.base = 0; buf.top = 0; }
            Ok(n)
        }
    }
}

impl Write for IOPipeWriter {
    fn write(&mut self, src: &[u8]) -> std::io::Result<usize> {
        let p = &*self.pipe;
        let buf = p.buf.lock().unwrap(); 
        if buf.eof {
            Ok(0)
        } 
        else {
            let mut buf = 
                if buf.top == buf.data.len() {
                    p.empty.wait(buf).unwrap()
                }
                else {
                    buf
                };
            let n = src.len().min(buf.data.len()-buf.top);
            let top = buf.top;
            buf.data[top..top+n].copy_from_slice(&src[..n]);
            buf.top += n;
            if buf.top > buf.base {
                p.filled.notify_all();
            }
            Ok(n)
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl IOPipe {
    fn new() -> IOPipe {
        IOPipe{
            filled : Condvar::new(),
            empty  : Condvar::new(),
            buf    : Mutex::new(IOBuffer { data: [0u8;4096], base: 0, top: 0, eof: false })
        }
    }
}

/// Create a new pair of (input,output)
pub fn new() -> (IOPipeReader,IOPipeWriter) {
    let pr = Arc::new(IOPipe::new());
    let pw = pr.clone();

    (IOPipeReader{pipe:pr},IOPipeWriter{pipe:pw})
}


