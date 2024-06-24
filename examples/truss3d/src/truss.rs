use std::io::{BufRead,BufReader};
use std::fs::File;

#[derive(Clone,Default)]
pub struct Truss {
    pub points         : Vec<[f64;3]>,
    pub node_type      : Vec<bool>,
    pub arcs           : Vec<(usize,usize)>,
    pub external_force : Vec<Vec<[f64;3]>>,
    pub total_material_volume : f64,
    pub kappa          : f64,
}

impl Truss {
    /// Read file. File format:
    /// ```
    /// kappa FLOAT
    /// w     FLOAT
    /// nodes
    ///     FLOAT FLOAT FLOAT "X"?
    ///     ...
    /// arcs
    ///     INT INT
    ///     ...
    /// forces
    ///     INT FLOAT FLOAT FLOAT
    ///     ...
    /// forces ...
    ///     ...
    /// ```
    pub fn from_file(filename : &str) -> Truss {
        enum State {
            Base,
            Nodes,
            Arcs,
            Forces,
        }
        let mut dd = Truss::default();

        let f = File::open(filename).unwrap();
        let br = BufReader::new(f);
        let mut state = State::Base;
        let mut forces : Vec<Vec<(usize,[f64;3])>> = Vec::new();

        for (lineno,l) in br.lines().enumerate() {
            let l = l.unwrap();

            if      let Some(rest) = l.strip_prefix("kappa")  { dd.kappa = rest.trim().parse().unwrap(); state = State::Base; }
            else if let Some(rest) = l.strip_prefix("w")      { dd.total_material_volume = rest.trim().parse().unwrap(); state = State::Base; }
            else if l.starts_with("nodes")  { state = State::Nodes; }
            else if l.starts_with("arcs")   { state = State::Arcs; }
            else if l.starts_with("forces") { forces.push(Vec::new()); state = State::Forces; }
            else {
                let llstrip = l.trim_start();
                if llstrip.is_empty() || llstrip.starts_with('#') {
                    // comment
                }
                else if ! l.starts_with(' ') {
                    panic!("Invalid data at line {}: '{}'",lineno+1,l.trim_end());
                }
                else {
                    match state {
                        State::Base   => {},
                        State::Nodes  => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let x : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let y : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let z : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            if let Some("X") = it.next() { dd.node_type.push(true); }
                            else { dd.node_type.push(false); }
                            dd.points.push([x,y,z]);
                        },
                        State::Arcs   => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let i : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let j : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            dd.arcs.push((i,j));
                        },
                        State::Forces => {
                            let mut it = l.trim().split(' ').filter(|v| v.len() > 0);
                            let a : usize = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let x : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let y : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            let z : f64 = it.next().unwrap_or_else(|| panic!("Missing data at line {}",lineno)).parse().expect("Invalid data");
                            forces.last_mut().unwrap().push((a,[x,y,z]));
                        }
                    }
                }
            }
        }
        
        // check

        if forces.is_empty() {
            panic!("Missing forces section");
        }
        if *dd.arcs.iter().map(|(i,j)| i.max(j)).max().unwrap() >= dd.points.len() {
            panic!("Arc end-point index out of bounds");
        }

        for ff in forces.iter() {
            if ff.iter().map(|v| v.0).max().unwrap() >= dd.points.len() {
                panic!("Force node index out of bounds");
            }

            let mut forcevec = vec![[0.0,0.0,0.0]; dd.points.len()];
            for &(i,f) in ff { forcevec[i] = f; }

            dd.external_force.push(forcevec);
        }
        println!("Truss:\n\t#nodes: {}\n\t#arcs: {}",dd.points.len(),dd.arcs.len());

        dd 
    }
}

