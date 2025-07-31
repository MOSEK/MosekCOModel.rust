
use mosekcomodel::*;

fn lo1() {
    let addr = "solve.mosek.com:30080".to_string();
    let mut m = optserver::Model::new(Some("SuperModel"));
    m.set_parameter((), optserver::SolverAddress(addr));

    let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
    let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
    let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
    let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

    // Create variable 'x' of length 4
    let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

    // Create constraints
    let _ = m.constraint(None, x.index(1), less_than(10.0));
    let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
    let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
    let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

    // Set the objective function to (c^t * x)
    m.objective(Some("obj"), Sense::Maximize, x.dot(c));

    // Solve the problem
    m.write_problem("lo1-nosol.json");
    m.solve();

    // Get the solution values
    let (psta,dsta) = m.solution_status(SolutionType::Default);
    println!("Status = {:?}/{:?}",psta,dsta);
    let xx = m.primal_solution(SolutionType::Default,&x);
    println!("x = {:?}", xx);
}

fn main() {
    lo1();
}

#[test]
fn test() { main() }
