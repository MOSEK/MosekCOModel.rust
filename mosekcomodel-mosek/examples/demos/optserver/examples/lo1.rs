use mosekcomodel::*;
use optserver::*;
use optserver::model::{OptserverHost,ModelOptserver};

type Model = ModelAPI<ModelOptserver>;
fn lo1() -> (Model,Variable<1>) {
    let a0 : &[f64] = &[ 3.0, 1.0, 2.0, 0.0 ];
    let a1 : &[f64] = &[ 2.0, 1.0, 3.0, 1.0 ];
    let a2 : &[f64] = &[ 0.0, 2.0, 0.0, 3.0 ];
    let c  : &[f64] = &[ 3.0, 1.0, 5.0, 1.0 ];

    // Create a model with the name 'lo1'
    let mut m = Model::new(Some("lo1"));
    // Create variable 'x' of length 4
    let x = m.variable(Some("x0"), nonnegative().with_shape(&[4]));

    // Create constraints
    let _ = m.constraint(None, x.index(1), less_than(10.0));
    let _ = m.constraint(Some("c1"), x.dot(a0), equal_to(30.0));
    let _ = m.constraint(Some("c2"), x.dot(a1), greater_than(15.0));
    let _ = m.constraint(Some("c3"), x.dot(a2), less_than(25.0));

    // Set the objective function to (c^t * x)
    m.objective(Some("obj"), Sense::Maximize, x.dot(c));

    (m,x)
}
fn main() {
    if let Ok(host) = std::env::var("OPTSERVER_HOST") {
        let (mut m,x) = lo1();
        m.set_parameter("optserver", OptserverHost(host));
        m.solve();

        // Get the solution values
        let (psta,dsta) = m.solution_status(SolutionType::Default);
        println!("Status = {:?}/{:?}",psta,dsta);
        let xx = m.primal_solution(SolutionType::Default,&x);
        println!("x = {:?}", xx);
    }
}

