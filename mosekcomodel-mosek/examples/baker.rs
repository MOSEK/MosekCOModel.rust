//!
//! Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//!
//! File:      baker.rs
//!
//! Purpose: Demonstrates a small linear problem.
//!
//! Source: "Linaer Algebra" by Knut Sydsaeter and Bernt Oeksendal.
//!
//! The problem: A baker has 150 kg flour, 22 kg sugar, 25 kg butter and two
//! recipes:
//! 1) Cakes, requiring 3.0 kg flour, 1.0 kg sugar and 1.2 kg butter per dozen.
//! 2) Breads, requiring 5.0 kg flour, 0.5 kg sugar and 0.5 kg butter per dozen.
//! Let the revenue per dozen cakes be $4 and the revenue per dozen breads be $6.
//!
//! We now wish to compute the combination of cakes and breads that will optimize
//! the total revenue.

extern crate mosekcomodel;
use mosekcomodel::*;
use mosekcomodel_mosek::Model;

fn main() {
    let _ingredientnames = [ "Flour", "Sugar", "Butter" ];
    let stock : &[f64] = &[ 150.0,   22.0,    25.0 ];

    let recipe_data : &[f64] = &[3.0, 5.0, 
                                 1.0, 0.5,
                                 1.2, 0.5 ];
    let product_names = [ "Cakes", "Breads" ];

    let revenue : &[f64] = &[ 4.0, 6.0 ];

    let mut model = Model::new(Some("Baker"));
    let recipe = matrix::dense([3,2],recipe_data);
    // "production" defines the amount of each product to bake.
    let production = model.variable(Some("production"),
                                    nonnegative().with_shape(&[2]));
    greater_than(0.0).with_shape(&[2]);
    // The objective is to maximize the total revenue.
    model.objective(Some("revenue"),
                    Sense::Maximize,
                    production.dot(revenue));

    // The prodoction is constrained by stock:
    model.constraint(None, recipe.mul(&production), less_than(stock));
    model.set_log_handler(|msg| print!("{}",msg));

    // We solve and fetch the solution:
    model.solve();
    let res = model.primal_solution(SolutionType::Default, &production).unwrap();
    println!("Solution:");
    for (n,r) in product_names.iter().zip(res.iter()) {
        println!(" Number of {} : {}",n,r);
    }
    let total_revenue : f64 = res.iter().zip(revenue.iter()).map(|(&a,&b)| a*b).sum();
    println!(" Revenue : ${}", total_revenue);
}

#[test]
fn test() { main() }
