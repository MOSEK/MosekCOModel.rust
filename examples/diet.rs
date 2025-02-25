// Copyright: Copyright (c) MOSEK ApS, Denmark. All rights reserved.
//
// File:      diet.rs
//
// Purpose: Solving Stigler's Nutrition model (DIET,SEQ=7)
//
// Source: GAMS Model library,
//           Dantzig, G B, Chapter 27.1. In Linear Programming and Extensions.
//           Princeton University Press, Princeton, New Jersey, 1963.
//
// Given a set of nutrients, foods with associated nutrient values, allowance of
// nutrients per day, the model will find the cheapest combination of foods
// which will provide the necessary nutrients.
//
//  Arguments for construction:
//  name      - Model name.
//  foods     - List of M names of the foods.
//  nutrients - List of N names of the nutrients.
//  daily_allowance - List of N floats denoting the daily allowance of each
//              nutrient.
//  nutritive_value - Two-dimensional MxN array of floats where each row
//              denotes the nutrient values for a single food per $ spent.
extern crate mosekcomodel;
extern crate itertools;

use mosekcomodel::*;
use itertools::izip;

fn diet(daily_allowance : &[f64],
        nutritive_value : matrix::NDArray<2>) -> Result<(Vec<f64>,Vec<f64>),String> {
    let m = nutritive_value.shape()[0];
    let n = nutritive_value.shape()[1];

    if daily_allowance.len() != n {
        return Err("Length of daily_allowance does not match the number of nutrients".to_string());
    }

    let mut model = Model::new(Some("Stingler's Diet Model"));

    let daily_purchase = model.variable(Some("Daily Purchase"),
                                        greater_than(0.0).with_shape(&[m]));

    let daily_nutrients = model.constraint(Some("Nutrient Balance"),
                                           &daily_purchase.clone().mul(nutritive_value),
                                           greater_than(daily_allowance.to_vec()));
    model.objective(None, Sense::Minimize, &daily_purchase.clone().sum());

    model.solve();

    let res_daily_purchase = model.primal_solution(SolutionType::Default, &daily_purchase).unwrap();
    let res_daily_nutrients = model.primal_solution(SolutionType::Default, &daily_nutrients).unwrap();

    Ok((res_daily_purchase,res_daily_nutrients))
}
fn main() {
  /* Main class with data definitions */
  let nutrient_unit = [
      "thousands",  "grams",        "grams",
      "milligrams", "thousand ius", "milligrams",
      "milligrams", "milligrams",   "milligrams"
    ];
    let nutrients = [
      "calorie",    "protein",      "calcium",
      "iron",       "vitamin a",    "vitamin b1",
      "vitamin b2", "niacin",       "vitamin c"
    ];
    let foods = [
      "wheat",         "cornmeal", "cannedmilk", "margarine", "cheese",
      "peanut butter", "lard",     "liver",      "porkroast", "salmon",
      "greenbeans",    "cabbage",  "onions",     "potatoes",  "spinach",
      "sweet potatos", "peaches",  "prunes",     "limabeans", "navybeans"
    ];
    // Nutritive value per $
    let nutritive_value = [
      //  calorie       calcium      vitamin a        vitamin b2      vitamin c
      //         protein        iron           vitamin b1      niacin
      [44.7,  1411.0,   2.0,   365.0,    0.0,    55.4,   33.3, 441.0,     0.0],  // wheat
      [36.0,   897.0,   1.7,    99.0,   30.9,    17.4,   7.9,  106.0,     0.0],  // cornmeal
      [ 8.4,   422.0,  15.1,     9.0,   26.0,     3.0,  23.5,   11.0,    60.0],  // cannedmilk
      [20.6,    17.0,   0.6,     6.0,   55.8,    0.2,    0.0,    0.0,     0.0],  // margarine
      [ 7.4,   448.0,  16.4,    19.0,   28.1,    0.8,   10.3,    4.0,     0.0],  // cheese
      [15.7,   661.0,   1.0,      48.0,  0.0,    9.6,    8.1,  471.0,     0.0],  // peanut butter
      [41.7,     0.0,   0.0,       0.0,  0.2,    0.0,    0.5,    5.0,     0.0],  // lard
      [ 2.2,   333.0,   0.2,   139.0,  169.2,    6.4,   50.8,  316.0,   525.0],  // liver
      [ 4.4,   249.0,   0.3,    37.0,    0.0,   18.2,    3.6,   79.0,     0.0],  // porkroast
      [ 5.8,   705.0,   6.8,    45.0,    3.5,    1.0,    4.9,  209.0,     0.0],  // salmon
      [ 2.4,   138.0,   3.7,    80.0,   69.0,    4.3,    5.8,   37.0,   862.0],  // greenbeans
      [ 2.6,   125.0,   4.0,      36.0,    7.2,  9.0,    4.5,   26.0,  5369.0],  // cabbage
      [ 5.8,   166.0,   3.8,    59.0,   16.6,    4.7,    5.9,   21.0,  1184.0],  // onions
      [14.3,   336.0,   1.8,   118.0,    6.7,   29.4,    7.1,  198.0,  2522.0],  // potatoes
      [ 1.1,   106.0,   0.0,   138.0,  918.4,    5.7,   13.8,   33.0,  2755.0],  // spinach
      [ 9.6,   138.0,   2.7,    54.0,  290.7,    8.4,    5.4,   83.0,  1912.0],  // sweet potatos
      [ 8.5,    87.0,   1.7,   173.0,   86.8,    1.2,    4.3,   55.0,    57.0],  // peaches
      [12.8,    99.0,   2.5,   154.0,   85.7,    3.9,    4.3,   65.0,   257.0],  // prunes
      [17.4,  1055.0,   3.7,   459.0,    5.1,   26.9,   38.2,   93.0,     0.0],  // limabeans
      [26.9,  1691.0,  11.4,   792.0,    0.0,   38.4,   24.6,  217.0,     0.0]   // navybeans
    ];

    let daily_allowance =
        [   3.0,     70.0,  0.8,    12.0,   5.0,      1.8,    2.7,   18.0,   75.0 ];
    let (res_purchase, res_nutrients) = diet(&daily_allowance, 
                                             NDArray::from(&nutritive_value)).unwrap();

    println!("Solution:");
    for (p,f) in res_purchase.iter().zip(foods.iter()).filter(|(p,_)| **p > 0.0) {
        println!("  {:15} : ${:.2}", *f,*p);
    }

    println!("Nutrients:");
    for (n,nut,nut_unit,all) in izip!(nutrients.iter(),res_nutrients.iter(),nutrient_unit.iter(),daily_allowance.iter()) {
        println!("  {:15} : {:>7.2} {} ({:.2})",n,nut,nut_unit,all);
    }


    let mut res = vec![0.0; daily_allowance.len()];
    res_purchase.iter().zip(nutritive_value.iter())
        .for_each(|(&c,nv)| nv.iter().zip(res.iter_mut()).for_each(|(&nvi,r)| *r += c * nvi));
}

#[test]
fn test() { main() }
