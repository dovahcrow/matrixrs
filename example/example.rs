extern crate matrixrs;

use matrixrs::{ToMatrix};

fn main() {
 
	let matrix = 
	[
	1i,3,4,4,
	6,3,7,9,
	5,7,0,8
	].iter().map(|x| x.clone()).to_matrix(3,4);
	println!("{}",matrix);

	let (row,col) = matrix.size();
	println!("the size of matrix is: {} rows,{} columns",row,col);

	let agmt = [1i,9,7,6].iter().map(|x| x.clone()).to_matrix(1,4);

	let matrixB = matrix.transpose().augment(&agmt.transpose()).transpose();
	println!("matrixB is: {}",matrixB);

	let matrix_add = matrixB.map(|a| a + 1);
	println!("matrix_add is: {}",matrix_add);

	let result = matrix_add.lu();
	let (a,b,c) = result.unwrap();
	println!("{},{},{}",a,b,c);

	let det = matrixB.det();
	println!("matrixB's det is: \n{}",det.unwrap());

	let matrixC = matrixB + matrix_add;
	println!("matrixC is: {}",matrixC);

	let matrix_transposed = !matrixC;
	println!("matrix_transposed is: {}",matrix_transposed);

	let matrix_mul = matrix_transposed * matrixC;
	println!("matrix_mul is: {}",matrix_mul);
}