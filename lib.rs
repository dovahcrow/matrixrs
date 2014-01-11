#[crate_id = "matrixrs#0.1"];

use std::vec;

// Matrix
// ------
// Generic slow 2D Matrix implementation in Rust for numbers.
struct Matrix<Num> {
	// number of rows and columns
	m : uint,
	n : uint,
	// table of data values in the matrix
	data : ~[~[Num]]
}

// all Nums in std:: already implement Clone
impl<Num: Clone> Matrix<Num> {
	fn from_fn(m : uint, n : uint, func : |uint, uint| -> Num) -> ~Matrix<Num> {
		// Create an MxN matrix by using a function that returns a number given
		// row and column.
		let mut i = 0;
		let mut data = vec::with_capacity(m);
		while i < m {
			data.push(vec::from_fn(n, |j:uint| -> Num { func(i, j) }));
			i += 1;
		}
		~Matrix{m:m, n:n, data:data}
	}

	fn from_Num(m : uint, n : uint, val : Num) -> ~Matrix<Num> {
		// Create an MxN matrix of val numbers.
		let mut i = 0;
		let mut data = vec::with_capacity(m);
		while i < m {
			data.push(vec::from_elem(n, val.clone()));
			i += 1;
		}
		~Matrix{m:m, n:n, data:data}
	}

	fn size(&self) -> (uint, uint) {
		// Return the size of a Matrix as row, column.
		((*self).m, (*self).n)
	}
	fn at(&self, row : uint, col : uint) -> Num {
		// Return the element at row, col.
		(*self).data[row][col].clone()
	}
	fn row(&self, row : uint) -> ~Matrix<Num> {
		// Return row r from an MxN matrix as a 1xN matrix.
		~Matrix{m: 1, n:(*self).n, data: ~[(*self).data[row].to_owned()]}
	}
	// Return col c from an MxN matrix as an Nx1 matrix.
	// TODO
}

fn zeros(m : uint, n : uint) -> ~Matrix<f64> {
	// Create an MxN zero matrix of type f64.
	Matrix::from_Num(m, n, 0.0)
}

fn ones(m : uint, n : uint) -> ~Matrix<f64> {
	// Create an MxN ones matrix of type f64.
	Matrix::from_Num(m, n, 1.0)
}

fn identity(dim : uint) -> ~Matrix<f64> {
	// Create a dimxdim identity matrix of type f64.
	Matrix::from_fn(dim, dim, |i, j| { if i == j { 1.0 } else { 0.0 }})
}

fn main() {
	let m1 = zeros(3,3);
	let m2 = ones(3,3);
	let m3 = identity(3);
	println!("{:?}\n{:?}\n{:?}", m1, m2, m3);
	println!("{:?}\n{:?}\n{:?}", m3.size(), m3.at(1,1), m3.row(2));
}
