#[crate_id = "matrixrs#0.1"];
#[crate_type="lib"];

use std::vec;
use std::num::Zero;

// Matrix
// ------
// Generic slow 2D Matrix implementation in Rust.
pub struct Matrix<T> {
	// number of rows and columns
	m : uint,
	n : uint,
	// table of data values in the matrix
	data : ~[~[T]]
}


impl<T> Matrix<T> {
	pub fn from_fn(m : uint, n : uint, func : |uint, uint| -> T) -> Matrix<T> {
		// Create an MxN matrix by using a function that returns a number given
		// row and column.
		let mut i = 0;
		let mut data = vec::with_capacity(m);
		while i < m {
			data.push(vec::from_fn(n, |j:uint| -> T { func(i, j) }));
			i += 1;
		}
		Matrix{m:m, n:n, data:data}
	}
	pub fn size(&self) -> (uint, uint) {
		// Return the size of a Matrix as row, column.
		((*self).m, (*self).n)
	}
}

impl<T:Clone> Matrix<T> {
	pub fn from_T(m : uint, n : uint, val : T) -> Matrix<T> {
		// Create an MxN matrix of val numbers.
		let mut i = 0;
		let mut data = vec::with_capacity(m);
		while i < m {
			data.push(vec::from_elem(n, val.clone()));
			i += 1;
		}
		Matrix{m:m, n:n, data:data}
	}
	fn at(&self, row : uint, col : uint) -> T {
		// Return the element at row, col.
		(*self).data[row][col].clone()
	}
	pub fn row(&self, row : uint) -> Matrix<T> {
		// Return row r from an MxN matrix as a 1xN matrix.
		Matrix{m: 1, n:(*self).n, data: ~[(*self).data[row].to_owned()]}
	}
	pub fn col(&self, col : uint) -> Matrix<T> {
		// Return col c from an MxN matrix as an Mx1 matrix.
		let mut c = vec::with_capacity((*self).m);
		let mut i = 0;
		while i < (*self).m {
			c.push(~[self.at(i, col)]);
			i += 1;
		}
		Matrix{m: (*self).m, n: 1, data: c}
	}
	pub fn augment(&self, mat : &Matrix<T>) -> Matrix<T> {
		// Augment the self matrix MxN with another matrix MxC
		Matrix::from_fn((*self).m, (*self).n+mat.n, |i,j| {
			if j < (*self).n { self.at(i, j) } else { mat.at(i, j - (*self).n) }
		})
	}
	pub fn transpose(&self) -> Matrix<T> {
		// Return the transpose of the matrix.
		Matrix::from_fn((*self).n, (*self).m, |i,j| { self.at(j, i) })
	}
	pub fn apply(&self, applier : |uint, uint|) {
		let mut i = 0;
		while i < (*self).m {
			let mut j = 0;
			while j < (*self).n {
				applier(i, j);
				j += 1;
			}
			i += 1;
		}
	}
	pub fn map(&self, mapper : |T| -> T) -> Matrix<T> {
		Matrix::from_fn((*self).m, (*self).n, |i,j| { mapper(self.at(i,j)) })
	}
}

// methods for Matrix of numbers
impl<T:Num+Clone> Matrix<T> {
	pub fn sum(&self) -> T {
		let mut acc : T = Zero::zero();
		self.apply(|i,j| { acc = acc+self.at(i,j) });
		acc
	}
	// Perform the LU decomposition of the matrix.
	// Find the determinant of the matrix.
	// Multiply another matrix by this one.
}

impl<T:Eq+Clone> Eq for Matrix<T> {
	fn eq(&self, _rhs: &Matrix<T>) -> bool {
		if self.size() == _rhs.size() {
			let mut equal = true;
			self.apply(|i,j| {
				equal = if self.at(i,j) == _rhs.at(i,j) { equal } else { false };
			});
			equal
		}
		else {
			false
		}
	}
}

// use + to add matrices
impl<T:Num+Clone> Add<Matrix<T>,Matrix<T>> for Matrix<T> {
	fn add(&self, _rhs: &Matrix<T>) -> Matrix<T> {
		assert!(self.size() == _rhs.size());
		Matrix::from_fn((*self).m, (*self).n, |i, j| {
			self.at(i,j) + _rhs.at(i,j)
		})
	}
}

// use unary - to negate matrices
impl<T:Num+Clone> Neg<Matrix<T>> for Matrix<T> {
	fn neg(&self) -> Matrix<T> {
		self.map(|n| { -n })
	}
}

// use binary - to subtract matrices
impl<T:Num+Clone> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
	fn sub(&self, _rhs: &Matrix<T>) -> Matrix<T> {
		self + (-_rhs)
	}
}

// use [(x,y)] to index matrices
impl<T:Clone> Index<(uint, uint), T> for Matrix<T> {
	fn index(&self, &_rhs: &(uint, uint)) -> T {
		match _rhs {
			(x,y) => self.at(x,y)
		}
	}
}

// use ! to transpose matrices
impl<T:Clone> Not<Matrix<T>> for Matrix<T> {
	fn not(&self) -> Matrix<T> {
		self.transpose()
	}
}

// use | to augment matrices
impl<T:Clone> BitOr<Matrix<T>,Matrix<T>> for Matrix<T> {
	fn bitor(&self, _rhs: &Matrix<T>) -> Matrix<T> {
		self.augment(_rhs)
	}
}

// convenience constructors
pub fn zeros(m : uint, n : uint) -> Matrix<f64> {
	// Create an MxN zero matrix of type f64.
	Matrix::from_T(m, n, 0.0)
}

pub fn ones(m : uint, n : uint) -> Matrix<f64> {
	// Create an MxN ones matrix of type f64.
	Matrix::from_T(m, n, 1.0)
}

pub fn identity(dim : uint) -> Matrix<f64> {
	// Create a dimxdim identity matrix of type f64.
	Matrix::from_fn(dim, dim, |i, j| { if i == j { 1.0 } else { 0.0 }})
}
