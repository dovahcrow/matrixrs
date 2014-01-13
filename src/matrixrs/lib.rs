#[crate_id = "matrixrs#0.1"];
#[crate_type="lib"];

use std::vec;
use std::num;
use std::num::Zero;
use std::num::One;
use std::num::abs;

/// Matrix
/// ------
/// Generic 2D Matrix implementation in Rust.
pub struct Matrix<T> {
	/// Number of rows
	m : uint,
	/// Number of columns
	n : uint,
	/// Table (vector of vector) of data values in the matrix
	data : ~[~[T]]
}


impl<T> Matrix<T> {
	pub fn from_fn(m : uint, n : uint, func : |uint, uint| -> T) -> Matrix<T> {
		//! Create an m-by-n matrix by using a function func
		//! that returns a number given row and column.
		let mut data = vec::with_capacity(m);
		for i in range(0, m) {
			data.push(vec::from_fn(n, |j:uint| -> T { func(i, j) }));
		}
		Matrix{m:m, n:n, data:data}
	}
	pub fn size(&self) -> (uint, uint) {
		//! Return the size of a Matrix as row, column.
		(self.m, self.n)
	}
}

impl<T:Clone> Matrix<T> {
	pub fn from_T(m : uint, n : uint, val : T) -> Matrix<T> {
		//! Create an m-by-n matrix, where each element is a clone of val.
		let mut data = vec::with_capacity(m);
		for _ in range(0, m) {
			data.push(vec::from_elem(n, val.clone()));
		}
		Matrix{m:m, n:n, data:data}
	}
	fn at(&self, row : uint, col : uint) -> T {
		//! Return the element at row, col.
		//! Wrapped by Index trait.
		self.data[row][col].clone()
	}
	pub fn row(&self, row : uint) -> Matrix<T> {
		//! Return specified row from an MxN matrix as a 1xN matrix.
		Matrix{m: 1, n:self.n, data: ~[self.data[row].to_owned()]}
	}
	pub fn col(&self, col : uint) -> Matrix<T> {
		//! Return specified col from an MxN matrix as an Mx1 matrix.
		let mut c = vec::with_capacity(self.m);
		for i in range(0, self.m) {
			c.push(~[self.at(i, col)]);
		}
		Matrix{m: self.m, n: 1, data: c}
	}
	pub fn augment(&self, mat : &Matrix<T>) -> Matrix<T> {
		//! Return a new matrix, self augmented by matrix mat.
		//! An MxN matrix augmented with an MxC matrix produces an Mx(N+C) matrix.
		Matrix::from_fn(self.m, self.n+mat.n, |i,j| {
			if j < self.n { self.at(i, j) } else { mat.at(i, j - self.n) }
		})
	}
	pub fn transpose(&self) -> Matrix<T> {
		//! Return the transpose of the matrix.
		//! The transpose of a matrix MxN has dimensions NxM.
		Matrix::from_fn(self.n, self.m, |i,j| { self.at(j, i) })
	}
	pub fn apply(&self, applier : |uint, uint|) {
		//! Call an applier function with each index in self.
		//! Input to applier is two parameters: row, col.
		for i in range(0, self.m) {
			for j in range(0, self.n) {
				applier(i, j);
			}
		}
	}
}
impl<T:Clone, U> Matrix<T> {
	pub fn map(&self, mapper : |T| -> U) -> Matrix<U> {
		//! Return a copy of self where each value has been
		//! operated upon by mapper.
		Matrix::from_fn(self.m, self.n, |i,j| { mapper(self.at(i,j)) })
	}
}

// methods for Matrix of numbers
impl<T:Num+Clone> Matrix<T> {
	pub fn sum(&self) -> T {
		//! Return the summation of all elements in self.
		let mut acc : T = Zero::zero();
		self.apply(|i,j| { acc = acc+self.at(i,j) });
		acc
	}
	fn dot(&self, other: &Matrix<T>) -> T {
		//! Return the product of the first row in self with the first row in other.
		let mut sum : T = Zero::zero();
		for i in range(0, self.n) {
			sum = sum + self.at(0, i) * other.at(i, 0);
		}
		sum
	}
}

impl<T:NumCast+Clone> Matrix<T> {
	pub fn to_f64(&self) -> Matrix<f64> {
		//! Return a new Matrix with all of the elements of self cast to f64.
		self.map(|n| -> f64 { num::cast(n).unwrap() })
	}
}

impl<T:NumCast+Clone, U:NumCast+Clone> Matrix<T> {
	pub fn approx_eq(&self, other: &Matrix<U>, threshold : f64) -> bool {
		//! Return whether all of the elements of self are within
		//! threshold of all of the corresponding elements of other.
		let other_f64 = other.to_f64();
		let self_f64 = self.to_f64();
		let mut equal = true;
		self.apply(|i,j| {
			equal = if abs(self_f64.at(i,j) - other_f64.at(i,j)) <= threshold {
				equal
			} else {
				false
			};
		});
		equal
	}
}

impl<T:Num+NumCast+Clone+Signed+Orderable> Matrix<T> {
	fn doolittle_pivot(&self) -> Matrix<T> {
		//! Return the pivoting matrix for self (for Doolittle algorithm)
		//! Assume that self is a square matrix.
		// initialize with a type T identity matrix
		let mut pivot = Matrix::from_fn(self.m, self.n, |i, j| {
			if i == j { One::one() } else { Zero::zero() }
		});
		// rearrange pivot matrix so max of each column of self is on
		// the diagonal of self when multiplied by the pivot
		for j in range(0,self.n) {
			let mut row_max = j;
			for i in range(j,self.m) {
				if abs(self.at(i,j)) > abs(self.at(row_max, j)) {
					row_max = i;
				}
			}
			// swap the maximum row with the current one
			let tmp = pivot.data[j].to_owned();
			pivot.data[j] = pivot.data[row_max].to_owned();
			pivot.data[row_max] = tmp;
		}
		pivot
	}
	pub fn plu_decomp(&self) -> (Matrix<T>, Matrix<f64>, Matrix<f64>) {
		//! Perform the LU decomposition of square matrix self, and return
		//! the tuple (P,L,U) where P*self = L*U.
		assert!(self.m == self.n);
		let P = self.doolittle_pivot();
		let PM = (P*(*self)).to_f64();
		let mut L = zeros(self.m, self.n);
		let mut U = zeros(self.m, self.n);
		for j in range(0, self.n) {
			L.data[j][j] = 1.0;
			for i in range(0, j+1) {
				let mut uppersum = 0.0;
				for k in range(0,i) {
					uppersum += U.at(k,j)*L.at(i,k);
				}
				U.data[i][j] = PM.at(i,j) - uppersum;
			}
			for i in range(j, self.m) {
				let mut lowersum = 0.0;
				for k in range(0,j) {
					lowersum += U.at(k,j)*L.at(i,k);
				}
				L.data[i][j] = (PM.at(i,j) - lowersum) / U.at(j,j);
			}
		}
		(P, L, U)
	}
}

impl<T:Clone+NumCast> ToStr for Matrix<T> {
	fn to_str(&self) -> ~str {
		//! Return a string representation of Matrix self.
		let self_to_f64 = self.to_f64();
		let mut repr = ~"";
		for i in range(0, self.m) {
			for j in range(0, self.n) {
				repr.push_str(format!("{:>.6f}  ", self_to_f64.at(i,j)));
			}
			repr.push_char(if (i + 1) == self.n { ' ' } else { '\n' });
		}
		repr
	}
}

impl<T:Eq+Clone> Eq for Matrix<T> {
	fn eq(&self, rhs: &Matrix<T>) -> bool {
		//! Return whether the elements of self equal the elements of rhs.
		if self.size() == rhs.size() {
			let mut equal = true;
			self.apply(|i,j| {
				equal = if self.at(i,j) == rhs.at(i,j) { equal } else { false };
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
	fn add(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the sum of two matrices with the same dimensions.
		//! If sizes don't match, fail.
		assert!(self.size() == rhs.size());
		Matrix::from_fn(self.m, self.n, |i, j| {
			self.at(i,j) + rhs.at(i,j)
		})
	}
}

// use unary - to negate matrices
impl<T:Num+Clone> Neg<Matrix<T>> for Matrix<T> {
	fn neg(&self) -> Matrix<T> {
		//! Return a matrix of the negation of each value in self.
		self.map(|n| { -n })
	}
}

// use binary - to subtract matrices
impl<T:Num+Clone> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
	fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the difference of two matrices with the same dimensions.
		//! If sizes don't match, fail.
		self + (-rhs)
	}
}

// use * to multiply matrices
impl<T:Num+Clone> Mul<Matrix<T>, Matrix<T>> for Matrix<T> {
	fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the product of multiplying two matrices.
		//! MxR matrix * RxN matrix = MxN matrix.
		//! If inner dimensions don't match, fail.
		assert!(self.n == (*rhs).m);
		Matrix::from_fn(self.m, (*rhs).n, |i,j| {
			self.row(i).dot(&rhs.col(j))
		})
	}
}

// use [(x,y)] to index matrices
impl<T:Clone> Index<(uint, uint), T> for Matrix<T> {
	fn index(&self, &rhs: &(uint, uint)) -> T {
		//! Return the element at the location specified by a (row, column) tuple.
		match rhs {
			(x,y) => self.at(x,y)
		}
	}
}

// use ! to transpose matrices
impl<T:Clone> Not<Matrix<T>> for Matrix<T> {
	fn not(&self) -> Matrix<T> {
		//! Return the transpose of self.
		self.transpose()
	}
}

// use | to augment matrices
impl<T:Clone> BitOr<Matrix<T>,Matrix<T>> for Matrix<T> {
	fn bitor(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return self augmented by matrix rhs.
		self.augment(rhs)
	}
}

// convenience constructors
pub fn zeros(m : uint, n : uint) -> Matrix<f64> {
	//! Create an MxN zero matrix of type f64.
	Matrix::from_T(m, n, 0.0)
}

pub fn ones(m : uint, n : uint) -> Matrix<f64> {
	//! Create an MxN ones matrix of type f64.
	Matrix::from_T(m, n, 1.0)
}

pub fn identity(dim : uint) -> Matrix<f64> {
	//! Create a dimxdim identity matrix of type f64.
	Matrix::from_fn(dim, dim, |i, j| { if i == j { 1.0 } else { 0.0 }})
}
