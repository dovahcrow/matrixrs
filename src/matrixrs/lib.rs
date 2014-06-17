//! Matrix -- Generic 2D Matrix implementation in Rust.
//! Current Version -- 0.3

#![crate_id = "matrixrs#0.3"]
#![crate_type="lib"]
#![allow(unused_must_use)]
#![deny(missing_doc)]

use std::num;
use std::num::{abs,zero,one,Zero,One};
use std::cmp::{PartialOrd};
use std::fmt::{Formatter,Show};
use std::vec::Vec;
use std::iter::Iterator;
use std::iter::FromIterator;

/// The Matrix struct represent a matrix
pub struct Matrix<T> {
	/// Number of rows
	row: uint,

	/// Number of columns
	col: uint,

	/// Table (Vector of Vector) of data values in the matrix
	/// its a vec of rows which are vecs of elems
	data: Vec<Vec<T>>
}

impl<T> Matrix<T> {
	pub fn from_fn(row : uint, col : uint, func : |uint, uint| -> T) -> Matrix<T> {
		//! Create an m-by-n matrix by using a function func
		//! that returns a number given row and column.
		//!
		//! ```rust
		//! assert_eq!(
		//! 	Matrix::from_fn(2, 2, |i,j| i+j),
		//!     Matrix{m: 2, n: 2, data: vec![vec![0,1], vec![1,2]]}
		//! );
		//! ```
		Matrix {
			row: row,
			col: col,
			data: range(0, row).map(|i| Vec::from_fn(col, |j| func(i, j))).collect()
		}
	}

	pub fn size(&self) -> (uint, uint) {
		//! Return the size of a Matrix as (row, column).
		(self.row, self.col)
	}
}

impl<T:Clone> Matrix<T> {
	pub fn from_elem(row: uint, col: uint, val: T) -> Matrix<T> {
		//! Create an m-by-n matrix, where each element is a clone of val.
		//!
		//! ```rust
		//! assert_eq!(
		//! 	Matrix::from_elem(2, 2, 10),
		//!		Matrix {m: 2, n: 2, data: vec![vec![10,10], vec![10,10]]
		//!	});
		//! ```

		Matrix {
			row: row,
			col: col,
			data: range(0, row).map(|_| Vec::from_elem(col, val.clone())).collect()
		}
	}

	pub fn at(&self, row: uint, col: uint) -> T {
		//! Return the element at row, col.
		//! Wrapped by Index trait.

		self.data.get(row).get(col).clone()
	}

	
	pub fn row_vec(&self, row: uint) -> Vec<T> {
		//! return a vec of a row

		let mut v = Vec::new();
		v.clone_from(self.data.get(row));
		v
	}

	pub fn row(&self, row: uint) -> Matrix<T> {
		//! Return specified row from an MxN matrix as a 1xN matrix.
		//!
		//! ```rust
		//! # use matrixrs::Matrix;
		//! assert_eq!(
		//! 	Matrix{m: 2, n: 1, data: vec![vec![1],vec![2]]}.row(0),
		//!     Matrix{m: 1, n: 1, data: vec![vec![1]]}
		//! );
		//! ```

		Matrix {
			row: 1,
			col: self.col,
			data: vec![self.row_vec(row)]
		}
	}

	
	pub fn col_vec(&self, col: uint) -> Vec<T> {
		//! return a vec of a column

		range(0, self.row).map(|i| self.at(i, col)).collect::<Vec<T>>()
	}

	pub fn col(&self, col: uint) -> Matrix<T> {
		//! Return specified col from an MxN matrix as an Mx1 matrix.
		//!
		//! ```rust
		//! assert_eq!(
		//! 	Matrix{m: 2, n: 2, data: vec![vec![1,3], vec![2,4]]}.col(1),
		//!		Matrix{m: 2, n: 1, data: vec![vec![3], vec![4]]}
		//! );
		//! ```
		Matrix {
			row: self.row,
			col: 1,
			data: vec![self.col_vec(col)]
		}
	}

	
	pub fn augment(&self, mat: &Matrix<T>) -> Matrix<T> {
		//! Return a new matrix, self augmented by matrix mat.
		//! An MxN matrix augmented with an MxC matrix produces an Mx(N+C) matrix.

		Matrix::from_fn(self.row, self.col + mat.col, |i,j| {
			if j < self.col {
				self.at(i, j) 
			} else {
				mat.at(i, j - self.col) 
			}
		})
	}

	pub fn transpose(&self) -> Matrix<T> {
		//! Return the transpose of the matrix.
		//! The transpose of a matrix MxN has dimensions NxM.

		Matrix::from_fn(self.col, self.row, |i,j| { self.at(j, i) })
	}

	pub fn apply(&self, applier: |uint, uint|) {
		//! Call an applier function with each index in self.
		//! Input to applier is two parameters: row, col.

		for i in range(0, self.row) {
			for j in range(0, self.col) {
				applier(i, j);
			}
		}
	}
}

impl<T:Show> Show for Matrix<T> {
	fn fmt(&self, fmt: &mut Formatter) -> std::fmt::Result {
		let max_width =
		self.data.iter().fold(0, |maxlen, rowvec| {
			let rowmax = rowvec.iter().fold(0, |maxlen, elem| {
				let l = format!("{}", elem).len();
				if maxlen > l {
					maxlen
				} else {
					l
				}
			});

			if maxlen > rowmax {
				maxlen
			} else {
				rowmax
			}
		});

		
		write!(fmt, "{}", "\n");
		for i in self.data.iter() {
			write!(fmt, "|");
			for v in i.iter() {
				let slen = format!("{}", v).len();
				let mut padding = " ".to_str();
				for _ in range(0, max_width-slen) {
					padding.push_str(" ");
				}
				write!(fmt, "{}{}", padding, v);
			}
			write!(fmt, " |\n");
		}
		Ok(())
	}
}
	
impl<T:Clone, U> Matrix<T> {
	pub fn map(&self, mapper : |T| -> U) -> Matrix<U> {
		//! Return a copy of self where each value has been
		//! operated upon by mapper.

		Matrix::from_fn(self.row, self.col, |i,j| { mapper(self.at(i,j)) })
	}
}

// methods for Matrix of numbers
impl<T:Add<T,T>+Mul<T,T>+Zero+Clone> Matrix<T> {
	pub fn sum(&self) -> T {
		//! Return the summation of all elements in self.
		//!
		//! ``rust
		//! assert_eq!(
		//!		Matrix{m: 2, n: 2, data: vec![vec![1,3], vec![2,4]]}.sum(),
		//! 	10
		//!	);
		//! ```
		self.iter().fold(zero(), |a:T, b| a + b)
	}

	fn dot(&self, other: &Matrix<T>) -> T {
		//! Return the product of the first row in self with the first row in other.
		range(0, self.col).fold(zero(), |acc: T, i| acc + self.at(0, i) * other.at(i, 0))
	}
}

impl<T:NumCast+Clone> Matrix<T> {
	pub fn to_f64(&self) -> Matrix<f64> {
		//! Return a new Matrix with all of the elements of self cast to f64.

		self.map(|n| -> f64 { num::cast(n).unwrap() })
	}
}

impl<T:NumCast+Clone, U:NumCast+Clone> Matrix<T> {
	pub fn approx_eq(&self, other: &Matrix<U>, threshold: f64) -> bool {
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

impl<T:Num+NumCast+Clone+Signed+PartialOrd> Matrix<T> {
	fn doolittle_pivot(&self) -> Matrix<T> {
		//! Return the pivoting matrix for self (for Doolittle algorithm)
		//! Assume that self is a square matrix.
		
		// initialize with a type T identity matrix
		let mut pivot = Matrix::from_fn(self.row, self.col, |i, j| {
			if i == j { one() } else { zero() }
		});

		// rearrange pivot matrix so max of each column of self is on
		// the diagonal of self when multiplied by the pivot
		for j in range(0,self.col) {
			let mut row_max = j;
			for i in range(j,self.row) {
				if abs(self.at(i,j)) > abs(self.at(row_max, j)) {
					row_max = i;
				}
			}

			// swap the maximum row with the current one
			let tmp = pivot.row_vec(j);
			pivot.data.as_mut_slice()[j] = pivot.row_vec(row_max);
			pivot.data.as_mut_slice()[row_max] = tmp;
		}
		pivot
	}
	pub fn lu(&self) -> Result<(Matrix<T>, Matrix<f64>, Matrix<f64>),String> {
		//! Perform the LU decomposition of square matrix self, and return
		//! the tuple (P,L,U) where P*self = L*U, and L and U are triangular.

		if self.col != self.row {
			return Err("col num don't match row num".to_str());
		}
		assert_eq!(self.row, self.col);
		let p = self.doolittle_pivot();
		let pm = (p*(*self)).to_f64();
		let mut l = eye(self.row);
		let mut u: Matrix<f64> = zeros(self.row, self.col);
		for j in range(0, self.col) {
			for i in range(0, j+1) {
				let mut uppersum = 0.0;
				for k in range(0,i) {
					uppersum += u.at(k,j)*l.at(i,k);
				}
				u.data.as_mut_slice()[i].as_mut_slice()[j] = pm.at(i,j) - uppersum;
			}
			for i in range(j, self.row) {
				let mut lowersum = 0.0;
				for k in range(0,j) {
					lowersum += u.at(k,j)*l.at(i,k);
				}
				l.data.as_mut_slice()[i].as_mut_slice()[j] = (pm.at(i,j) - lowersum) / u.at(j,j);
				// println!("{:?}",l.at(i,j));
			}
		}
		Ok((p, l, u))
	}
	pub fn det(&self) -> Result<f64,String> {
		//! Return the determinant of square matrix self
		//! via LU decomposition.
		//! If not a square matrix, fail.

		if self.col != self.row {
			return Err("col num don't match row num".to_str());
		}
		match self.lu().unwrap() {
			// |L|=1 because it L is unitriangular
			// |P|=1 or -1 because it's a permutation matrix
			// |U|=product of U's diagonal
			(p, _, u) => {
				// return the product of the diagonal
				let mut prod = 1.0;
				let mut swaps = 0;
				for i in range(0, self.row) {
					prod *= u.at(i,i);
					swaps += if p.at(i,i) == one() { 0 } else { 1 };
				}
				// flip the sign of the determinant based on swaps of P
				if (swaps/2) % 2 == 1 {
					Ok(-prod)
				} else {
					Ok(prod)
				}
			}
		}
	}
}



/// use + to add matrices
impl<T:Add<T,T>+Clone> Add<Matrix<T>,Matrix<T>> for Matrix<T> {
	fn add(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the sum of two matrices with the same dimensions.
		//! If sizes don't match, fail.

		assert_eq!(self.size(), rhs.size());
		Matrix::from_fn(self.row, self.col, |i, j| {
			self.at(i,j) + rhs.at(i,j)
		})
	}
}

/// use unary - to negate matrices
impl<T:Neg<T>+Clone> Neg<Matrix<T>> for Matrix<T> {
	fn neg(&self) -> Matrix<T> {
		//! Return a matrix of the negation of each value in self.

		self.map(|n| { -n })
	}
}

/// use binary - to subtract matrices
impl<T:Neg<T>+Add<T,T>+Clone> Sub<Matrix<T>, Matrix<T>> for Matrix<T> {
	fn sub(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the difference of two matrices with the same dimensions.
		//! If sizes don't match, fail.

		self + (-rhs)
	}
}

/// use * to multiply matrices
impl<T:Add<T,T>+Mul<T,T>+Zero+Clone> Mul<Matrix<T>, Matrix<T>> for Matrix<T> {
	fn mul(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return the product of multiplying two matrices.
		//! MxR matrix * RxN matrix = MxN matrix.
		//! If inner dimensions don't match, fail.

		assert_eq!(self.col, rhs.col);
		Matrix::from_fn(self.row, rhs.row, |i,j| {
			self.row(i).dot(&rhs.col(j))
		})
	}
}

//these will be comments until RFC #48 landed
// impl<T:Add<T,T>+Mul<T,T>+Zero+Clone+ToMatrix> Mul<T, Matrix<T>> for Matrix<T> {
// 	fn mul(&self, rhs: &T) -> Matrix<T> {
// 		let mrhs = rhs.to_matrix(self.row, self.col);
// 		self * mrhs
// 	}
// }

/// use [(x,y)] to index matrices
impl<T:Clone> Index<(uint, uint), T> for Matrix<T> {
	fn index(&self, &rhs: &(uint, uint)) -> T {
		//! Return the element at the location specified by a (row, column) tuple.

		match rhs {
			(x,y) => self.at(x,y)
		}
	}
}

/// use ! to transpose matrices
impl<T:Clone> Not<Matrix<T>> for Matrix<T> {
	fn not(&self) -> Matrix<T> {
		//! Return the transpose of self.

		self.transpose()
	}
}

/// use | to augment matrices
impl<T:Clone> BitOr<Matrix<T>,Matrix<T>> for Matrix<T> {
	fn bitor(&self, rhs: &Matrix<T>) -> Matrix<T> {
		//! Return self augmented by matrix rhs.

		self.augment(rhs)
	}
}

/// use ^ to exponentiate matrices
impl<T:Add<T,T>+Mul<T,T>+num::Zero+Clone> BitXor<uint, Matrix<T>> for Matrix<T> {
	fn bitxor(&self, rhs: &uint) -> Matrix<T> {
		//! Return a matrix of self raised to the power of rhs.
		//! Self must be a square matrix.

		assert_eq!(self.row, self.col);
		let mut ret = Matrix::from_fn(self.row, self.col, |i,j| {
			self.at(i, j)
		});
		for _ in range(1, *rhs) {
			ret = self*ret;
		}
		ret
	}
}

// convenience constructors
pub fn zeros<T:Zero+Clone>(row : uint, col : uint) -> Matrix<T> {
	//! Create an MxN zero matrix of type which implements num::Zero trait.

	Matrix::from_elem(row, col, zero())
}

pub fn ones<T:One+Clone>(row : uint, col : uint) -> Matrix<T> {
	//! Create an MxN ones matrix of type which implements num::One trait.

	Matrix::from_elem(row, col, one())
}

pub fn eye<T:Zero+One>(dim : uint) -> Matrix<T> {
	//! Create a dim x dim identity matrix of type which implements num::Zero and num::One trait.

	Matrix::from_fn(dim, dim, |i, j| { if i == j { one() } else { zero() }})
}

//-----------------------------------------------------------------

/// A trait that convert T to Matrix<T>
pub trait ToMatrix {
	/// The convertion function
	fn to_matrix(&self, row: uint, col: uint) -> Matrix<Self>;
}

impl<T:Clone> ToMatrix for T {
	fn to_matrix(&self, row: uint, col: uint) -> Matrix<T> {
		Matrix::from_elem(row, col, self.clone())
	}
}

//--------------------------------------------------------------------

/// The object that has the implemention of Iterator trait
pub struct MatrixIter<'a,T> {
	matrix: &'a Matrix<T>,
	curr_row: uint,
	curr_col: uint
}

impl<T:Clone> Matrix<T> {
	pub fn iter<'a>(&'a self) -> MatrixIter<'a,T> {
		//! Return a iterator of the matrix
		MatrixIter {
			matrix: self,
			curr_row: 0,
			curr_col: 0
		}
	}
}

/// convert from an iterator
impl<T:Clone> FromIterator<T> for Matrix<T> {

	/// convert to a square matrix that some elements might be truncated 
    fn from_iter<I: Iterator<T>>(mut iterator: I) -> Matrix<T> {
    	//!
    	//! ```
    	//!	let v = vec![1,2,3,4,5];
    	//! let m: Matrix<int> = v.iter().map(|x| x.clone()).collect();
    	//! println!("{}",m);
    	//!
    	//! Output:
    	//! 
    	//! | 1 2 |
    	//! | 3 4 |
    	//! ```

    	let cp: Vec<T> = iterator.collect();

       	let num_round = (cp.iter().count() as f64).sqrt().floor().powi(2) as uint;

       	let num_row = (num_round as f64).sqrt() as uint;

       	let mut cp_iter = cp.iter();
       	Matrix {
       		col: num_row,
       		row: num_row,
       		data: range(0,num_row).map(
       			|_| cp_iter.by_ref().take(num_row).map(|x| x.clone()).collect()
       			).collect()
       	}
    }
}

impl<'a,T:Clone> Iterator<T> for MatrixIter<'a,T> {
	fn next(&mut self) -> Option<T> {
		match (self.curr_row, self.curr_col) {
			(row, col) if row < self.matrix.row && col < self.matrix.col => {
				if self.matrix.col == col + 1 {
					self.curr_row += 1;
					self.curr_col = 0;	
				} else {
					self.curr_col +=1;
				}
				Some(self.matrix.at(row, col))
			}
			_ => None
		}
		
	}
}
