//! Matrix -- Generic 2D Matrix implementation in Rust.
//! Current Version -- 0.3

#![crate_name = "matrixrs"]
#![crate_type="lib"]
#![allow(unused_must_use)]
#![deny(missing_doc)]
#![feature(macro_rules)]

extern crate debug;
use std::num;
use std::num::{abs,zero,one,Zero,One};
use std::cmp::{PartialOrd};
use std::fmt::{Formatter,Show};
use std::vec::Vec;
use std::iter::Iterator;
use std::iter::FromIterator;
use std::slice::Items;
use std::default::Default;

/// The Matrix struct represent a matrix

#[deriving(Clone)]
pub struct Matrix<T,D> {
	/// Number of rows
	pub nrow: uint,

	/// Number of columns
	pub ncol: uint,

	/// Table (Vector of Vector) of data values in the matrix
	/// its a vec of rows which are vecs of elems
	pub data: Vec<Vec<T>>
}

impl<T:Default> Matrix<T,Dimension> {
	pub fn new(nrow: uint, ncol: uint) -> Matrix<T,Dimension> {
		//! make a new matrix with default value
		Matrix::from_fn(nrow, ncol, |_,_| Default::default())
	}
}

impl<T> Matrix<T,Dimension> {
	pub fn from_fn(nrow: uint, ncol: uint, func: |uint, uint| -> T) -> Matrix<T,Dimension> {
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
			nrow: nrow,
			ncol: ncol,
			data: range(0, nrow).map(|i| Vec::from_fn(ncol, |j| func(i, j))).collect()
		}
	}

	pub fn size(&self) -> (uint, uint) {
		//! Return the size of a Matrix as (row, column).
		(self.nrow, self.ncol)
	}

	pub fn set(&mut self, row: uint, col: uint, val: T) {
		//! set value for the specific element of matrix
		self.data.as_mut_slice()[row - 1].as_mut_slice()[col - 1] = val;
	}
}

impl<T:Clone> Matrix<T> {
	pub fn from_elem(nrow: uint, ncol: uint, val: T) -> Matrix<T> {
		//! Create an m-by-n matrix, where each element is a clone of val.
		//!
		//! ```rust
		//! assert_eq!(
		//! 	Matrix::from_elem(2, 2, 10),
		//!		Matrix {m: 2, n: 2, data: vec![vec![10,10], vec![10,10]]
		//!	});
		//! ```

		Matrix {
			nrow: nrow,
			ncol: ncol,
			data: range(0, nrow).map(|_| Vec::from_elem(ncol, val.clone())).collect()
		}
	}

	pub fn from_vec(nrow: uint, ncol: uint, vals: Vec<T>) -> Matrix<T> {
		//! deprecated use to_matrix in stead.

		assert_eq!(nrow * ncol, vals.len());
		let mut v_iter = vals.iter();
			Matrix {
       		ncol: ncol,
       		nrow: nrow,
       		data: range(0,nrow).map(
       			|_| v_iter.by_ref().take(ncol).map(|x| x.clone()).collect()
       			).collect()
       	}
		
	}

	pub fn at(&self, row: uint, col: uint) -> T {
		//! Return the element at row, col.
		//! Wrapped by Index trait.

		self.data[row][col].clone()
	}

	
	pub fn row_vec(&self, row: uint) -> Vec<T> {
		//! return a vec of a row

		let mut v = Vec::new();
		v.clone_from(&self.data[row]);
		v
	}

	pub fn row_mat(&self, row: uint) -> Matrix<T> {
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
			nrow: 1,
			ncol: self.ncol,
			data: vec![self.row_vec(row)]
		}
	}

	
	pub fn col_vec(&self, col: uint) -> Vec<T> {
		//! return a vec of a column

		range(0, self.nrow).map(|i| self.at(i, col)).collect::<Vec<T>>()
	}

	pub fn col_mat(&self, col: uint) -> Matrix<T> {
		//! Return specified col from an MxN matrix as an Mx1 matrix.
		//!
		//! ```rust
		//! assert_eq!(
		//! 	Matrix{m: 2, n: 2, data: vec![vec![1,3], vec![2,4]]}.col(1),
		//!		Matrix{m: 2, n: 1, data: vec![vec![3], vec![4]]}
		//! );
		//! ```
		let mut cols: Vec<Vec<T>> = Vec::new();
		for v in self.col_vec(col).iter() {
			cols.push(vec![v.clone()]);
		}
		Matrix {
			nrow: self.nrow,
			ncol: 1,
			data: cols
		}
	}

	
	pub fn augment(&self, mat: &Matrix<T>) -> Matrix<T> {
		//! Return a new matrix, self augmented by matrix mat.
		//! An MxN matrix augmented with an MxC matrix produces an Mx(N+C) matrix.

		Matrix::from_fn(self.nrow, self.ncol + mat.ncol, |i,j| {
			if j < self.ncol {
				self.at(i, j) 
			} else {
				mat.at(i, j - self.ncol) 
			}
		})
	}

	pub fn transpose(&self) -> Matrix<T> {
		//! Return the transpose of the matrix.
		//! The transpose of a matrix MxN has dimensions NxM.

		Matrix::from_fn(self.ncol, self.nrow, |i,j| { self.at(j, i) })
	}

	pub fn apply(&self, applier: |uint, uint|) {
		//! Call an applier function with each index in self.
		//! Input to applier is two parameters: row, col.

		for i in range(0, self.nrow) {
			for j in range(0, self.ncol) {
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
				let mut padding = " ".to_string();
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

		Matrix::from_fn(self.nrow, self.ncol, |i,j| { mapper(self.at(i,j)) })
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
		range(0, self.ncol).fold(zero(), |acc: T, i| {acc + self.at(0, i) * other.at(i, 0)})
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
		let mut pivot = Matrix::from_fn(self.nrow, self.ncol, |i, j| {
			if i == j { one() } else { zero() }
		});

		// rearrange pivot matrix so max of each column of self is on
		// the diagonal of self when multiplied by the pivot
		for j in range(0, self.ncol) {
			let mut row_max = j;
			for i in range(j, self.nrow) {
				if abs(self.at(i, j)) > abs(self.at(row_max, j)) {
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

		if self.ncol != self.nrow {
			return Err("col num don't match row num".to_string());
		}
		assert_eq!(self.nrow, self.ncol);
		let p = self.doolittle_pivot();
		let pm = (p*(*self)).to_f64();
		let mut l = eye(self.nrow);
		let mut u: Matrix<f64> = zeros(self.nrow, self.ncol);
		for j in range(0, self.ncol) {
			for i in range(0, j+1) {
				let mut uppersum = 0.0;
				for k in range(0,i) {
					uppersum += u.at(k,j)*l.at(i,k);
				}
				u.data.as_mut_slice()[i].as_mut_slice()[j] = pm.at(i,j) - uppersum;
			}
			for i in range(j, self.nrow) {
				let mut lowersum = 0.0;
				for k in range(0,j) {
					lowersum += u.at(k,j)*l.at(i,k);
				}
				l.data.as_mut_slice()[i].as_mut_slice()[j] = (pm.at(i,j) - lowersum) / u.at(j,j);
			}
		}
		Ok((p, l, u))
	}
	pub fn det(&self) -> Result<f64,String> {
		//! Return the determinant of square matrix self
		//! via LU decomposition.
		//! If not a square matrix, fail.

		if self.ncol != self.nrow {
			return Err("col num don't match row num".to_string());
		}
		match self.lu().unwrap() {
			// |L|=1 because it L is unitriangular
			// |P|=1 or -1 because it's a permutation matrix
			// |U|=product of U's diagonal
			(p, _, u) => {
				// return the product of the diagonal
				let mut prod = 1.0;
				let mut swaps = 0i;
				for i in range(0, self.nrow) {
					prod *= u.at(i, i);
					swaps += if p.at(i, i) == one() { 0 } else { 1 };
				}
				// flip the sign of the determinant based on swaps of P
				if (swaps / 2) % 2 == 1 {
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
		Matrix::from_fn(self.nrow, self.ncol, |i, j| {
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

		assert_eq!(self.ncol, rhs.nrow);
		Matrix::from_fn(self.nrow, rhs.ncol, |i,j| {
			self.row_mat(i).dot(&rhs.col_mat(j))
		})
	}
}

//these will be comments until RFC #48 landed
// impl<T:Add<T,T>+Mul<T,T>+Zero+Clone+ToMatrix> Mul<T, Matrix<T>> for Matrix<T> {
// 	fn mul(&self, rhs: &T) -> Matrix<T> {
// 		let mrhs = rhs.to_matrix(self.nrow, self.ncol);
// 		self * mrhs
// 	}
// }

/// use [(x,y)] to index matrices
impl<T:Clone> Index<(uint, uint), T> for Matrix<T> {
	fn index<'a>(&'a self, &rhs: &(uint, uint)) -> &'a T {
		//! Return the element at the location specified by a (row, column) tuple.

		match rhs {
			(x,y) => &self.data[x][y]
		}
	}
}

/// use ! to transpose matrices
impl<T:Clone> Not<Matrix<T>> for Matrix<T> {
	fn not(&self) -> Matrix<T> {
		//! Return the transpose of self.
		//! if you don't want to use this, use .transpose() in stead.

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

		assert_eq!(self.nrow, self.ncol);
		let mut ret = Matrix::from_fn(self.nrow, self.ncol, |i,j| {
			self.at(i, j)
		});
		for _ in range(1, *rhs) {
			ret = self*ret;
		}
		ret
	}
}

// impl<T:Zero> Zero for Matrix<T> {
// 	fn zero() -> Matrix<T> {

// 	}
// }
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
pub trait ToMatrix<T> {
	/// The convertion function
	fn to_matrix(&mut self, row: uint, col: uint) -> Matrix<T>;
}

impl<T:Clone,U:Iterator<T>> ToMatrix<T> for U {
	fn to_matrix(&mut self, row: uint, col: uint) -> Matrix<T> {
		let v: Vec<T> = self.by_ref().map(|a| a.clone()).collect();
		Matrix::from_vec(row, col, v)
	}
}

// impl<T:Clone> ToMatrix<T> for T {
// 	fn to_matrix(&self, row: uint, col: uint) -> Matrix<T> {
// 		Matrix::from_elem(row, col, self.clone())
// 	}
// }

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
/// The object that has the implemention of Iterator trait
pub struct MatrixMutIter<'a,T> {
	matrix: &'a mut Matrix<T>,
	curr_row: uint,
	curr_col: uint
}

impl<T:Clone> Matrix<T> {
	pub fn mut_iter<'a>(&'a mut self) -> MatrixMutIter<'a,T> {
		//! Return a iterator of the matrix
		MatrixMutIter {
			matrix: self,
			curr_row: 0,
			curr_col: 0
		}
	}
}

/// convert from an iterator
impl<T:Clone> FromIterator<T> for Matrix<T> {

	/// convert to a matrix from iterator where first elem of
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
    	//! 
    	//! ```rust
    	//!	let v = vec![1,2,3,4,5,6];
    	//! let m: Matrix<int> = v.iter().map(|x| x.clone()).to_matrix(2,3);
    	//! println!("{}",m);
    	//!
    	//! Output:
    	//! | 1 2 |
    	//! | 3 4 |
    	//! | 5 6 |
    	//! ```
    	let cp: Vec<T> = iterator.collect();

       	let num_round = (cp.iter().count() as f64).sqrt().floor().powi(2) as uint;

       	let num_row = (num_round as f64).sqrt() as uint;

       	let mut cp_iter = cp.iter();
       	Matrix {
       		ncol: num_row,
       		nrow: num_row,
       		data: range(0,num_row).map(
       			|_| cp_iter.by_ref().take(num_row).map(|x| x.clone()).collect()
       			).collect()
       	}
    }
}

impl<'a,T:Clone> Matrix<T> {
	pub fn row_iter(&'a mut self) -> Items<'a,Vec<T>> {
		//!
		self.data.iter()
	}

	// pub fn cols(self) -> Items<'a,Vec<T>> {
	// 	//!
	// 	let m = self.matrix.transpose();
	// 	m.iter().rows()
	// }
}

impl<'a,T:Clone> Iterator<T> for MatrixIter<'a,T> {
	fn next(&mut self) -> Option<T> {
		match (self.curr_row, self.curr_col) {
			(row, col) if row < self.matrix.nrow && col < self.matrix.ncol => {
				if self.matrix.ncol == col + 1 {
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

impl<'a,T:Clone> Iterator<T> for MatrixMutIter<'a,T> {
	fn next(&mut self) -> Option<T> {
		match (self.curr_row, self.curr_col) {
			(row, col) if row < self.matrix.nrow && col < self.matrix.ncol => {
				if self.matrix.ncol == col + 1 {
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

// -----------------------------------------------------------------

#[macro_export]
macro_rules! matrix (
	($([ $($elem:expr),+ ])+) => ({

		let mut v = Vec::new();
		$(
			let mut vsub = Vec::new();
			$(
				vsub.push($elem);
			)+
			
			v.push(vsub);
		)+
		let len = v[0].len(); 

		for vs in v.iter() {
			assert_eq!(len,vs.len());
		}

		Matrix {
			nrow: v.len(),
			ncol: len,
			data: v
		}
	});
)