extern mod matrixrs;

use matrixrs::Matrix;

#[test]
fn test_eq() {
	// test equality
	let m1 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	let m2 = Matrix{m:2, n:2, data: ~[~[1,1],~[2,2]]};
	assert!(m1 == m2);
	// test inequality
	let m3 = Matrix{m:3, n:1, data: ~[~[2],~[3],~[1]]};
	let m4 = Matrix{m:3, n:1, data: ~[~[1],~[2],~[3]]};
	assert!(m3 != m4);
	// make sure it doesn't break if dimensions don't match
	let m5 = Matrix{m:2, n:1, data: ~[~[2],~[3]]};
	let m6 = Matrix{m:1, n:2, data: ~[~[2,3]]};
	assert!(m5 != m6)
}
