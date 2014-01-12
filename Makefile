build:
	rustpkg build matrixrs

test:
	rustpkg test matrixrs

clean:
	rustpkg clean matrixrs

doc:
	rustdoc src/matrixrs/lib.rs
