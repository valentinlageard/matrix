# Notes

## Development notes

- [x] Choose the internal representation
    - While Vec uses heap allocation, it seems the right choice because arrays can't be sized at run time
    - Matrix should be implemented using a contiguous one-dimensional layout. 2d indexing should be implemented on top of this representation.
- [x] How to use FMA ?
    - It seems that `.mul_add()` already uses fma but we need to confirm it by checking the assembly code generated. This would be the best choice because the code would be more portable (the compiler chooses the operation given the target architecture). The problem being that it's unclear if it vectorizes the result in simd.
    - We can directly uses x86 fma instructions using the `core::arch` module if the first solution fails.
    - It seems the solution is : use `std::simd` to easily get simd lanes, then use `core::arch` to be able to use fma since `std::simd` doesn't seem to implement fma instructions. We need to check if these two libs work well together.
    - `std::simd` and `std::arch::x86` work well together. We can split arrays into an array of f32x4 with `as_simd` or `as_simd_mut` and then we can convert them to `__m128` easily. We then can use the `_mm_fmadd_ps` function on those. Now we just need to implement it !
- [x] Should we implement so that we can use the standard numeric operators +, -. ... ?
    - Actually, the "operator overloading" traits can define an arbitrary output type. This means that we could in theory return a Result for instance. This would be impractical because it would mean that we would need to unpack the result when chaining linear algebra operations.
    - Another solution is to panic if the user tries to make an operation that fails (e.g. shape mismatch). This way, we can get rid of the error and always return a valid output or dramatically fail. This is consistent with the subject saying that if vectors don't have the same dimension, the result is undefined.
- [x] Should we make available both in-place and pure versions of the methods ?
    - I think yes because in place is less practical but can be a bit more efficient because we don't need to allocate a new internal vec. In place methods can be either consuming the vector or mutably borrowing it. What is best ?
    - Pure implementations can however be chained more easily.
    - Only PURE implementations !
- [x] Ideally we would need a benchmark environment to check empirically for the complexity requirements.
    - We could make a python script that executes various versions of the programs, collect the data and print nice graphs with matplotlib.
    - Or we could set up the criterion package !

## TODO

Generalities:
- [x] Set up a benchmark environment to test complexity and optimizations.
- [x] Organiser le code en au moins 2 modules: Vector et Matrix
- [ ] Séparer les unit tests dans des fichiers spéciaux
- [ ] Find a better way to separate ints and floats in the traits LinearCombination, DotProduct, ...
- [ ] Trouver un moyen d'automatiser la compilation avec les target features: fma, avx2, ...
- [ ] Reimplement everything for complex vector spaces

Vector:
- [x] Create a vector struct with basic operations: add, sub, scale.
- [x] Add indexing support
- [x] Add constructors from various types
- [x] Add iterator support
- [x] Implement the LinearCombination trait on slices of vectors. The f32 implementation must use fma.
- [x] Implement dot product (must use fma for f32)
- [ ] Implement LinearInterpolation
    - [x] Scalar implementation
    - [x] Vector implementation
- [ ] Implement norm
- [ ] Implement cosine
- [ ] Implement cross product
- [x] Implement reshape to a Matrix

Matrix
- [ ] Add indexing support
- [x] Add constructors:
    - [x] from a vector and a shape
    - [x] from a shape and a function that takes row and col indices
- [ ] Add iterator support
- [ ] Implement basic operations:
    - [ ] Add
    - [ ] Sub
    - [ ] Scale
- [ ] Implement matrix multiplication
    - [ ] With a vector
    - [ ] With a matrix
- [ ] Implement trace
- [ ] Implement transpose
- [ ] Implement reduced row echelon form
- [ ] Implement determinant
- [ ] Implement inverse
- [ ] Implement rank
- [ ] (Optional) Implement projection matrix
