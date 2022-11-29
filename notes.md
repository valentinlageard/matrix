# Notes

## Development notes

- [ ] Choose the internal representation
    - While Vec uses heap allocation, it seems the right choice because arrays can't be sized at run time
    - Matrix should be implemented using a contiguous one-dimensional layout. 2d indexing should be implemented on top of this representation.
- [ ] How to use FMA
    - It seems that `.mul_add()` already uses fma but we need to confirm it by checking the assembly code generated. This would be the best choice because the code would be more portable (the compiler chooses the operation given the target architecture).
    - We can directly uses x86 fma instructions using the `core::arch` module if the first solution fails.
- [x] Should we implement so that we can use the standard numeric operators +, -. ... ?
    - Actually, the "operator overloading" traits can define an arbitrary output type. This means that we could in theory return a Result for instance. This would be impractical because it would mean that we would need to unpack the result when chaining linear algebra operations.
    - Another solution is to panic if the user tries to make an operation that fails (e.g. shape mismatch). This way, we can get rid of the error and always return a valid output or dramatically fail. This is consistent with the subject saying that if vectors don't have the same dimension, the result is undefined.
- [ ] Should we make available both in-place and pure versions of the methods ?
    - I think yes because in place is less practical but can be a bit more efficient because we don't need to allocate a new internal vec. In place methods can be either consuming the vector or mutably borrowing it. What is best ?
    - Pure implementations can however be chained more easily.
- [ ] Ideally we would need a benchmark environment to check empirically for the complexity requirements. We could make a python script that executes various versions of the programs, collect the data and print nice graphs with matplotlib.