# Notes

## Roadmap

- [ ] Choose the internal representation
    - While Vec uses heap allocation, it seems the right choice because arrays can't be sized at run time
    - Matrix should be implemented using a contiguous one-dimensional layout. 2d indexing should be implemented on top of this representation.
- [ ] How to use FMA
    - It seems that `.mul_add()` already uses fma but we need to confirm it by checking the assembly code generated. This would be the best choice because the code would be more portable (the compiler chooses the operation given the target architecture).
    - We can directly uses x86 fma instructions using the `core::arch` module if the first solution fails.