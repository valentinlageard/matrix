use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use std::simd::{f32x8, SimdFloat, StdFloat};
use std::slice;
use std::slice::SliceIndex;

use crate::matrix::Matrix;

pub trait Int {}
pub trait Float {}

impl Float for f64 {}
impl Float for f32 {}
impl Int for i64 {}
impl Int for i32 {}
impl Int for i16 {}
impl Int for i8 {}
impl Int for isize {}
impl Int for u64 {}
impl Int for u32 {}
impl Int for u16 {}
impl Int for u8 {}
impl Int for usize {}

/// A vector designed for linear algebra operations
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub struct Vector<T>
where
    T: Copy,
{
    scalars: Vec<T>,
    shape: (usize,),
}

impl<T> Vector<T>
where
    T: Copy,
{
    /// Returns the shape of the vector
    pub fn shape(&self) -> (usize,) {
        self.shape
    }

    /// Returns the size (total number of elements) of the vector
    pub fn size(&self) -> usize {
        self.shape.0
    }

    /// Returns an iterator on the vector's scalars
    pub fn iter(&self) -> slice::Iter<T> {
        self.scalars.iter()
    }

    /// Returns a mutable iterator on the vector's scalars
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.scalars.iter_mut()
    }

    pub fn reshape(self, shape: (usize, usize)) -> Matrix<T> {
        assert_eq!(
            shape.0 * shape.1,
            self.size(),
            "Vector of size {:?} can't be reshaped into matrix of shape {:?}",
            self.size(),
            shape
        );
        Matrix {
            scalars: self.scalars,
            shape
        }
    }
}

// ==========================================INDEXING===============================================

impl<T, I: SliceIndex<[T]>> Index<I> for Vector<T>
where
    T: Copy,
{
    type Output = I::Output;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(&*self.scalars, index)
    }
}

impl<T, I: SliceIndex<[T]>> IndexMut<I> for Vector<T>
where
    T: Copy,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(&mut *self.scalars, index)
    }
}

// ====================================FROM CONSTRUCTION============================================

impl<T> From<Vec<T>> for Vector<T>
where
    T: Copy,
{
    fn from(value: Vec<T>) -> Self {
        let shape = (value.len(),);
        Vector {
            scalars: value,
            shape,
        }
    }
}

impl<T> From<&Vec<T>> for Vector<T>
where
    T: Copy,
{
    fn from(value: &Vec<T>) -> Self {
        Vector {
            scalars: value.clone(),
            shape: (value.len(),),
        }
    }
}

impl<T> From<&[T]> for Vector<T>
where
    T: Copy,
{
    fn from(value: &[T]) -> Self {
        Vector {
            scalars: value.to_vec(),
            shape: (value.len(),),
        }
    }
}

impl<T> From<&mut [T]> for Vector<T>
where
    T: Copy,
{
    fn from(value: &mut [T]) -> Self {
        Vector {
            scalars: value.to_vec(),
            shape: (value.len(),),
        }
    }
}

impl<T, const N: usize> From<[T; N]> for Vector<T>
where
    T: Copy,
{
    fn from(value: [T; N]) -> Self {
        Vector {
            scalars: value.to_vec(),
            shape: (value.len(),),
        }
    }
}

// =========================================ITERATORS===============================================

impl<T> FromIterator<T> for Vector<T>
where
    T: Copy,
{
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let scalars: Vec<T> = iter.into_iter().collect();
        let shape = (scalars.len(),);
        Vector { scalars, shape }
    }
}

impl<T> IntoIterator for Vector<T>
where
    T: Copy,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.scalars.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Vector<T>
where
    T: Copy,
{
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.scalars.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Vector<T>
where
    T: Copy,
{
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.scalars.iter_mut()
    }
}

// ====================================ARITHMETIC OPERATORS=========================================

/// Adds two vectors
impl<T> Add for &Vector<T>
where
    T: Copy + Add + Add<Output = T>,
{
    type Output = Vector<T>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_vector_shape_match(self, rhs);
        Vector {
            scalars: self
                .scalars
                .iter()
                .zip(&rhs.scalars)
                .map(|(&x, &y)| x + y)
                .collect(),
            shape: self.shape,
        }
    }
}

/// Substracts two vectors
impl<T> Sub for &Vector<T>
where
    T: Copy + Sub + Sub<Output = T>,
{
    type Output = Vector<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_vector_shape_match(self, rhs);
        Vector {
            scalars: self
                .scalars
                .iter()
                .zip(&rhs.scalars)
                .map(|(&x, &y)| x - y)
                .collect(),
            shape: self.shape,
        }
    }
}

/// Scales a vector by a scalar
///
/// Be aware that this operator isn't implemented in a commutative manner (see RFC 2451) !
impl<T> Mul<T> for &Vector<T>
where
    T: Copy + Mul + Mul<Output = T>,
{
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector {
            scalars: self.scalars.iter().map(|&x| x * rhs).collect(),
            shape: self.shape,
        }
    }
}

// =======================================LINEAR ALGEBRA============================================

/// A trait for linear combination
///
/// This trait allows to "specialize" the behaviour on &\[Vector\<f32\>\] so we can use fma
/// operations
pub trait LinearCombination<'a> {
    /// The coefficients type
    type Coefficients;
    /// The result type
    type LinearCombinationResult;

    /// Returns the linear combination of self with coefficients
    fn linear_combination(
        &'a self,
        coefficients: Self::Coefficients,
    ) -> Self::LinearCombinationResult;
}

/// Generic implementation of the linear combination
impl<'a, T> LinearCombination<'a> for [Vector<T>]
where
    T: 'a + Default + Copy + Int + Add + Mul + Sub + Add<Output = T> + Mul<Output = T>,
{
    type Coefficients = &'a [T];
    type LinearCombinationResult = Vector<T>;

    fn linear_combination(
        &'a self,
        coefficients: Self::Coefficients,
    ) -> Self::LinearCombinationResult {
        // Check for errors
        assert_eq!(
            self.len(),
            coefficients.len(),
            "Length mismatch between {:?} vectors and {:?} coefficients",
            self.len(),
            coefficients.len()
        );
        for vector in self {
            assert_vector_shape_match(&self[0], vector);
        }

        // Create the result vector initialized with default values (0 for number types)
        let result = Vector {
            scalars: vec![T::default(); self[0].size()],
            shape: self[0].shape(),
        };

        // Compute scaled vectors
        let scaled_vectors: Vec<Vector<T>> = coefficients
            .iter()
            .zip(self)
            .map(|(&coefficient, vector)| vector * coefficient)
            .collect();

        // Accumulate scaled vectors
        scaled_vectors
            .iter()
            .fold(result, |acc, scaled_vector| &acc + scaled_vector)
    }
}

/// Optimized implementation of linear combination for f32
impl<'a> LinearCombination<'a> for [Vector<f32>] {
    type Coefficients = &'a [f32];
    type LinearCombinationResult = Vector<f32>;

    fn linear_combination(
        &'a self,
        coefficients: Self::Coefficients,
    ) -> Self::LinearCombinationResult {
        // Check for errors
        assert_eq!(
            self.len(),
            coefficients.len(),
            "Length mismatch between {:?} vectors and {:?} coefficients",
            self.len(),
            coefficients.len()
        );
        for vector in self {
            assert_vector_shape_match(&self[0], vector);
        }

        const N_LANES: usize = 8;
        let vector_size = self[0].size();

        // Create the vector to store the result (and the partial computations !)
        let mut result_vector = Vector {
            scalars: vec![0.; vector_size],
            shape: (vector_size,),
        };

        let (res_prefix, res_middle, res_suffix) = result_vector.scalars.as_simd_mut::<N_LANES>();

        // For prefix results, apply normal logic
        for (i, res) in res_prefix.iter_mut().enumerate() {
            for (&coefficient, vector) in coefficients.iter().zip(self.iter()) {
                *res = coefficient.mul_add(vector[i], *res);
            }
        }

        // For middle results, hint the compiler to use packed fma operations
        for (i, packed_res) in res_middle.iter_mut().enumerate() {
            // We're using wrapping ops to bypass the overflow checks
            let slice_idx = res_prefix.len().wrapping_add(i.wrapping_mul(N_LANES));
            *packed_res = coefficients.iter().zip(self.iter()).fold(
                f32x8::splat(0.),
                |accum, (coeff, vector)| {
                    let packed_coefficient = f32x8::splat(*coeff);
                    // SAFETY: The vector slice is guaranteed to be valid because I did my pointer
                    // arithmetics fine when computing the slice index.  In addition, the beginning
                    // of the function asserts the vectors size
                    let vec_slice = unsafe { &vector.scalars.get_unchecked(slice_idx..) };
                    let packed_vec = unsafe { slice_to_simd_f32x8_unchecked(vec_slice) };
                    packed_vec.mul_add(packed_coefficient, accum)
                },
            );
        }

        // For suffix results, apply normal logic
        for (i, res) in res_suffix.iter_mut().enumerate() {
            for (&coefficient, vector) in coefficients.iter().zip(self.iter()) {
                *res = coefficient.mul_add(
                    vector[res_prefix.len() + res_middle.len() * N_LANES + i],
                    *res,
                );
            }
        }
        result_vector
    }
}

/// A trait for the dot product operation
pub trait DotProduct {
    /// The dot product result type
    type Output;

    /// Returns the dot product
    fn dot_product(&self, rhs: &Self) -> Self::Output;
}

impl<T> DotProduct for Vector<T>
where
    T: Copy + Default + Int + Mul + Mul<Output = T> + Add<Output = T>,
{
    type Output = T;

    fn dot_product(&self, rhs: &Self) -> Self::Output {
        assert_vector_shape_match(self, rhs);

        self.scalars
            .iter()
            .zip(rhs.scalars.iter())
            .fold(T::default(), |acc, (&x, &y)| acc + (x * y))
    }
}

impl DotProduct for Vector<f32> {
    type Output = f32;

    fn dot_product(&self, rhs: &Self) -> Self::Output {
        assert_vector_shape_match(self, rhs);

        // The naive algorithm is more efficient than the fma optimized one for vectors with size
        // less than 16 so we select the algorithm based on the vector size
        match self.size().cmp(&16) {
            // Naive algorithm
            Ordering::Less => self
                .scalars
                .iter()
                .zip(rhs.scalars.iter())
                .fold(0., |acc, (&x, &y)| x.mul_add(y, acc)),
            // Fma algorithm
            _ => {
                const N_LANES: usize = 8;
                // Divide the vectors in chunks
                let v1_chunks = self.scalars.chunks_exact(N_LANES);
                let v2_chunks = rhs.scalars.chunks_exact(N_LANES);
                let v1_remainder = v1_chunks.remainder();
                let v2_remainder = v2_chunks.remainder();
                // Multiply each chunk and accumulate then reduce to a sum
                let chunk_result = v1_chunks
                    .zip(v2_chunks)
                    .fold(f32x8::splat(0.), |accum, (v1_chunk, v2_chunk)| {
                        let v1_simd = unsafe { slice_to_simd_f32x8_unchecked(v1_chunk) };
                        let v2_simd = unsafe { slice_to_simd_f32x8_unchecked(v2_chunk) };
                        v1_simd.mul_add(v2_simd, accum)
                    })
                    .reduce_sum();
                // Naive algorithm for the remainder chunk
                let remainder_result = v1_remainder
                    .iter()
                    .zip(v2_remainder)
                    .fold(0., |accum, (&x1, &x2)| x1.mul_add(x2, accum));
                chunk_result + remainder_result
            }
        }
    }
}

pub trait LinearInterpolation {
    type Output;

    fn lerp(self, rhs: Self, t: f32) -> Self::Output;
}

/// Linear interpolation for scalar types
impl<T> LinearInterpolation for T
where
    T: Add<Output = T> + Mul<f32, Output = T>,
{
    type Output = T;

    fn lerp(self, rhs: Self, t: f32) -> Self::Output {
        self * (1. - t) + rhs * t
    }
}

/// Linear interpolation for vector types
///
/// No specialization for f32 has been stabilized as the compiler vectorizes and unrolls
/// aggressively resulting in a better algorithm than anything I tried
impl<'a, T> LinearInterpolation for &'a Vector<T>
where
    T: Copy + Add<Output = T> + Mul<f32, Output = T>,
{
    type Output = Vector<T>;

    fn lerp(self, rhs: Self, t: f32) -> Self::Output {
        assert_vector_shape_match(self, rhs);
        Vector {
            scalars: self
                .scalars
                .iter()
                .zip(rhs.scalars.iter())
                .map(|(&x1, &x2)| x1 * (1.0 - t) + x2 * t)
                .collect(),
            shape: self.shape(),
        }
    }
}

// /// Fma optimized implementation for f32 vector types (isn't better that what the compiler does)
// impl<'a> LinearInterpolation for &'a Vector<f32> {
//     type Output = Vector<f32>;

//     fn lerp(self, rhs: Self, t: f32) -> Self::Output {
//         // TODO: Check vector sizes !

//         const N_LANES: usize = 8;
//         // Create the vector to store the result (and the partial computations !)
//         let mut result_vector = Vector {
//             scalars: vec![0.0f32; self.size()],
//             shape: self.shape(),
//         };

//         let (res_prefix, res_middle, res_suffix) =
// result_vector.scalars.as_simd_mut::<N_LANES>();         let middle_start_idx = res_prefix.len();
//         let suffix_start_idx = res_prefix.len() + res_middle.len() * N_LANES;

//         for ((x1, x2), res_scalar) in self.scalars.iter().zip(rhs.scalars.iter()).zip(res_prefix)
// {             *res_scalar = x1 * (1.0 - t) + x2 * t;
//         }

//         let packed_t = f32x8::splat(t);
//         let packed_one_minus_t = f32x8::splat(1.0 - t);
//         for (i, packed_res) in res_middle.iter_mut().enumerate() {
//             let slice_idx = middle_start_idx.wrapping_add(i.wrapping_mul(N_LANES));
//             // SAFETY: We already check the vectors size and our pointer arithmetic is right
//             let slice_x1 = unsafe { self.scalars.get_unchecked(slice_idx..) };
//             let slice_x2 = unsafe { rhs.scalars.get_unchecked(slice_idx..) };
//             let packed_x1 = unsafe { slice_to_simd_f32x8_unchecked(slice_x1) };
//             let packed_x2 = unsafe { slice_to_simd_f32x8_unchecked(slice_x2) };
//             *packed_res = packed_t.mul_add(packed_x2, (packed_one_minus_t) * packed_x1);
//         }

//         for (i, res_scalar) in res_suffix.iter_mut().enumerate() {
//             // SAFETY: We already check the vectors size and our pointer arithmetic is right
//             let x1 = unsafe { self.scalars.get_unchecked(suffix_start_idx + i) };
//             let x2 = unsafe { rhs.scalars.get_unchecked(suffix_start_idx + i) };
//             *res_scalar = x1 * (1.0 - t) + x2 * t;
//         }

//         // // TODO: Performance isn't stable and isn't always better than naive implementation...
//         // // Chunkize the input vectors and the result vector
//         // let v1_chunks = self.scalars.chunks_exact(N_LANES);
//         // let v2_chunks = rhs.scalars.chunks_exact(N_LANES);
//         // // Maybe we could use something else than as_chunks_mut ?
//         // let (result_vector_chunks, result_vector_remainder) =
//         //     result_vector.scalars.as_chunks_mut::<N_LANES>();
//         // let v1_remainder = v1_chunks.remainder();
//         // let v2_remainder = v2_chunks.remainder();
//         // // Broadcast the interpolator value to a simd vector
//         // let packed_t = f32x8::splat(t);
//         // let packed_one_minus_t = f32x8::splat(1.0 - t);

//         // // For each chunk, use packed fma
//         // v1_chunks.zip(v2_chunks).zip(result_vector_chunks).for_each(
//         //     |((v1_chunk, v2_chunk), packed_res)| {
//         //         let packed_x1 = unsafe { slice_to_simd_f32x8_unchecked(v1_chunk) };
//         //         let packed_x2 = unsafe { slice_to_simd_f32x8_unchecked(v2_chunk) };
//         //         // let mut simdres = unsafe { slice_to_simd_f32x8_unchecked(res_chunk) };
//         //         *packed_res = packed_t.mul_add(packed_x2, (packed_one_minus_t) *
// packed_x1).into();         //         // *res_chunk = t_simd.mul_add(simd2,
// -t_simd.mul_add(simd1, -simd1)).into();         //     },
//         // );

//         // // For the remainder, use scalar fma
//         // v1_remainder
//         //     .iter()
//         //     .zip(v2_remainder.iter())
//         //     .zip(result_vector_remainder)
//         //     .for_each(|((&x1, &x2), res)| *res = x1 * (1.0 - t) + x2 * t);

//         result_vector
//     }
// }

// TODO: Implement LinearInterpolation for Matrix<T: Int> et Matrix<f32>

// ==============================================UTILS==============================================

#[inline]
fn assert_vector_shape_match<T: Copy>(v1: &Vector<T>, v2: &Vector<T>) {
    assert_eq!(
        v1.shape(),
        v2.shape(),
        "Shape mismatch: vector of shape {:?} can't be combined to vector of shape {:?}",
        v1.shape(),
        v2.shape()
    );
}

#[inline]
unsafe fn slice_to_simd_f32x8_unchecked(slice: &[f32]) -> f32x8 {
    let mut array = [0.; 8];
    for (i, elem) in array.iter_mut().enumerate() {
        *elem = *slice.get_unchecked(i);
    }
    f32x8::from(array)
}

// ==============================================TESTS==============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_shape_and_size() {
        let my_vector = Vector::from([0_f32, 1., 2., 3.]);
        assert_eq!(my_vector.shape(), (4,));
        assert_eq!(my_vector.size(), 4);
    }

    #[test]
    fn check_construction() {
        let array = [1i32, 2, 3];
        let mut mut_array = [1i32, 2, 3];
        let vec = vec![1i32, 2, 3];
        let v_from_slice = Vector::from(&array[..]);
        let v_from_fixed_array = Vector::from(array);
        let v_from_mut_slice = Vector::from(&mut mut_array[..]);
        let v_from_vec_ref = Vector::from(&vec);
        let v_from_vec = Vector::from(vec);
        let v_from_iter = Vector::from_iter(array.into_iter());

        let vec = vec![1i32, 2, 3];

        assert_eq!(v_from_slice.scalars, vec);
        assert_eq!(v_from_fixed_array.scalars, vec);
        assert_eq!(v_from_mut_slice.scalars, vec);
        assert_eq!(v_from_vec_ref.scalars, vec);
        assert_eq!(v_from_vec.scalars, vec);
        assert_eq!(v_from_iter.scalars, vec);
    }

    #[test]
    fn check_iteration() {
        let v: Vector<i32> = Vector::from([1, 2, 3]);
        let mut v_mut = v.clone();
        let vec: Vec<i32> = vec![2, 3, 4];
        let vec_from_v_iter: Vec<i32> = v.iter().map(|&x| x + 1).collect();
        let vec_from_v_mut_iter: Vec<i32> = v_mut.iter_mut().map(|&mut x| x + 1).collect();

        assert_eq!(vec_from_v_iter, vec);
        assert_eq!(vec_from_v_mut_iter, vec);

        for i in &v {
            println!("{i}");
        }
        for i in v {
            println!("{i}");
        }
        for i in &mut v_mut {
            println!("{i}");
        }
    }

    #[test]
    fn check_indexing() {
        let my_vector = Vector::from([0_f32, 1., 2., 3.]);
        assert_eq!(my_vector[0], 0.);
        assert_eq!(my_vector[1], 1.);
        assert_eq!(my_vector[2], 2.);
        assert_eq!(my_vector[3], 3.);
        assert_eq!(my_vector[0..=1], [0., 1.]);
    }

    #[test]
    fn check_indexing_mut() {
        let mut my_vector = Vector::from([0_f32, 1., 2., 3.]);
        my_vector[0] += 1.;
        my_vector[1] += 2.;
        my_vector[2] += 3.;
        my_vector[3] += 4.;
        assert_eq!(my_vector[0], 1.);
        assert_eq!(my_vector[1], 3.);
        assert_eq!(my_vector[2], 5.);
        assert_eq!(my_vector[3], 7.);
    }

    #[test]
    fn check_comparison() {
        let mut my_array = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(my_array.to_vec());
        let my_vector2 = Vector::from(my_array.to_vec());
        my_array[0] += 1.;
        let my_vector3 = Vector::from(my_array);
        assert_eq!(my_vector1, my_vector2);
        assert_ne!(my_vector1, my_vector3);
    }

    #[test]
    fn check_addition() {
        let my_vector1: Vector<f32> = Vector::from([0., 1., 2., 3.]);
        let my_vector2 = Vector::from([0., 1., 2., 3.]);
        let my_vector_result1 = &my_vector1 + &my_vector2;

        assert_eq!(my_vector_result1, Vector::from([0_f32, 2., 4., 6.]));
    }

    #[test]
    #[should_panic]
    fn check_addition_error() {
        let my_vector1 = Vector::from([0_f32, 1., 2., 3.]);
        let my_vector2 = Vector::from([0_f32, 1., 2.]);
        let _ = &my_vector1 + &my_vector2;
    }

    #[test]
    fn check_substraction() {
        let my_vector1 = Vector::from([0_f32, 1., 2., 3.]);
        let my_vector2 = Vector::from([0_f32, 1., 2., 3.]);
        let my_vector_result = &my_vector1 - &my_vector2;

        assert_eq!(my_vector_result, Vector::from([0_f32, 0., 0., 0.]));
    }

    #[test]
    #[should_panic]
    fn check_substraction_error() {
        let my_vector1 = Vector::from([0_f32, 1., 2., 3.]);
        let my_vector2 = Vector::from([0_f32, 1., 2.]);
        let _ = &my_vector1 - &my_vector2;
    }

    #[test]
    fn check_scaling() {
        let my_array1 = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_array2 = (0..=10).map(|x| (x * 2) as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(my_array1);
        let my_vector2 = Vector::from(my_array2);
        let my_vector_result1 = &my_vector1 * 2_f32;
        // The following line doesn't compile because the operator can't be commutatively
        // implemented let my_vector_result2 = 2_f32 * &my_vector1;

        assert_eq!(my_vector_result1, my_vector2);
        // assert_eq!(my_vector_result2, my_vector2);
    }

    // TODO: check cases linear combination should panic !

    #[test]
    fn check_linear_combination() {
        for n_vectors in 1..20 {
            for vector_size in 1..=100 {
                // Check for f32
                let vectors: Vec<Vector<f32>> = (0..n_vectors)
                    .map(|_| Vector::from_iter((0..vector_size).map(|x| x as f32)))
                    .collect();
                let coefficients: Vec<f32> = vec![1.; n_vectors];
                let result_should_be =
                    Vector::from_iter((0..vector_size).map(|x| (x * n_vectors) as f32));
                let result = vectors.linear_combination(&coefficients);
                assert_eq!(result, result_should_be);

                // Check default implementation
                let vectors: Vec<Vector<i32>> = (0..n_vectors)
                    .map(|_| Vector::from_iter((0..vector_size).map(|x| x as i32)))
                    .collect();
                let coefficients: Vec<i32> = vec![1; n_vectors];
                let result_should_be =
                    Vector::from_iter((0..vector_size).map(|x| (x * n_vectors) as i32));
                let result = vectors.linear_combination(&coefficients);
                assert_eq!(result, result_should_be);
            }
        }
    }

    #[test]
    fn check_dot_product() {
        let expected_results = [
            0, 1, 5, 14, 30, 55, 91, 140, 204, 285, 385, 506, 650, 819, 1015, 1240, 1496, 1785,
            2109, 2470, 2870, 3311, 3795, 4324, 4900, 5525, 6201, 6930, 7714, 8555, 9455, 10416,
            11440, 12529, 13685, 14910, 16206, 17575, 19019, 20540, 22140, 23821, 25585, 27434,
            29370, 31395, 33511, 35720, 38024, 40425, 42925, 45526, 48230, 51039, 53955, 56980,
            60116, 63365, 66729, 70210, 73810, 77531, 81375, 85344, 89440, 93665, 98021, 102510,
            107134, 111895, 116795, 121836, 127020, 132349, 137825, 143450, 149226, 155155, 161239,
            167480, 173880, 180441, 187165, 194054, 201110, 208335, 215731, 223300, 231044, 238965,
            247065, 255346, 263810, 272459, 281295, 290320, 299536, 308945, 318549, 328350,
        ];
        for i in 0..100 {
            // Check for f32
            let v1: Vector<f32> = Vector::from_iter((0..=i).map(|x| x as f32));
            let v2 = Vector::from_iter((0..=i).map(|x| x as f32));
            assert_eq!(v1.dot_product(&v2), expected_results[i] as f32);

            // Check for default implementation
            let v1: Vector<i32> = Vector::from_iter((0..=i).map(|x| x as i32));
            let v2 = Vector::from_iter((0..=i).map(|x| x as i32));
            assert_eq!(v1.dot_product(&v2), expected_results[i]);
        }
    }

    // TODO: Check cases where dot product should panic !

    #[test]
    fn check_linear_interpolation() {
        for vector_size in 1..=1000 {
            let v1 = Vector::from_iter((0..vector_size).map(|x| x as f32));
            let v2 = Vector::from_iter((0..vector_size).map(|x| (x + 1) as f32));

            let result = v1.lerp(&v2, 0.0);
            assert_eq!(result, v1);

            let result = v1.lerp(&v2, 1.0);
            assert_eq!(result, v2);

            let expected_result = Vector::from_iter((0..vector_size).map(|x| x as f32 + 0.5));
            let result = v1.lerp(&v2, 0.5);
            assert_eq!(result, expected_result);
        }
    }
}
