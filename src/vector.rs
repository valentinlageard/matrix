use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Mul, Sub};
use std::simd::{f32x8, StdFloat};
use std::slice;
use std::slice::SliceIndex;

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
    pub scalars: Vec<T>,
    pub shape: (usize,),
}

// struct Matrix<T> {
//     scalars: Vec<T>,
//     shape: (usize, usize),
// }

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
        assert_eq!(
            self.shape, rhs.shape,
            "Shape mismatch: vector of shape {:?} can't be added to vector of shape {:?}",
            self.shape, rhs.shape
        );
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
        assert_eq!(
            self.shape, rhs.shape,
            "Shape mismatch: vector of shape {:?} can't be substracted to vector of shape {:?}",
            self.shape, rhs.shape
        );
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

// Potential TODO: Make the mul operation commutative. Seems to require lots of boilerplate code (like below) oradvanced macros
// impl Mul<&Vector<f32>> for f32
// {
//     type Output = Vector<f32>;

//     fn mul(self, rhs: &Vector<f32>) -> Self::Output {
//         rhs * self
//     }
// }

// This would be cool but it doesn't work in rust...
// See RFC 2451
// impl<T> Mul<&Vector<T>> for T {
//     type Output = Vector<T>;

//     fn mul(self, rhs: &Vector<T>) -> Self::Output {
//         rhs * self
//     }
// }

// =======================================LINEAR ALGEBRA============================================

/// A trait for linear combination
///
/// This trait allows to "specialize" the behaviour on &\[Vector\<f32\>\] so we can use fma operations
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
        let vector_shape = self[0].shape();
        for vector in self {
            assert_eq!(
                vector.shape(),
                vector_shape,
                "Shape mismatch: vector of shape {:?} can't be combined to vector of shape {:?}",
                vector.shape(),
                vector_shape
            )
        }

        // Create the result vector initialized with default values (0 for number types)
        let result = Vector {
            scalars: vec![T::default(); vector_shape.0],
            shape: vector_shape,
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
        let vector_size = self[0].size();
        for vector in self {
            assert_eq!(
                vector.size(),
                vector_size,
                "Shape mismatch: vector of shape {:?} can't be combined to vector of shape {:?}",
                vector.size(),
                vector_size
            )
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
                f32x8::splat(0.0),
                |accum, (coeff, vector)| {
                    let packed_coefficient = f32x8::splat(*coeff);
                    // SAFETY: The vector slice is guaranteed to be valid because I did my math fine when computing the slice index
                    // In addition, the beginning of the function asserts the vectors size
                    let vec_slice = unsafe { &vector.scalars.get_unchecked(slice_idx..) };
                    let packed_vec = unsafe {
                        let mut array = [vec_slice[0]; N_LANES];
                        let mut i = 0;
                        while i < N_LANES {
                            array[i] = *vec_slice.get_unchecked(i);
                            i += 1;
                        }
                        f32x8::from(array)
                    };
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
    T: Copy + Default + Mul + Mul<Output = T> + Add<Output = T>,
{
    type Output = T;

    fn dot_product(&self, rhs: &Self) -> Self::Output {
        assert_eq!(
            self.shape(),
            rhs.shape(),
            "Shape mismatch: vector of shape {:?} can't be combined to vector of shape {:?}",
            self.shape(),
            rhs.shape()
        );
        self.scalars
            .iter()
            .zip(rhs.scalars.iter())
            .fold(T::default(), |acc, (&x, &y)| acc + (x * y))
    }
}

// =============================================TESTS================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_shape_and_size() {
        let my_vector = Vector::from([0_f32, 1.0, 2.0, 3.0]);
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
        let my_vector = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        assert_eq!(my_vector[0], 0.0);
        assert_eq!(my_vector[1], 1.0);
        assert_eq!(my_vector[2], 2.0);
        assert_eq!(my_vector[3], 3.0);
        assert_eq!(my_vector[0..=1], [0., 1.]);
    }

    #[test]
    fn check_indexing_mut() {
        let mut my_vector = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        my_vector[0] += 1.;
        my_vector[1] += 2.;
        my_vector[2] += 3.;
        my_vector[3] += 4.;
        assert_eq!(my_vector[0], 1.0);
        assert_eq!(my_vector[1], 3.0);
        assert_eq!(my_vector[2], 5.0);
        assert_eq!(my_vector[3], 7.0);
    }

    #[test]
    fn check_comparison() {
        let mut my_array = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(my_array.to_vec());
        let my_vector2 = Vector::from(my_array.to_vec());
        my_array[0] += 1.0;
        let my_vector3 = Vector::from(my_array);
        assert_eq!(my_vector1, my_vector2);
        assert_ne!(my_vector1, my_vector3);
    }

    #[test]
    fn check_addition() {
        let my_vector1: Vector<f32> = Vector::from([0., 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from([0., 1.0, 2.0, 3.0]);
        let my_vector_result1 = &my_vector1 + &my_vector2;

        assert_eq!(my_vector_result1, Vector::from([0_f32, 2.0, 4.0, 6.0]));
    }

    #[test]
    #[should_panic]
    fn check_addition_error() {
        let my_vector1 = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from([0_f32, 1.0, 2.0]);
        let _ = &my_vector1 + &my_vector2;
    }

    #[test]
    fn check_substraction() {
        let my_vector1 = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        let my_vector_result = &my_vector1 - &my_vector2;

        assert_eq!(my_vector_result, Vector::from([0_f32, 0.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic]
    fn check_substraction_error() {
        let my_vector1 = Vector::from([0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from([0_f32, 1.0, 2.0]);
        let _ = &my_vector1 - &my_vector2;
    }

    #[test]
    fn check_scaling() {
        let my_array1 = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_array2 = (0..=10).map(|x| (x * 2) as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(my_array1);
        let my_vector2 = Vector::from(my_array2);
        let my_vector_result1 = &my_vector1 * 2_f32;
        // The following line doesn't compile because the operator can't be commutatively implemented
        // let my_vector_result2 = 2_f32 * &my_vector1;

        assert_eq!(my_vector_result1, my_vector2);
        // assert_eq!(my_vector_result2, my_vector2);
    }

    #[test]
    fn check_linear_combination() {
        for n_vectors in 1..20 {
            for vector_size in 1..=100 {
                // Check for f32
                let vectors: Vec<Vector<f32>> = (0..n_vectors)
                    .map(|_| Vector::from_iter((0..vector_size).map(|x| x as f32)))
                    .collect();
                let coefficients: Vec<f32> = vec![1.0; n_vectors];
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
        let v1: Vector<f32> = Vector::from_iter((0..10).map(|x| x as f32));
        let v2: Vector<f32> = Vector::from_iter((0..10).map(|x| x as f32));

        assert_eq!(v1.dot_product(&v2), 285.0);
    }
}
