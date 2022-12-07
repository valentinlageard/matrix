// #![allow(dead_code)]
#![feature(portable_simd)]

use std::arch::x86_64::{__m128, _mm_fmadd_ps, _mm_set1_ps, _mm_set_ps, _mm_setzero_ps};
use std::ops::{Add, AddAssign, Mul, Sub};
use std::simd::f32x4;
use std::fmt::Debug;

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

// struct Matrix<T> {
//     scalars: Vec<T>,
//     shape: (usize, usize),
// }

impl<T> Vector<T>
where
    T: Copy,
{
    /// Creates a vector from a slice
    pub fn from(slice: &[T]) -> Self {
        Vector {
            scalars: slice.to_vec(),
            shape: (slice.len(),),
        }
    }

    /// Returns the shape of the vector
    pub fn shape(&self) -> (usize,) {
        self.shape
    }

    /// Returns the size (total number of elements) of the vector
    pub fn size(&self) -> usize {
        self.shape.0
    }
}

impl<T> Add for &Vector<T>
where
    T: Copy + Add,
    Vec<T>: FromIterator<<T as Add>::Output>,
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

impl<T> Sub for &Vector<T>
where
    T: Copy + Sub,
    Vec<T>: FromIterator<<T as Sub>::Output>,
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

impl<T> Mul<T> for &Vector<T>
where
    T: Copy + Mul,
    Vec<T>: FromIterator<<T as Mul>::Output>,
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

pub trait LinearCombination<'a> {
    type Coefficients;
    type LinearCombinationResult;

    fn linear_combination(
        &'a self,
        coefficients: Self::Coefficients,
    ) -> Self::LinearCombinationResult;
}

impl<'a, T> LinearCombination<'a> for [Vector<T>]
where
    T: 'a + Default + Int + Copy + Add + Mul + Sub,
    Vec<T>: FromIterator<<T as Mul>::Output> + FromIterator<<T as Add>::Output>,
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

        // Create an accums vector to store the accumulating results
        let n_accums = self[0].shape().0 / 4 + (self[0].shape().0 % 4 == 0) as usize;
        let mut accums = (0..n_accums + 1)
            .map(|_| unsafe { _mm_setzero_ps() })
            .collect::<Vec<__m128>>();
        // For each accum
        for scalar_row in (0..self[0].shape().0).step_by(4) {
            // For each coefficient
            let accum = &mut accums[scalar_row / 4];
            for (v_idx, &coefficient) in coefficients.iter().enumerate() {
                // Pack coefficient and self' scalars
                let packed_coefficient = unsafe { _mm_set1_ps(coefficient) };
                // TODO: We could isolate the remainder's logic from the chunk's logic so we don't
                // need to get/copy/unwrap for the valid chunks
                let packed_vector_scalars = unsafe {
                    _mm_set_ps(
                        self[v_idx]
                            .scalars
                            .get(scalar_row + 3)
                            .copied()
                            .unwrap_or_default(),
                        self[v_idx]
                            .scalars
                            .get(scalar_row + 2)
                            .copied()
                            .unwrap_or_default(),
                        self[v_idx]
                            .scalars
                            .get(scalar_row + 1)
                            .copied()
                            .unwrap_or_default(),
                        self[v_idx]
                            .scalars
                            .get(scalar_row)
                            .copied()
                            .unwrap_or_default(),
                    )
                };
                // Perform fused multiply accumulate
                *accum = unsafe { _mm_fmadd_ps(packed_vector_scalars, packed_coefficient, *accum) };
            }
        }
        // Convert back to a vector
        // TODO: There may be a better way to convert back ?
        let mut result = Vector::<f32>::from(&vec![0.; self[0].shape().0]);
        for (i, &accum) in accums.iter().enumerate() {
            let accum = f32x4::from(accum).to_array().to_vec();
            for (j, &accum_value) in accum.iter().enumerate() {
                if i * 4 + j > result.shape().0 - 1 {
                    break;
                }
                result.scalars[i * 4 + j] = accum_value;
            }
        }
        result
    }
}

pub trait DotProduct {
    type Output;

    fn dot_product(&self, rhs: &Self) -> Self::Output;
}

impl<T> DotProduct for Vector<T>
where
    T: Copy + Mul + Default + AddAssign<<T as std::ops::Mul>::Output> + Debug,
    Vec<T>: FromIterator<<T as Mul>::Output>,
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
            .fold(T::default(), |mut acc, (&x, &y)| {
                acc += x * y;
                acc
            })
    }
}

// pub fn linear_combination<'a, T>(vectors: &'a [Vector<T>], coefficients: &'a [T])
// where
//     T: Copy,
//     Vector<T>: LinearCombination<'a>,
// {
//     vectors.linear_combination(coefficients);
// }

// =============================================TESTS================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_shape_and_size() {
        let my_vector = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        assert_eq!(my_vector.shape(), (4,));
        assert_eq!(my_vector.size(), 4);
    }

    #[test]
    fn check_comparison() {
        let mut my_array = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(&my_array);
        let my_vector2 = Vector::from(&my_array);
        my_array[0] += 1.0;
        let my_vector3 = Vector::from(&my_array);
        assert_eq!(my_vector1, my_vector2);
        assert_ne!(my_vector1, my_vector3);
    }

    #[test]
    fn check_addition() {
        let my_vector1: Vector<f32> = Vector::from(&[0., 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from(&[0., 1.0, 2.0, 3.0]);
        let my_vector_result1 = &my_vector1 + &my_vector2;

        assert_eq!(my_vector_result1, Vector::from(&[0_f32, 2.0, 4.0, 6.0]));
    }

    #[test]
    #[should_panic]
    fn check_addition_error() {
        let my_vector1 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from(&[0_f32, 1.0, 2.0]);
        let _ = &my_vector1 + &my_vector2;
    }

    #[test]
    fn check_substraction() {
        let my_vector1 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        let my_vector_result = &my_vector1 - &my_vector2;

        assert_eq!(my_vector_result, Vector::from(&[0_f32, 0.0, 0.0, 0.0]));
    }

    #[test]
    #[should_panic]
    fn check_substraction_error() {
        let my_vector1 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from(&[0_f32, 1.0, 2.0]);
        let _ = &my_vector1 - &my_vector2;
    }

    #[test]
    fn check_scaling() {
        let my_array1 = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_array2 = (0..=10).map(|x| (x * 2) as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(&my_array1);
        let my_vector2 = Vector::from(&my_array2);
        let my_vector_result1 = &my_vector1 * 2_f32;
        // The following line doesn't compile because the relation isn't commutative right now
        // let my_vector_result2 = 2_f32 * &my_vector1;

        assert_eq!(my_vector_result1, my_vector2);
        // assert_eq!(my_vector_result2, my_vector2);
    }

    #[test]
    fn check_linear_combination() {
        let v1: Vector<f32> = Vector::from(&[1., 2., 3.]);
        let v2 = Vector::from(&[0., 10., -100.]);
        let result = [v1, v2].linear_combination(&[10., -2.]);

        let v3: Vector<i32> = Vector::from(&[1, 2, 3]);
        let v4 = Vector::from(&[0, 10, -100]);
        let result2 = [v3, v4].linear_combination(&[10, -2]);

        assert_eq!(result, Vector::<f32>::from(&[10., 0., 230.]));
        assert_eq!(result2, Vector::<i32>::from(&[10, 0, 230]));
    }

    #[test]
    fn check_dot_product() {
        let v1: Vector<f32> = Vector::from(&((0..10).map(|x| x as f32).collect::<Vec<f32>>()));
        let v2: Vector<f32> = Vector::from(&((0..10).map(|x| x as f32).collect::<Vec<f32>>()));

        assert_eq!(v1.dot_product(&v2), 285.0);
    }
}
