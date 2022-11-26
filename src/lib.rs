// #![allow(dead_code)]

use std::ops::{Add, Sub, Mul};

#[derive(Debug, Eq, PartialEq)]
/// Error types for the matrix library
pub enum VectorError {
    /// This variant is thrown when the vectors shapes do not match
    ShapeMismatch,
}

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

impl<'a, T> Vector<T>
where
    T: 'a + Copy + Mul<Output = T>,
    &'a T: Add<Output = T> + Sub<Output = T>
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

    /// Adds two vectors
    ///
    /// # Errors
    /// Returns Err(VectorError::ShapeMismatch) if the vectors have different shapes
    pub fn add(&'a self, rhs: &'a Vector<T>) -> Result<Vector<T>, VectorError> {
        if self.shape() != rhs.shape() {
            return Err(VectorError::ShapeMismatch);
        }
        Ok(Vector {
            scalars: self
                .scalars
                .iter()
                .zip(&rhs.scalars)
                .map(|(x, y)| x + y)
                .collect(),
            shape: self.shape(),
        })
    }

    /// Substracts two vectors
    ///
    /// # Errors
    /// Returns Err(VectorError::ShapeMismatch) if the vectors have different shapes
    pub fn sub(&'a self, rhs: &'a Vector<T>) -> Result<Vector<T>, VectorError> {
        if self.shape() != rhs.shape() {
            return Err(VectorError::ShapeMismatch);
        }
        Ok(Vector {
            scalars: self
                .scalars
                .iter()
                .zip(&rhs.scalars)
                .map(|(x, y)| x - y)
                .collect(),
            shape: self.shape(),
        })
    }

    /// Scales a vector by a scalar
    pub fn scale(&'a self, rhs: T) -> Vector<T> {
        Vector {
            scalars: self.scalars.iter().map(|&x| x * rhs).collect(),
            shape: self.shape(),
        }
    }
}

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
        let my_array1 = [0_f32, 1.0, 2.0, 3.0];
        let my_array2 = [0_f32, 1.0, 2.0];
        let my_vector1 = Vector::from(&my_array1);
        let my_vector2 = Vector::from(&my_array2);
        let my_vector3 = Vector::from(&my_array1);

        assert_eq!(
            my_vector1.add(&my_vector3),
            Ok(Vector::from(&[0_f32, 2.0, 4.0, 6.0]))
        );
        assert_eq!(my_vector1.add(&my_vector2), Err(VectorError::ShapeMismatch));
    }

    #[test]
    #[ignore = "takes time"]
    fn check_big_addition() {
        let my_array = (0..=100000000).map(|x| x as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(&my_array);
        let my_vector2 = Vector::from(&my_array);

        let _my_vector3 = my_vector1.add(&my_vector2);
    }

    #[test]
    fn check_substraction() {
        let my_array1 = [0_f32, 1.0, 2.0, 3.0];
        let my_array2 = [0_f32, 1.0, 2.0];
        let my_vector1 = Vector::from(&my_array1);
        let my_vector2 = Vector::from(&my_array2);
        let my_vector3 = Vector::from(&my_array1);

        assert_eq!(
            my_vector1.sub(&my_vector3),
            Ok(Vector::from(&[0_f32, 0.0, 0.0, 0.0]))
        );
        assert_eq!(my_vector1.sub(&my_vector2), Err(VectorError::ShapeMismatch));
    }

    #[test]
    fn check_scaling() {
        let my_array1 = (0..=10).map(|x| x as f32).collect::<Vec<f32>>();
        let my_array2 = (0..=10).map(|x| (x * 2) as f32).collect::<Vec<f32>>();        
        let my_vector1 = Vector::from(&my_array1);
        let my_vector2 = Vector::from(&my_array2);

        assert_eq!(my_vector1.scale(2_f32), my_vector2);
    }
}
