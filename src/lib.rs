// #![allow(dead_code)]

use std::ops::{Add, Mul, Sub};

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

/// Generic linear combination
pub fn linear_combination<T>(vectors: &[Vector<T>], coefficients: &[T]) -> Vector<T>
where
    T: Copy + Default + Add + Mul + std::fmt::Debug,
    Vec<T>: FromIterator<<T as Mul>::Output> + FromIterator<<T as Add>::Output>,
{
    assert_eq!(
        vectors.len(),
        coefficients.len(),
        "Length mismatch between {:?} vectors and {:?} coefficients",
        vectors.len(),
        coefficients.len()
    );
    let vector_shape = vectors[0].shape();
    for vector in vectors {
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

    // Compute the scaled vectors
    let scaled_vectors: Vec<Vector<T>> = coefficients
        .iter()
        .zip(vectors)
        .map(|(&coefficient, vector)| vector * coefficient)
        .collect();

    println!("scaled_vectors: {:?}", scaled_vectors);

    scaled_vectors
        .iter()
        .fold(result, |acc, scaled_vector| &acc + scaled_vector)
}


// Potential TODO: Make the mul operation commutative. Seems to require lots of boilerplate code (like below) or advanced macros
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
        let my_vector1 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
        let my_vector2 = Vector::from(&[0_f32, 1.0, 2.0, 3.0]);
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
    #[ignore = "takes time"]
    fn check_big_addition() {
        let my_array = (0..=100000000).map(|x| x as f32).collect::<Vec<f32>>();
        let my_vector1 = Vector::from(&my_array);
        let my_vector2 = Vector::from(&my_array);

        let _my_vector3 = my_vector1.add(&my_vector2);
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
        let v1 = Vector::from(&[1_f32, 2., 3.]);
        let v2 = Vector::from(&[0_f32, 10., -100.]);
        let result = linear_combination(&[v1, v2], &[10., -2.]);

        assert_eq!(result, Vector::from(&[10_f32, 0., 230.]));
    }
}
