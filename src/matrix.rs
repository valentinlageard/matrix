/// A vector designed for linear algebra operations
#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone)]
pub struct Matrix<T>
where
    T: Copy,
{
    pub scalars: Vec<T>,
    pub shape: (usize, usize), // (row, column)
}

impl<T> Matrix<T>
where
    T: Copy,
{
    /// Returns the shape of the vector
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Returns the size (total number of elements) of the vector
    pub fn size(&self) -> usize {
        self.shape.0 * self.shape.1
    }

    // // TODO: Check if this is the optimal way to give iterators over the rows
    // /// Returns an iterator on the matrix rows
    // pub fn iter(&self) -> slice::Chunks<T> {
    //     self.scalars.chunks(self.shape.0)
    // }

    // // TODO: Check if this is the optimal way to give iterators over the rows
    // /// Returns a mutable iterator on the matrix rows
    // pub fn iter_mut(&mut self) -> slice::ChunksMut<T> {
    //     self.scalars.chunks_mut(self.shape.0)
    // }

    pub fn from(values: Vec<T>, shape: (usize, usize)) -> Self {
        assert_eq!(
            values.len(),
            shape.0 * shape.1,
            "Values with length {:?} don't match shape {:?}",
            values.len(),
            shape
        );
        Matrix {
            scalars: values,
            shape,
        }
    }

    pub fn from_fn<F>(shape: (usize, usize), mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        let mut values = Vec::with_capacity(shape.0 * shape.1);
        for i in 0..shape.0 {
            for j in 0..shape.1 {
                values.push(f(i, j));
            }
        }
        Matrix {
            scalars: values,
            shape,
        }
    }
}

// ====================================FROM CONSTRUCTION============================================

// impl<T> From<&mut [T]> for Vector<T>
// where
//     T: Copy,
// {
//     fn from(value: &mut [T]) -> Self {
//         Vector {
//             scalars: value.to_vec(),
//             shape: (value.len(),),
//         }
//     }
// }

// impl<T, const N: usize, const M: usize> From<[[T; N]; M]> for Vector<T>
// where
//     T: Copy,
// {
//     fn from(value: [T; N]) -> Self {
//         Vector {
//             scalars: value.to_vec(),
//             shape: (value.len(),),
//         }
//     }
// }

// ==============================================TESTS==============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_shape_and_size() {
        let my_matrix = Matrix::from(vec![0_f32, 1., 2., 3.], (2, 2));
        assert_eq!(my_matrix.shape(), (2, 2));
        assert_eq!(my_matrix.size(), 4);
    }

    #[test]
    fn check_construction() {
        let my_matrix = Matrix::from(vec![0, 1, 2, 3, 4, 5], (2, 3));
        let my_matrix2 = Matrix::from_fn((3, 2), |i, j| (i + j) as i32);
        assert_eq!(my_matrix.scalars, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(my_matrix.shape(), (2, 3));
        assert_eq!(my_matrix2.scalars, vec![0, 1, 1, 2, 2, 3]);
        assert_eq!(my_matrix2.shape(), (3, 2));
        // TODO: add more tests with indexing !
    }

    #[test]
    #[should_panic]
    fn check_construction_error() {
        let _my_matrix = Matrix::from(vec![0, 1, 2, 3, 4], (2,2));
    }

    
}
