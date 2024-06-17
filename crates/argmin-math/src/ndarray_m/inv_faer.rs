use crate::ArgminInv;
use crate::Error;
use ndarray::Array2;
use num_complex::Complex;
use {faer::{ Mat, prelude::SolverCore, SimpleEntity, Entity }, faer_ext::*};


macro_rules! make_inv_simple {
    ($t:ty) => {
        impl ArgminInv<Array2<$t>> for Array2<$t>
        {
            
            #[inline]
            fn inv(&self) -> Result<Array2<$t>, Error> {
                Ok(self.view()
                    .into_faer()
                    .partial_piv_lu()
                    .inverse()
                    .as_ref()
                    .into_ndarray()
                    .to_owned())
            }
        }

        // inverse for scalars (1d solvers)
        impl ArgminInv<$t> for $t {
            #[inline]
            fn inv(&self) -> Result<$t, Error> {
                Ok(1.0 / self)
            }
        }
    };
    
}

macro_rules! make_inv_complex {
    ($t:ty) => {
        impl ArgminInv<Array2<$t>> for Array2<$t>
        {
            
            #[inline]
            fn inv(&self) -> Result<Array2<$t>, Error> {
                Ok(self.view()
                    .into_faer_complex()
                    .partial_piv_lu()
                    .inverse()
                    .as_ref()
                    .into_ndarray_complex()
                    .to_owned())
            }
        }

        // inverse for scalars (1d solvers)
        impl ArgminInv<$t> for $t {
            #[inline]
            fn inv(&self) -> Result<$t, Error> {
                Ok(1.0 / self)
            }
        }
    };
}

make_inv_simple!(f32);
make_inv_simple!(f64);
make_inv_complex!(Complex<f32>);
make_inv_complex!(Complex<f64>);