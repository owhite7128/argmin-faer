// Copyright 2018-2024 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(any(feature = "ndarray_latest", feature = "ndarray_v0_15", feature = "ndarray_v0_14", feature = "ndarray_v0_13"))]
pub mod inv { 
    use crate::ArgminInv;
    use crate::Error;
    use ndarray::Array2;
    use num_complex::Complex;
    use ndarray_linalg::Inverse;
    macro_rules! make_inv {
    ($t:ty) => {
        impl ArgminInv<Array2<$t>> for Array2<$t>
        where
            Array2<$t>: Inverse,
        {
            #[inline]
            fn inv(&self) -> Result<Array2<$t>, Error> {
                Ok(<Self as Inverse>::inv(&self)?)
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
make_inv!(f32);
make_inv!(f64);
make_inv!(Complex<f32>);
make_inv!(Complex<f64>); 

}

#[cfg(all(not(any(feature = "ndarray_latest", feature = "ndarray_v0_15", feature = "ndarray_v0_14", feature = "ndarray_v0_13")), any(feature = "ndarray_latest-faer", feature = "ndarray_v0_15-faer", feature = "ndarray_v0_14-faer", feature = "ndarray_v0_13-faer")))]
pub mod inv {
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
}

pub use inv::*;

// All code that does not depend on a linked ndarray-linalg backend can still be tested as normal.
// To avoid dublicating tests and to allow convenient testing of functionality that does not need ndarray-linalg the tests are still included here.
// The tests expect the name for the crate containing the tested functions to be argmin_math
#[cfg(test)]
use crate as argmin_math;
include!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/ndarray-tests-src/inv.rs"
));