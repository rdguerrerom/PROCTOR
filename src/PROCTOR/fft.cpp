/**
 * @file fft.cpp
 * @brief Collection of routines that are necessary for implementing the split-step propagation method.
 * routines accept and return std::shared_ptr to allow for chaining transformations of the data.
 * @author Ruben Guerrero
 * @version 0.0.1
 * @date 2022-02-23
 */

#include <stdexcept>
#include <memory>
#include <algorithm>
#include <iostream>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>


namespace PROCTOR {

    // => Rank 1 transforms <+ //

    /**
     * @brief Wrapper for the 1D FFT routine provided by Eigen. 
     *
     * @param vecIn Input 1D signal
     *
     * @return F(vecIn), the Fourier transform of vecIn.
     */
    inline std::shared_ptr< Eigen::VectorXcd > fft1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        // size of the task 
        const int nCols = vecIn->cols();
        // FFT task
        Eigen::FFT< double > fft;
        // Storage for the output 
        std::shared_ptr< Eigen::VectorXcd > vecOut = std::make_shared< Eigen::VectorXcd>(nCols);
        // temporary storage for the task
        Eigen::VectorXcd tmpOut(nCols);
        // task execution
        fft.fwd(tmpOut, *vecIn);
        // assignment of the task results to the output
        *vecOut = tmpOut;
        // returning the output
        return vecOut;
    }

    /**
     * @brief Wrapper for the 1D FFT routine provided by Eigen.
     *
     * @param vecIn Input 1D signal
     *
     * @return F^{-1}(vecIn), the inverse Fourier transform of vecIn.
     */
    inline std::shared_ptr< Eigen::VectorXcd > inv_fft1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        // size of the task
        const int nCols = vecIn->cols();
        // FFT task
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::VectorXcd > vecOut = std::make_shared< Eigen::VectorXcd>(nCols);
        // temporary storage for the task
        Eigen::VectorXcd tmpOut(nCols);
        // task execution
        fft.inv(tmpOut, *vecIn);
        // assignment of the task results to the output
        *vecOut = tmpOut;
        // returning the output
        return vecOut;
    }

    /**
     * @brief  Rearranges a Fourier transform, vecIn, by shifting its zero-frequency component 
     * to the center of the vector.
     *
     * @param vecIn Fourier transform of the signal of interest.
     *
     * @return vecIn with its zero-frequency component shifted to the center of the vector.
     */
    inline std::shared_ptr< Eigen::VectorXcd >  fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        const int nCols = vecIn->cols();
        int k = 0;
        int c = (int) floor((float)nCols/2);
        // For odd and for even numbers of element use different algorithm
        if (nCols % 2 == 0)
        {// fast
            for (k = 0; k < c; k++)
                std::swap( vecIn->data()[k], vecIn->data()[k+c]);
        }
        else
        {// slow
            std::complex<double> tmp = vecIn->data()[0];
            for (k = 0; k < c; k++)
            {
                vecIn->data()[k] = vecIn->data()[c + k + 1];
                vecIn->data()[c + k + 1] = vecIn->data()[k + 1];
            }
            vecIn->data()[c] = tmp;
        }
        return vecIn;
    }

    /**
     * @brief Rearranges a zero-frequency-shifted Fourier transform, vecIn, back to the 
     * original Fourier transform output.
     *
     * @param vecIn Zero-frequency-shifted Fourier transform.
     *
     * @return signal ordered as the original Fourier transform output.
     */
    inline std::shared_ptr< Eigen::VectorXcd > inv_fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        const int nCols = vecIn->cols();
        int k = 0;
        int c = (int) floor((float)nCols/2);
        // For odd and for even numbers of element use different algorithm
        if (nCols % 2 == 0)
        {// fast
            for ( k = 0; k < c; k++  )
                std::swap( vecIn->data()[k], vecIn->data()[k+c]);
        }
        else
        {// slow
            std::complex<double> tmp = vecIn->data()[nCols-1];
            for ( k = c-1; k >= 0; k--  )
            {
                vecIn->data()[ c + k + 1 ] = vecIn->data()[ k ];
                vecIn->data()[ k ] = vecIn->data()[ c + k ];
            }
            vecIn->data()[c] = tmp;
        }
        return vecIn;
    }


    // => Rank 2 transforms <+ //

    /**
     * @brief Returns the two-dimensional Fourier transform of a matrix.
     *
     * @param matIn Input 2D signal 
     *
     * @return F(matIn), the Fourier transform of matIn.
     */
    inline std::shared_ptr< Eigen::MatrixXcd > fft2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // FFT task per dimension
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::MatrixXcd > matOut = std::make_shared< Eigen::MatrixXcd > (nRows, nCols);
        // temporary storage for the task
        Eigen::VectorXcd tmpOut(nCols);
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // task execution for the k-th row
            fft.fwd(tmpOut, matIn->row(k));
            // assignment of the task results to the output
            matOut->row(k) = tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut.resize(nRows);
        // iterating through the columns
        for (int k = 0; k < matOut->cols(); ++k) {
            // task execution for the k-th column 
            fft.fwd(tmpOut, matOut->col(k));
            // assignment of the task results to the output
            matOut->col(k) = tmpOut;
        }
        // returning the output
        return matOut;
    }

    /**
     * @brief Returns the two-dimensional discrete inverse Fourier transform of a matrix. 
     *
     * @param matIn Input 2D signal
     *
     * @return F^{-1}(matIn), the inverse Fourier transform of matIn.  
     */
    inline std::shared_ptr< Eigen::MatrixXcd > inv_fft2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // FFT task per dimension
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::MatrixXcd > matOut = std::make_shared< Eigen::MatrixXcd > (nRows, nCols);
        // temporary storage for the task
        Eigen::VectorXcd tmpOut(nCols);
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // task execution for the k-th row
            fft.inv(tmpOut, matIn->row(k));
            // assignment of the task results to the output
            matOut->row(k) = tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut.resize(nRows);
        // iterating through the columns
        for (int k = 0; k < matOut->cols(); ++k) {
            // task execution for the k-th column
            fft.inv(tmpOut, matOut->col(k));
            // assignment of the task results to the output
            matOut->col(k) = tmpOut;
        }
        // returning the output
        return matOut;
    }

    /**
     * @brief Rearranges a Fourier transform, matIn, by shifting its zero-frequency component
     * to the center of the matrix.
     *
     * @param matIn  2D Fourier transform of the signal of interest.
     *
     * @return  matIn with its zero-frequency component shifted to the center of the matrix.
     */
    inline std::shared_ptr< Eigen::MatrixXcd >  fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // temporary storage for the task
        std::shared_ptr< Eigen::VectorXcd > tmpOut(new Eigen::VectorXcd(nCols));
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->row(k);
            // shift execution for the k-th row
            fftshift1(tmpOut);
            // copying the shifted content back to the k-th row
            matIn->row(k) = *tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut->resize(nRows);
        // iterating through the columns
        for (int k = 0; k < nCols; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->col(k);
            // shift execution for the k-th row
            fftshift1(tmpOut);
            // copying the shifted content back to the k-th row
            matIn->col(k)=*tmpOut;
        }
        return matIn;
    }


    /**
     * @brief Rearranges a zero-frequency-shifted 2D Fourier transform, matIn, back to the
     * original Fourier transform output.
     *
     * @param matIn Zero-frequency-shifted 2D Fourier transform.
     *
     * @return signal ordered as the original 2D Fourier transform output.
     */
    inline std::shared_ptr< Eigen::MatrixXcd > inv_fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // temporary storage for the task
        std::shared_ptr< Eigen::VectorXcd > tmpOut(new Eigen::VectorXcd(nCols));
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->row(k);
            // shift execution for the k-th row
            inv_fftshift1(tmpOut);
            // copying the shifted content back to the k-th row
            matIn->row(k) = *tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut->resize(nRows);
        // iterating through the columns
        for (int k = 0; k < nCols; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->col(k);
            // shift execution for the k-th row
            inv_fftshift1(tmpOut);
            // copying the shifted content back to the k-th row
            matIn->col(k)=*tmpOut;
        }
        return matIn;
    }

    inline Eigen::MatrixXcd divide(const Eigen::MatrixXcd& D_, const Eigen::MatrixXcd& N_)
    {
        return D_.partialPivLu().solve(N_);
    }



    /**
     * @brief Computes the matrix exponential exp(A) using the (p,p)-degree rational 
     * Pade approximation to the exponential function, heavily based on:
     *
     * Title: Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later
     * Authors: Cleve Moler and Charles Van Loan
     * DOI:https://epubs.siam.org/doi/10.1137/S00361445024180
     * Further references:
     * [1] Higham, N. J., “The Scaling and Squaring Method for the Matrix Exponential Revisited,” SIAM J. Matrix Anal. Appl., 26(4) (2005), pp. 1179–1193.
     * [2] Al-Mohy, A. H. and N. J. Higham, “A new scaling and squaring algorithm for the matrix exponential,” SIAM J. Matrix Anal. Appl., 31(3) (2009), pp. 970–989.
     * [3] Golub, G. H. and C. F. Van Loan, Matrix Computation, p. 384, Johns Hopkins University Press, 1983.
     * [4] Moler, C. B. and C. F. Van Loan, “Nineteen Dubious Ways to Compute the Exponential of a Matrix,” SIAM Review 20, 1978, pp. 801–836. Reprinted and updated as “Nineteen Dubious Ways to Compute the Exponential of a Matrix, Twenty-Five Years Later,” SIAM Review 45, 2003, pp. 3–49.
     *
     * The main purpose of this routine is computing the short-time propagators like e^{i*V*dt}.
     *
     * @param iH Complex matrix i*V*dt.   
     * @param p Degree of the rational Pade approximation to the exponential function, b default p=10.
     *
     * @return  short-time propagators like e^{i*V*dt}.
     */
    inline std::shared_ptr< Eigen::MatrixXcd > pade_exp( std::shared_ptr< Eigen::MatrixXcd > iH, int p=10 )
    {
        if(iH->rows() == iH->cols())
            throw std::runtime_error("iH should be squared");

        int n = iH->rows();
        // Pade coefficients
        Eigen::VectorXcd c(p+1);
        c(0) = std::complex<double>( 1.0 , 0.0 );
        for (int k = 0; k < p; k++)
            c(k+1) = c(k)*std::complex<double>((p+1.-(k+1.))/((k+1.)*(2.*p+1.-(k+1.))), 0. );

        // Scaling
        Eigen::MatrixXcd iHs; // scaled
        std::complex<double>  s = iH->lpNorm<Eigen::Infinity>();
        // in the future we should consider switching to compute V^T*e^(\lamnda)*V for cases where it result faster or more accurate.
        if (s.real() > 0.5) {
            double _s = std::max(0., std::trunc(std::log(s.real())/std::log(2.))+2.);
            iHs = std::pow(2.,-_s)*(*iH);
        }
        else{
            iHs = *iH;
        }

        // Horner evaluation of the irreducible fraction (see ref. above)

        Eigen::MatrixXcd I_(n,n);
        I_.setIdentity();
        Eigen::MatrixXcd iH2 = iHs*iHs;
        Eigen::MatrixXcd Q = c(p)*I_;
        Eigen::MatrixXcd P = c(p-1)*I_;
        std::shared_ptr<Eigen::MatrixXcd> E( new Eigen::MatrixXcd(n, n));
        int odd = 1;
        for (int k = p-1; k >= 0; k--) {
            if (odd == 1){
                Q = Q*iH2 + c(k)*I_;
            }else{
                P = P*iH2 + c(k)*I_;
            }
            odd = 1-odd;
        }
        if (odd == 1) {
            Q = Q*iHs;
            Q = Q - P;
            *E = -(I_ + std::complex<double>(2.0, 0.0)*divide(Q,P));
        }
        else {
            P = P*iHs;
            Q = Q - P;
            *E = I_ + std::complex<double>(2.0, 0.0)*divide(Q,P);
        }

        // Squaring
        for (int k = 0; k < s.real(); k++)
            *E = (*E)*(*E);

        return E;

    }
}
int main(){

    size_t n_nodes = 16;
    /*     
    // 1D test
    std::shared_ptr<Eigen::VectorXcd> psi_1D( new Eigen::VectorXcd(n_nodes));
    psi_1D->setRandom();
    std::cout<<"psi\n"<<*psi_1D<<std::endl;
    psi_1D = PROCTOR::fft1(psi_1D);
    PROCTOR::fftshift1(psi_1D);
    PROCTOR::inv_fftshift1(psi_1D);
    psi_1D = PROCTOR::inv_fft1(psi_1D);
    std::cout<<"psi\n"<<*psi_1D<<std::endl;
    */

    // 2D test
    std::shared_ptr<Eigen::MatrixXcd> psi_2D( new Eigen::MatrixXcd(n_nodes, n_nodes));
    psi_2D->setRandom();
    std::cout<<"psi\n"<<*psi_2D<<std::endl;
    psi_2D = PROCTOR::fft2(psi_2D);
    psi_2D =PROCTOR::fftshift2(psi_2D);
    psi_2D =PROCTOR::inv_fftshift2(psi_2D);
    psi_2D = PROCTOR::inv_fft2(psi_2D);
    std::cout<<"psi\n"<<*psi_2D<<std::endl;
    psi_2D = PROCTOR::inv_fft2(PROCTOR::inv_fftshift2(PROCTOR::fftshift2(PROCTOR::fft2(psi_2D))));  
    std::cout<<"psi\n"<<*psi_2D<<std::endl;
    return 0;
}
