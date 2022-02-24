#pragma once

#include<Eigen/Dense>
#include <memory>

#ifdef __cplusplus
extern "C" {
#endif
    namespace PROCTOR {

        /**
         * @brief  1D FFT routine. 
         *
         * @param vecIn Input 1D signal
         *
         * @return F(vecIn), the Fourier transform of vecIn.
         */
        inline std::shared_ptr< Eigen::VectorXcd > fft1(std::shared_ptr< Eigen::VectorXcd > vecIn);

        /**
         * @brief 1D FFT routine.
         *
         * @param vecIn Input 1D signal
         *
         * @return F^{-1}(vecIn), the inverse Fourier transform of vecIn.
         */
        inline std::shared_ptr< Eigen::VectorXcd > inv_fft1(std::shared_ptr< Eigen::VectorXcd > vecIn);

        /**
         * @brief  Rearranges a Fourier transform, vecIn, by shifting its zero-frequency component 
         * to the center of the vector.
         *
         * @param vecIn Fourier transform of the signal of interest.
         *
         * @return vecIn with its zero-frequency component shifted to the center of the vector.
         */
        inline std::shared_ptr< Eigen::VectorXcd >  fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn);

        /**
         * @brief Rearranges a zero-frequency-shifted Fourier transform, vecIn, back to the 
         * original Fourier transform output.
         *
         * @param vecIn Zero-frequency-shifted Fourier transform.
         *
         * @return signal ordered as the original Fourier transform output.
         */
        inline std::shared_ptr< Eigen::VectorXcd > inv_fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn);

        /**
         * @brief Returns the two-dimensional Fourier transform of a matrix.
         *
         * @param matIn Input 2D signal 
         *
         * @return F(matIn), the Fourier transform of matIn.
         */
        inline std::shared_ptr< Eigen::MatrixXcd > fft2(std::shared_ptr< Eigen::MatrixXcd > matIn);

        /**
         * @brief Returns the two-dimensional discrete inverse Fourier transform of a matrix. 
         *
         * @param matIn Input 2D signal
         *
         * @return F^{-1}(matIn), the inverse Fourier transform of matIn.  
         */
        inline std::shared_ptr< Eigen::MatrixXcd > inv_fft2(std::shared_ptr< Eigen::MatrixXcd > matIn);

        /**
         * @brief Rearranges a Fourier transform, matIn, by shifting its zero-frequency component
         * to the center of the matrix.
         *
         * @param matIn  2D Fourier transform of the signal of interest.
         *
         * @return  matIn with its zero-frequency component shifted to the center of the matrix.
         */
        inline std::shared_ptr< Eigen::MatrixXcd >  fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn);

        /**
         * @brief Rearranges a zero-frequency-shifted 2D Fourier transform, matIn, back to the
         * original Fourier transform output.
         *
         * @param matIn Zero-frequency-shifted 2D Fourier transform.
         *
         * @return signal ordered as the original 2D Fourier transform output.
         */
        inline std::shared_ptr< Eigen::MatrixXcd > inv_fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn);

        /**
         * @brief Computes the matrix exponential exp(A) using the (p,p)-degree rational 
         * Pade approximation to the exponential function, heavily based on:
         *
         * The main purpose of this routine is computing the short-time propagators like e^{i*V*dt}.
         *
         * @param iH Complex matrix i*V*dt.   
         * @param p Degree of the rational Pade approximation to the exponential function, b default p=10.
         *
         * @return  short-time propagators like e^{i*V*dt}.
         */
        inline std::shared_ptr< Eigen::MatrixXcd > pade_exp( std::shared_ptr< Eigen::MatrixXcd > iH, int p );

    } // end of namespace



#ifdef __cplusplus
}  // extern "C"
#endif
