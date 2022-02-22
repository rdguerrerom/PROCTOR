#include <memory>
#include <algorithm>
#include <iostream>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>

namespace PROCTOR {

    // => Rank 1 transforms <+ //

    inline std::shared_ptr< Eigen::VectorXcd > fft1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        // size of the task 
        const int nCols = vecIn->cols();
        // FFT task
        Eigen::FFT< double > fft;
        // Storage for the output 
        std::shared_ptr< Eigen::VectorXcd > vecOut = std::make_shared< Eigen::VectorXcd>(nCols);
        // temporary storage fot the task
        Eigen::VectorXcd tmpOut(nCols);
        // task execution
        fft.fwd(tmpOut, *vecIn);
        // asignment of the task results to the output
        *vecOut = tmpOut;
        // returning the outptut
        return vecOut;
    }

    inline std::shared_ptr< Eigen::VectorXcd > inv_fft1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        // size of the task
        const int nCols = vecIn->cols();
        // FFT task
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::VectorXcd > vecOut = std::make_shared< Eigen::VectorXcd>(nCols);
        // temporary storage fot the task
        Eigen::VectorXcd tmpOut(nCols);
        // task execution
        fft.inv(tmpOut, *vecIn);
        // asignment of the task results to the output
        *vecOut = tmpOut;
        // returning the outptut
        return vecOut;
    }

    inline void fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        const int nCols = vecIn->cols();
        // thow an assertiong of nCols is odd!!
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
    }

    inline void inv_fftshift1(std::shared_ptr< Eigen::VectorXcd > vecIn)
    {
        const int nCols = vecIn->cols();
        // thow an assertiong of nCols is odd!!
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
    }


    // => Rank 2 transforms <+ //

    inline std::shared_ptr< Eigen::MatrixXcd > fft2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // FFT task per dimension
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::MatrixXcd > matOut = std::make_shared< Eigen::MatrixXcd > (nRows, nCols);
        // temporary storage fot the task
        Eigen::VectorXcd tmpOut(nCols);
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // task execution for the k-th row
            fft.fwd(tmpOut, matIn->row(k));
            // asignment of the task results to the output
            matOut->row(k) = tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut.resize(nRows);
        // iterating through the columns
        for (int k = 0; k < matOut->cols(); ++k) {
            // task execution for the k-th column 
            fft.fwd(tmpOut, matOut->col(k));
            // asignment of the task results to the output
            matOut->col(k) = tmpOut;
        }
        // returning the outptut
        return matOut;
    }

    void fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // temporary storage fot the task
        std::shared_ptr< Eigen::VectorXcd > tmpOut(new Eigen::VectorXcd(nCols));
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->row(k);
            // shift execution for the k-th row
            fftshift1(tmpOut);
            // copying the shifted content back to thek-th row
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
            // copying the shifted content back to thek-th row
            matIn->col(k)=*tmpOut;
        }
    }


    inline std::shared_ptr< Eigen::MatrixXcd > inv_fft2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // FFT task per dimension
        Eigen::FFT< double > fft;
        // Storage for the output
        std::shared_ptr< Eigen::MatrixXcd > matOut = std::make_shared< Eigen::MatrixXcd > (nRows, nCols);
        // temporary storage fot the task
        Eigen::VectorXcd tmpOut(nCols);
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // task execution for the k-th row
            fft.inv(tmpOut, matIn->row(k));
            // asignment of the task results to the output
            matOut->row(k) = tmpOut;
        }
        // in case -f non-squared grids.
        tmpOut.resize(nRows);
        // iterating through the columns
        for (int k = 0; k < matOut->cols(); ++k) {
            // task execution for the k-th column
            fft.inv(tmpOut, matOut->col(k));
            // asignment of the task results to the output
            matOut->col(k) = tmpOut;
        }
        // returning the outptut
        return matOut;
    }

    void inv_fftshift2(std::shared_ptr< Eigen::MatrixXcd > matIn)
    {
        // size of the task
        const int nRows = matIn->rows();
        const int nCols = matIn->cols();
        // temporary storage fot the task
        std::shared_ptr< Eigen::VectorXcd > tmpOut(new Eigen::VectorXcd(nCols));
        // iterating through the rows
        for (int k = 0; k < nRows; ++k) {
            // moving the row to the temporary storage
            *tmpOut = matIn->row(k);
            // shift execution for the k-th row
            inv_fftshift1(tmpOut);
            // copying the shifted content back to thek-th row
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
            // copying the shifted content back to thek-th row
            matIn->col(k)=*tmpOut;
        }
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
    PROCTOR::fftshift2(psi_2D);
    PROCTOR::inv_fftshift2(psi_2D);
    psi_2D = PROCTOR::inv_fft2(psi_2D);
    std::cout<<"psi\n"<<*psi_2D<<std::endl;
      
    return 0;
}
