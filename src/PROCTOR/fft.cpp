#include <memory>
#include <iostream>
#include <unsupported/Eigen/FFT>
#include <unsupported/Eigen/CXX11/Tensor>
#include<Eigen/Dense>

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

int main(){
    size_t n_nodes = 16;
   /* 
    // 1D test
    std::shared_ptr<Eigen::VectorXcd> psi_1D( new Eigen::VectorXcd(n_nodes));
    psi_1D->setRandom();
    std::cout<<"psi\n"<<*psi_1D<<std::endl;
    psi_1D = fft1(psi_1D);
    psi_1D = inv_fft1(psi_1D);
    std::cout<<"psi\n"<<*psi_1D<<std::endl;
    */

    // 2D test
    std::shared_ptr<Eigen::MatrixXcd> psi_2D( new Eigen::MatrixXcd(n_nodes, n_nodes));
    psi_2D->setRandom();
    std::cout<<"psi\n"<<*psi_2D<<std::endl;
    psi_2D = fft2(psi_2D);
    psi_2D = inv_fft2(psi_2D);
    std::cout<<"psi\n"<<*psi_2D<<std::endl;


    return 0;
}
