#include <catch.hpp>
#include <numeric>

#include "level0.hpp"
#include <Eigen/Dense>
#define N_NODES_EVEN 16
#define N_NODES_ODD 15

/**
 * @brief Free particle wave packet in physical space
 *
 * @param a  Width of the wave packet. 
 * @param m  Mass of the particle
 * @param x0 Centroid of the wave packet
 * @param x  Coordinate of the particle.
 * @param t  Time
 *
 * @return Wavepacket amplitude as a function of the position 
 */
std::complex<double> psi_x1(double a, double m, double x0, double x, double t) {
  const int n_dim = 1;
  const std::complex<double> I = std::complex<double>(0.0, 1.0);
  std::complex<double> nrm_cte = std::pow(a / (a + I * t / m), n_dim * 0.5);
  std::complex<double> arg = -(x - x0) * (x - x0) / 2.0 / (a + I * t / m);
  return nrm_cte * std::exp(arg);
}

/**
 * @brief Free particle wave packet in Fourier space 
 *
 * @param a Width of the wave packet.
 * @param m  Mass of the particle
 * @param k0 Centroid of the wave packet
 * @param k  Momentum of the particle
 * @param t  Time
 *
 * @return Wave packet amplitude as a function of the momentum.
 */
std::complex<double> psi_k1(double a, double m, double k0, double k, double t) {
  const int n_dim = 1;
  const std::complex<double> I = std::complex<double>(0.0, 1.0);
  std::complex<double> nrm_cte = std::pow(2.0 * M_PI * a, n_dim * 0.5);
  std::complex<double> arg = -(a + I * t / m) * (k - k0) * (k - k0) / 2.0;
  return nrm_cte * std::exp(arg);
}

TEST_CASE("Addition and subtraction") {
  // Rank 1 tests
  std::shared_ptr<Eigen::VectorXcd> vecIn(new Eigen::VectorXcd(N_NODES_EVEN));
  std::shared_ptr<Eigen::VectorXcd> vecOut(new Eigen::VectorXcd(N_NODES_EVEN));

  vecIn->setRandom();
  vecOut = PROCTOR::inv_fft1(
      PROCTOR::inv_fftshift1(PROCTOR::fftshift1(PROCTOR::fft1(vecIn))));
  REQUIRE(vecIn->isApprox(*vecOut));
  // parameters for the wave packet uused in the tests
  double a = 1., m = 1., x0 = 0., k0=0., t = 0.;
  // building x an k grids with even spacing
  // step-size in x
  double dx = 0.4;
  int N = N_NODES_EVEN;
  std::vector<double> x(N_NODES_EVEN);
  // populating x with valuues between --N_NODES_EVEN / 2 and N_NODES_EVEN / 2-1
  std::iota(x.begin(), x.end(), (double)(-N_NODES_EVEN / 2));
  // x scaled by dx
  std::transform(x.begin(), x.end(), x.begin(),
      [N, dx, x0](double i) { return i * dx - x0; });
  // size of the simulation grid
  double L = x[15] - x[0];
  // step-size in k
  double dk = 2.0*M_PI/L;
  std::vector<double> k(N_NODES_EVEN);
  // populating x with valuues between --N_NODES_EVEN / 2 and N_NODES_EVEN / 2-1
  std::iota(k.begin(), k.end(), (double)(-N_NODES_EVEN / 2));
  // k scaled by dk
  std::transform(k.begin(), k.end(), k.begin(),
      [N, dk, k0](double i) { return i * dk - k0; });
  // evaluating psi(x) to be transformed as psi(k)=FFT(psi(x)) 
  for (int i = 0; i < N_NODES_EVEN; ++i)
    vecIn->data()[i] = psi_x1(a, m, x0, x[i], t);
  // psi(x) as computed using MATLAB's fft to compare against.
  std::complex<double> results_sw_x_v[] = {
    std::complex<double>(6.256708184272224, 0.),  std::complex<double>(-3.879565191776933, 0.),
    std::complex<double>(0.903679989061602, 0.),  std::complex<double>(-0.088483753877274, 0.),
    std::complex<double>(-0.002516976014255, 0.), std::complex<double>(-0.004477653106418, 0.),
    std::complex<double>(-0.003877145084366, 0.), std::complex<double>(-0.003569309659352, 0),
    std::complex<double>(-0.003471737038137, 0),  std::complex<double>(-0.003569309659352, 0.),
    std::complex<double>(-0.003877145084366, 0.), std::complex<double>(-0.004477653106418, 0.),
    std::complex<double>(-0.002516976014255, 0.), std::complex<double>(-0.088483753877274, 0.),
    std::complex<double>(0.903679989061602, 0.),  std::complex<double>(-3.879565191776933, 0.)};
  Eigen::Map<Eigen::VectorXcd> results_sw_x(results_sw_x_v, N_NODES_EVEN);
  vecIn = PROCTOR::fft1(vecIn);
  // checking correctness
  REQUIRE(vecIn->isApprox( results_sw_x ));
  //  evaluating psi(k) to be transformed as psi(x)=FFT^{-1}(psi(k))
  for (int i = 0; i < N_NODES_EVEN; ++i)
    vecIn->data()[i] = psi_k1(a, m, k0, k[i], t);
  // psi(x) as computed using MATLAB's ifft to compare against.
  std::complex<double> results_sw_k_v[] = {
    std::complex<double>(0.375000011422485, 0.), std::complex<double>(-0.349538485703525, 0.),
    std::complex<double>(0.283065238864099, 0.), std::complex<double>(-0.199163587188170, 0.),
    std::complex<double>(0.121759699746132, 0.), std::complex<double>(-0.064733815822271, 0.),
    std::complex<double>(0.030166250634437, 0.), std::complex<double>(-0.013221179943784, 0),
    std::complex<double>(0.008331747403682, 0.), std::complex<double>(-0.013221179943784, 0.),
    std::complex<double>(0.030166250634437, 0.), std::complex<double>(-0.064733815822271, 0.),
    std::complex<double>(0.121759699746132, 0.), std::complex<double>(-0.199163587188170, 0.),
    std::complex<double>(0.283065238864099, 0.), std::complex<double>(-0.349538485703525, 0.)};
  Eigen::Map<Eigen::VectorXcd> results_sw_k(results_sw_k_v, N_NODES_EVEN);
  vecIn = PROCTOR::inv_fft1(vecIn);
  // checking correctness
  REQUIRE(vecIn->isApprox( results_sw_k ));


  // Rank 2 tests
  std::shared_ptr<Eigen::MatrixXcd> matIn(
      new Eigen::MatrixXcd(N_NODES_EVEN, N_NODES_EVEN));
  std::shared_ptr<Eigen::MatrixXcd> matOut(
      new Eigen::MatrixXcd(N_NODES_EVEN, N_NODES_EVEN));

  matIn->setRandom();
  matOut = PROCTOR::inv_fft2(
      PROCTOR::inv_fftshift2(PROCTOR::fftshift2(PROCTOR::fft2(matIn))));
  REQUIRE(matIn->isApprox(*matOut));
}
