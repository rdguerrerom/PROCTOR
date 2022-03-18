#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>
//#include <pybind11/eigen.h>
//#include <pybind11/stl.h>
#include "level0.hpp"
#include <map>
#include <omp.h>
#include <tuple>
#include <typeinfo>

template <typename T> std::string type_name();

#ifdef __cplusplus
extern "C" {
#endif
namespace PROCTOR {

class Simulation0D {
public:
  // Constructor
  Simulation0D() = default;
  // Destructor
  ~Simulation0D() {}
  // Setters
  // Getters
};

/**
 * @brief Container for non-adiabatic quantum dynamics simulations.
 * The electronic Hamiltonian will be organized by blocks as follows:
 *
 *          |singlets(s)>  |doublets(d)> |triplets(t)>
 * <singlets|  [s|s]           [s|d]         [s|t]
 *
 * <doublets|  [d|s]           [d|d]         [d|t]
 *
 * <triplets|  [t|s]           [t|d]         [t|t]
 *
 * The PES, DM and TDM  will go into the [s|s], [d|d] or [t|t] blocks.
 * The SOC will go in the off-diagonal blocks coupling electronic states
 * of different spin-multiplicity
 *
 * Plan is that all this information should be loaded using the setters.
 * Also the initial wave packet, its starting block and the state withing
 * the block shall be set trough setters.
 */
class Simulation1D {
public:
  // Constructor
  Simulation1D() = default;
  /**
   * @brief Constructor for an empty simulation. The initial state shall be
   loaded using a
   * different mechanism.

   *
   * @param[in] x             Simulation grid for the coordinate provided by the
   user
   * @param[in] dx            Grid spacing
   * @param[in] dt            Time-step for the simulation
   * @param[in] n_grid_points Number of grid points on the simulation grid
   * @param[in] n_singlets    Number of singlet states
   * @param[in] n_doublets    Number of doublet states
   * @param[in] n_triplets    Number of triplet states
   */
  Simulation1D(const Eigen::VectorXd &x, double dx, double dt,
               int n_grid_points, int n_singlets, int n_doublets,
               int n_triplets, int suzuki_method_order = 4)
      : _x(x), _dx(dx), _dt(dt), _n_grid_points(n_grid_points),
        _n_singlets(n_singlets), _n_doublets(n_doublets),
        _n_triplets(n_triplets) {
    // checking that the information makes sense

    // Default behavior is that  the information for the initial wave packet is
    // loaded in position  representation
    _position_representation = true;
    _momentum_representation = false;
    // set up the order of the approximant
    set_suzuki_order(suzuki_method_order);
    _pn = _suzuki_fractal_decomposition_coeffs(_suzuki_fractal_order);
    /*double count=0;
    for(auto i:_pn)
    {
      std::cout << i << std::endl;
      count +=i;
    }
    std::cout << "Suzuki accuum: " << count <<std::endl;*/

    // resizing containers
    _grid_idx.resize(_n_grid_points);
    std::iota(_grid_idx.begin(), _grid_idx.end(), 0);
    // total number of PES
    int total_electronic_states = _n_singlets + _n_doublets + _n_triplets;

    // populating
    double L = _x(_n_grid_points - 1) - _x(0);
    // step-size in k
    _dk = 2.0 * M_PI / L;
    // discretization of the corresponding momentum
    std::vector<double> _k;
    // filling the momentum grid
    _k.resize(_n_grid_points);
    // populating _k with values between -_n_grid_points / 2 and _n_grid_points
    // / 2-1
    std::iota(_k.begin(), _k.end(), (double)(0));
    // _k scaled by _dk
    // fixing capture issues
    double dK = _dk;
    int N = _n_grid_points;
    // see Spectral Methods in MATLAB. Lloyd N. Trefethen, pag 24.
    std::transform(_k.begin(), _k.end(), _k.begin(), [N, dK](double i) {
      return i < N / 2 ? i * dK : (i - N) * dK;
    });
    // Filling the kinetic energy
    _exp_T.resize(_n_grid_points);
    for (int j = 0; j < _n_grid_points; ++j) {
      // 1/2 factor makes this ready for a TVT splitting.
      double theta = -_k[j] * _k[j] / (2.0) * (_dt / (2.0 * _hbar));
      _exp_T(j) = std::complex<double>(cos(theta), sin(theta));
    }
  }
  // Destructor
  ~Simulation1D() {}

  // Setters
  void set_suzuki_order(int n) {
    if (n % 2 == 0) {
      _suzuki_fractal_order = n / 2;
    } else {
      n += (n & 1);
      _suzuki_fractal_order = n / 2 + 1;
    }
  }

  /**
   * @brief Allows loading a wave packet to the state_idx-th PES of type
   * electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES within all the PES of  type
   * electronic_state_type.
   * @param electronic_state_type  Which is the spin quantum number of this PES
   * where this wave packet stands.
   * @param psi_0                  Wave packet evaluated on the simulation grid,
   * _x.
   */
  void set_wave_packet(int state_idx, std::string electronic_state_type,
                       Eigen::Ref<Eigen::VectorXcd> psi_0) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, Eigen::VectorXcd> &wp :
         _electronic_wave_packet[electronic_state_type]) {
      if (std::get<0>(wp) == state_idx) {
        std::get<1>(wp) = psi_0;
        done = true;
      }
    }
    // else add new information
    if (!done)
      _electronic_wave_packet[electronic_state_type].push_back(
          std::tuple<int, Eigen::VectorXcd>{state_idx, psi_0});
  }
  /**
   * @brief Allows loading the state_idx-th PES of type electronic_state_type.
   * Being s the  spin quantum number, the string electronic_state_type can be
   * any choice between "Singlet"(s=0), "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES within all the PES of  type
   * electronic_state_type.
   * @param electronic_state_type  Which is the spin quantum number of this PES
   * where this wave packet stands.
   * @param PES                    PES evaluated on the simulation grid, _x.
   */
  void set_PES(int state_idx, std::string electronic_state_type,
               Eigen::Ref<Eigen::VectorXd> PES) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, Eigen::VectorXd> &pes :
         _electronic_PES[electronic_state_type]) {
      if (std::get<0>(pes) == state_idx) {
        std::get<1>(pes) = PES;
        done = true;
      }
    }
    for (std::tuple<int, bool> &pesl :
         _electronic_PES_loaded[electronic_state_type]) {
      if (std::get<0>(pesl) == state_idx) {
        std::get<1>(pesl) = true;
        done = true;
      }
    }
    // else add new information
    if (!done) {
      _electronic_PES[electronic_state_type].push_back(
          std::tuple<int, Eigen::VectorXd>{state_idx, PES});
      _electronic_PES_loaded[electronic_state_type].push_back(
          std::tuple<int, bool>{state_idx, true});
    }
    // std::cout<<
    // std::get<0>(_electronic_PES[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<1>(_electronic_PES[electronic_state_type].back())<<std::endl;
  }
  /**
   * @brief Allows loading a DM affecting the  state_idx-th PES of type
   * electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES that this DM is affecting.
   * @param electronic_state_type  Which is the spin quantum number of the PES
   * that this DM is affecting.
   * @param PES                    PES evaluated on the simulation grid, _x.
   */
  void set_DM(int state_idx, std::string electronic_state_type,
              Eigen::Ref<Eigen::VectorXd> DM) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, Eigen::VectorXd> &dm :
         _electronic_DM[electronic_state_type]) {
      if (std::get<0>(dm) == state_idx) {
        std::get<1>(dm) = DM;
        done = true;
      }
    }
    for (std::tuple<int, bool> &dml :
         _electronic_DM_loaded[electronic_state_type]) {
      if (std::get<0>(dml) == state_idx) {
        std::get<1>(dml) = true;
        done = true;
      }
    }
    // else add new information
    if (!done) {
      _electronic_DM[electronic_state_type].push_back(
          std::tuple<int, Eigen::VectorXd>{state_idx, DM});
      _electronic_DM_loaded[electronic_state_type].push_back(
          std::tuple<int, bool>{state_idx, true});
    }
    // std::cout<<
    // std::get<0>(_electronic_DM[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<1>(_electronic_DM[electronic_state_type].back())<<std::endl;
  }
  /**
   * @brief Allows loading a NACME coupling the  state1_idx and state2_idx PES
   * of type electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx             Index of the first PES that this NACME is
   * affecting.
   * @param state2_idx             Index of the second PES that this NACME is
   * affecting.
   * @param electronic_state_type  Which is the spin quantum number of the two
   * PES that this term is affecting.
   * @param NACME                  NACME evaluated on the simulation grid, _x.
   */
  void set_NACME(int state1_idx, int state2_idx,
                 std::string electronic_state_type,
                 Eigen::Ref<Eigen::VectorXd> NACME) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, int, Eigen::VectorXd> &nacme :
         _electronic_NACME[electronic_state_type]) {
      if (std::get<0>(nacme) == state1_idx &&
          std::get<1>(nacme) == state2_idx) {
        std::get<2>(nacme) = NACME;
        done = true;
      }
    }
    for (std::tuple<int, int, bool> &nacmel :
         _electronic_NACME_loaded[electronic_state_type]) {
      if (std::get<0>(nacmel) == state1_idx &&
          std::get<1>(nacmel) == state2_idx) {
        std::get<2>(nacmel) = true;
        done = true;
      }
    }
    // else add new information
    if (!done) {
      _electronic_NACME[electronic_state_type].push_back(
          std::tuple<int, int, Eigen::VectorXd>{state1_idx, state2_idx, NACME});
      _electronic_NACME_loaded[electronic_state_type].push_back(
          std::tuple<int, int, bool>{state1_idx, state2_idx, true});
    }
    // std::cout<<
    // std::get<0>(_electronic_NACME[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<1>(_electronic_NACME[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<2>(_electronic_NACME[electronic_state_type].back())<<std::endl;
  }
  /**
   * @brief Allows loading a TDM coupling the  state1_idx and state2_idx PES of
   * type electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx             Index of the first PES that this TDM is
   * affecting.
   * @param state2_idx             Index of the second PES that this TDM is
   * affecting.
   * @param electronic_state_type  Which is the spin quantum number of the two
   * PES that this term is affecting.
   * @param NACME                  NACME evaluated on the simulation grid, _x.
   */
  void set_TDM(int state1_idx, int state2_idx,
               std::string electronic_state_type,
               Eigen::Ref<Eigen::VectorXd> TDM) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, int, Eigen::VectorXd> &tdm :
         _electronic_TDM[electronic_state_type]) {
      if (std::get<0>(tdm) == state1_idx && std::get<1>(tdm) == state2_idx) {
        std::get<2>(tdm) = TDM;
        done = true;
      }
    }
    for (std::tuple<int, int, bool> &tdml :
         _electronic_TDM_loaded[electronic_state_type]) {
      if (std::get<0>(tdml) == state1_idx && std::get<1>(tdml) == state2_idx) {
        std::get<2>(tdml) = true;
        done = true;
      }
    }
    // else add new information
    if (!done) {
      _electronic_TDM[electronic_state_type].push_back(
          std::tuple<int, int, Eigen::VectorXd>{state1_idx, state2_idx, TDM});
      _electronic_TDM_loaded[electronic_state_type].push_back(
          std::tuple<int, int, bool>{state1_idx, state2_idx, true});
    }
    // std::cout<<
    // std::get<0>(_electronic_TDM[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<1>(_electronic_TDM[electronic_state_type].back())<<std::endl;
    // std::cout<<
    // std::get<2>(_electronic_TDM[electronic_state_type].back())<<std::endl;
  }
  /**
   * @brief Allows loading a SOC coupling the  state1_idx of type
   * electronic_state1_type and state2_idx PES of type electronic_state2_type.
   * Being s the  spin quantum number, the string electronic_state_type can be
   * any choice between "Singlet"(s=0), "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx               Index of the first PES that this TDM is
   * affecting.
   * @param state2_idx               Index of the second PES that this TDM is
   * affecting.
   * @param electronic_state1_type   Type of electronic state of the first PES
   * that this term is coupling.
   * @param electronic_state2_type   Type of electronic state of the second PES
   * that this term is coupling.
   * @param SOC                      SOC evaluated on the simulation grid, _x.
   */
  void set_SOC(int state1_idx, int state2_idx,
               std::string electronic_state1_type,
               std::string electronic_state2_type,
               Eigen::Ref<Eigen::VectorXd> SOC) {
    bool done = false;
    // check if this key already exist, if so, rewrite it
    for (std::tuple<int, int, Eigen::VectorXd> &soc :
         _electronic_SOC[std::tuple<std::string, std::string>{
             electronic_state1_type, electronic_state2_type}]) {
      if (std::get<0>(soc) == state1_idx && std::get<1>(soc) == state2_idx) {
        std::get<2>(soc) = SOC;
        done = true;
      }
    }
    for (std::tuple<int, int, bool> &socl :
         _electronic_SOC_loaded[std::tuple<std::string, std::string>{
             electronic_state1_type, electronic_state2_type}]) {
      if (std::get<0>(socl) == state1_idx && std::get<1>(socl) == state2_idx) {
        std::get<2>(socl) = true;
        done = true;
      }
    }
    // else add new information
    if (!done) {
      _electronic_SOC[std::tuple<std::string, std::string>{
                          electronic_state1_type, electronic_state2_type}]
          .push_back(std::tuple<int, int, Eigen::VectorXd>{state1_idx,
                                                           state2_idx, SOC});
      _electronic_SOC_loaded[std::tuple<std::string, std::string>{
                                 electronic_state1_type,
                                 electronic_state2_type}]
          .push_back(std::tuple<int, int, bool>{state1_idx, state2_idx, true});
    }
    // std::cout << std::get<0>(
    //                  _electronic_SOC[std::tuple<std::string, std::string>{
    //                                      electronic_state1_type,
    //                                      electronic_state2_type}]
    //                      .back())
    //           << std::endl;
    // std::cout << std::get<1>(
    //                  _electronic_SOC[std::tuple<std::string, std::string>{
    //                                      electronic_state1_type,
    //                                      electronic_state2_type}]
    //                      .back())
    //           << std::endl;
    // std::cout << std::get<2>(
    //                  _electronic_SOC[std::tuple<std::string, std::string>{
    //                                      electronic_state1_type,
    //                                      electronic_state2_type}]
    //                      .back())
    //           << std::endl;
  }
  // Getters

  /**
   * @brief In the event of having a time-dependent perturbation interacting
   * with the wave packet the problem iis reduced to an ordinary decomposition
   * of exponential operators plus the time-dependent signal avaluated at
   * discrete times shifted forward. this implementation follows the references:
   *
   * [1] Suzuki, Masuo. "General decomposition theory of ordered exponentials."
   * Proceedings of the Japan Academy, Series B 69.7 (1993): 161-166.
   *
   * Idea is that a list of discrete times is returned to the driving routine in
   * python to evaluate the perturbation signal, the reslts of the evaluation is
   * passed back to this class using the routine set_perturbation_samples, then
   * this information is used by the routines take_step_forward or
   * take_step_backward that updates the simulation.
   *
   * @return Forward sampling points in time
   */
  std::vector<double> get_perturbation_sampling_forward() {
    std::vector<double> sampling_times;
    for (auto pn : _pn) {
      sampling_times.push_back(simulation_time + 1.5 * pn * _dt);
    }
    return sampling_times;
  }
  /**
   * @brief In the event of having a time-dependent perturbation interacting
   * with the wave packet the problem iis reduced to an ordinary decomposition
   * of exponential operators plus the time-dependent signal avaluated at
   * discrete times shifted forward. this implementation follows the references:
   *
   * [1] Suzuki, Masuo. "General decomposition theory of ordered exponentials."
   * Proceedings of the Japan Academy, Series B 69.7 (1993): 161-166.
   *
   * Idea is that a list of discrete times is returned to the driving routine in
   * python to evaluate the perturbation signal, the reslts of the evaluation is
   * passed back to this class using the routine set_perturbation_samples, then
   * this information is used by the routines take_step_forward or
   * take_step_backward that updates the simulation.
   *
   * @return Backward sampling points in time
   */
  std::vector<double> get_perturbation_sampling_backward() {
    std::vector<double> sampling_times;
    for (auto pn : _pn) {
      sampling_times.push_back(simulation_time - 1.5 * pn * _dt);
    }
    return sampling_times;
  }

  /**
   * @brief Set the pertrbation samples to be used by the take_step_forward or
   * take_step_backward methods.
   *
   * @param perturbation_samples  Discrete samples of the time-dependent signal
   * evaluuated at the sampling points provided either
   * get_perturbation_sampling_forward or get_perturbation_sampling_backward
   *                              routines.
   */
  void
  set_perturbation_signal_samples(std::vector<double> perturbation_samples) {
    _perturbation_signal_samples = perturbation_samples;
  }

  /**
   * @brief Check if the PES setup is correct
   *
   * @return True, if the setup is correct. False if not.
   */
  bool electronic_PES_complete() const {
    // Handler for the logic of  the blocking  of the Hamiltonian for different
    // simulation setups we are going from
    if (_n_singlets > 0 && _n_doublets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_PES_loaded.size() != _singlet_PES.size())
        return false;
      for (auto &&[st, ready] : _singlet_PES_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_PES_loaded.size() != _doublet_PES.size())
        return false;
      for (auto &&[st, ready] : _doublet_PES_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_PES_loaded.size() != _triplet_PES.size())
        return false;
      for (auto &&[st, ready] : _triplet_PES_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0 && _n_doublets > 0) {
      // checking the singlet block
      if (_singlet_PES_loaded.size() != _singlet_PES.size())
        return false;
      for (auto &&[st, ready] : _singlet_PES_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_PES_loaded.size() != _doublet_PES.size())
        return false;
      for (auto &&[st, ready] : _doublet_PES_loaded)
        if (ready != true)
          return false;

    } else if (_n_singlets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_PES_loaded.size() != _singlet_PES.size())
        return false;
      for (auto &&[st, ready] : _singlet_PES_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_PES_loaded.size() != _triplet_PES.size())
        return false;
      for (auto &&[st, ready] : _triplet_PES_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0 && _n_triplets > 0) {
      // checking the doublet block
      if (_doublet_PES_loaded.size() != _doublet_PES.size())
        return false;
      for (auto &&[st, ready] : _doublet_PES_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_PES_loaded.size() != _triplet_PES.size())
        return false;
      for (auto &&[st, ready] : _triplet_PES_loaded)
        if (ready != true)
          return false;

    } else if (_n_triplets > 0) {
      // checking the triplet block
      if (_triplet_PES_loaded.size() != _triplet_PES.size())
        return false;
      for (auto &&[st, ready] : _triplet_PES_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0) {
      // checking the doublet block
      if (_doublet_PES_loaded.size() != _doublet_PES.size())
        return false;
      for (auto &&[st, ready] : _doublet_PES_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0) {
      // checking the singlet block
      if (_singlet_PES_loaded.size() != _singlet_PES.size())
        return false;
      for (auto &&[st, ready] : _singlet_PES_loaded)
        if (ready != true)
          return false;
    } else {
      throw std::runtime_error("PES setup is not supported by this version.");
    }
  }
  /**
   * @brief Check if the DM setup is correct
   *
   * @return True, if the setup is correct. False if not.
   */
  bool electronic_DM_complete() const {
    // Handler for the logic of  the blocking  of the Hamiltonian for different
    // simulation setups we are going from
    if (_n_singlets > 0 && _n_doublets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_DM_loaded.size() != _singlet_DM.size())
        return false;
      for (auto &&[st, ready] : _singlet_DM_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_DM_loaded.size() != _doublet_DM.size())
        return false;
      for (auto &&[st, ready] : _doublet_DM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_DM_loaded.size() != _triplet_DM.size())
        return false;
      for (auto &&[st, ready] : _triplet_DM_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0 && _n_doublets > 0) {
      // checking the singlet block
      if (_singlet_DM_loaded.size() != _singlet_DM.size())
        return false;
      for (auto &&[st, ready] : _singlet_DM_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_DM_loaded.size() != _doublet_DM.size())
        return false;
      for (auto &&[st, ready] : _doublet_DM_loaded)
        if (ready != true)
          return false;

    } else if (_n_singlets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_DM_loaded.size() != _singlet_DM.size())
        return false;
      for (auto &&[st, ready] : _singlet_DM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_DM_loaded.size() != _triplet_DM.size())
        return false;
      for (auto &&[st, ready] : _triplet_DM_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0 && _n_triplets > 0) {
      // checking the doublet block
      if (_doublet_DM_loaded.size() != _doublet_DM.size())
        return false;
      for (auto &&[st, ready] : _doublet_DM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_DM_loaded.size() != _triplet_DM.size())
        return false;
      for (auto &&[st, ready] : _triplet_DM_loaded)
        if (ready != true)
          return false;

    } else if (_n_triplets > 0) {
      // checking the triplet block
      if (_triplet_DM_loaded.size() != _triplet_DM.size())
        return false;
      for (auto &&[st, ready] : _triplet_DM_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0) {
      // checking the doublet block
      if (_doublet_DM_loaded.size() != _doublet_DM.size())
        return false;
      for (auto &&[st, ready] : _doublet_DM_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0) {
      // checking the singlet block
      if (_singlet_DM_loaded.size() != _singlet_DM.size())
        return false;
      for (auto &&[st, ready] : _singlet_DM_loaded)
        if (ready != true)
          return false;
    } else {
      throw std::runtime_error("DM setup is not supported by this version.");
    }
  }
  /**
   * @brief Check if the DM setup is correct
   *
   * @return True, if the setup is correct. False if not.
   */
  bool electronic_TDM_complete() const {
    // Handler for the logic of  the blocking  of the Hamiltonian for different
    // simulation setups we are going from
    if (_n_singlets > 0 && _n_doublets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_TDM_loaded.size() != _singlet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_TDM_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_TDM_loaded.size() != _doublet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_TDM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_TDM_loaded.size() != _triplet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_TDM_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0 && _n_doublets > 0) {
      // checking the singlet block
      if (_singlet_TDM_loaded.size() != _singlet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_TDM_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_TDM_loaded.size() != _doublet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_TDM_loaded)
        if (ready != true)
          return false;

    } else if (_n_singlets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_TDM_loaded.size() != _singlet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_TDM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_TDM_loaded.size() != _triplet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_TDM_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0 && _n_triplets > 0) {
      // checking the doublet block
      if (_doublet_TDM_loaded.size() != _doublet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_TDM_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_TDM_loaded.size() != _triplet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_TDM_loaded)
        if (ready != true)
          return false;

    } else if (_n_triplets > 0) {
      // checking the triplet block
      if (_triplet_TDM_loaded.size() != _triplet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_TDM_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0) {
      // checking the doublet block
      if (_doublet_TDM_loaded.size() != _doublet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_TDM_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0) {
      // checking the singlet block
      if (_singlet_TDM_loaded.size() != _singlet_TDM.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_TDM_loaded)
        if (ready != true)
          return false;
    } else {
      throw std::runtime_error("TDM setup is not supported by this version.");
    }
  }
  /**
   * @brief Check if the NACME setup is correct
   *
   * @return True, if the setup is correct. False if not.
   */
  bool electronic_NACME_complete() const {
    // Handler for the logic of  the blocking  of the Hamiltonian for different
    // simulation setups we are going from
    if (_n_singlets > 0 && _n_doublets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_NACME_loaded.size() != _singlet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_NACME_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_NACME_loaded.size() != _doublet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_NACME_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_NACME_loaded.size() != _triplet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_NACME_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0 && _n_doublets > 0) {
      // checking the singlet block
      if (_singlet_NACME_loaded.size() != _singlet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_NACME_loaded)
        if (ready != true)
          return false;
      // checking the doublet block
      if (_doublet_NACME_loaded.size() != _doublet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_NACME_loaded)
        if (ready != true)
          return false;

    } else if (_n_singlets > 0 && _n_triplets > 0) {
      // checking the singlet block
      if (_singlet_NACME_loaded.size() != _singlet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_NACME_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_NACME_loaded.size() != _triplet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_NACME_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0 && _n_triplets > 0) {
      // checking the doublet block
      if (_doublet_NACME_loaded.size() != _doublet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_NACME_loaded)
        if (ready != true)
          return false;
      // checking the triplet block
      if (_triplet_NACME_loaded.size() != _triplet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_NACME_loaded)
        if (ready != true)
          return false;

    } else if (_n_triplets > 0) {
      // checking the triplet block
      if (_triplet_NACME_loaded.size() != _triplet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _triplet_NACME_loaded)
        if (ready != true)
          return false;

    } else if (_n_doublets > 0) {
      // checking the doublet block
      if (_doublet_NACME_loaded.size() != _doublet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _doublet_NACME_loaded)
        if (ready != true)
          return false;
    } else if (_n_singlets > 0) {
      // checking the singlet block
      if (_singlet_NACME_loaded.size() != _singlet_NACME.size())
        return false;
      for (auto &&[st1, st2, ready] : _singlet_NACME_loaded)
        if (ready != true)
          return false;
    } else {
      throw std::runtime_error("NACME setup is not supported by this version.");
    }
  }
  /**
   * @brief Allows getting a wave packet to the state_idx-th PES of type
   * electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES within all the PES of  type
   * electronic_state_type.
   * @param electronic_state_type  Which is the spin quantum number of this PES
   * where this wave packet stands.
   * @param psi_0                  Wave packet evaluated on the simulation grid,
   * _x.
   */
  Eigen::Ref<Eigen::VectorXcd>
  get_wave_packet(int state_idx, std::string electronic_state_type,
                  Eigen::Ref<Eigen::VectorXcd> psi_0) {
    // check if this key already exist, if so,  return the corresponding psi.
    for (std::tuple<int, Eigen::VectorXcd> &wp :
         _electronic_wave_packet[electronic_state_type]) {
      if (std::get<0>(wp) == state_idx)
        return std::get<1>(wp);
    }
  }
  /**
   * @brief Allows getting the state_idx-th PES of type electronic_state_type.
   * Being s the  spin quantum number, the string electronic_state_type can be
   * any choice between "Singlet"(s=0), "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES within all the PES of  type
   * electronic_state_type.
   * @param electronic_state_type  Which is the spin quantum number of this PES
   * where this wave packet stands.
   * @param PES                    PES evaluated on the simulation grid, _x.
   */
  Eigen::Ref<Eigen::VectorXd> get_PES(int state_idx,
                                      std::string electronic_state_type,
                                      Eigen::Ref<Eigen::VectorXd> PES) {
    bool done = false;
    // check if this key already exist, if so,  return the corres return the
    // corresponding PESS
    for (std::tuple<int, Eigen::VectorXd> &pes :
         _electronic_PES[electronic_state_type]) {
      if (std::get<0>(pes) == state_idx) {
        return std::get<1>(pes);
      }
    }
  }
  /**
   * @brief Allows getting a DM affecting the  state_idx-th PES of type
   * electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state_idx              Index of the PES that this DM is affecting.
   * @param electronic_state_type  Which is the spin quantum number of the PES
   * that this DM is affecting.
   * @param PES                    PES evaluated on the simulation grid, _x.
   */
  Eigen::Ref<Eigen::VectorXd> get_DM(int state_idx,
                                     std::string electronic_state_type,
                                     Eigen::Ref<Eigen::VectorXd> DM) {
    // check if this key already exist, if so,  return the corresponding DM.
    for (std::tuple<int, Eigen::VectorXd> &dm :
         _electronic_DM[electronic_state_type]) {
      if (std::get<0>(dm) == state_idx) {
        return std::get<1>(dm);
      }
    }
  }
  /**
   * @brief Allows getting a NACME coupling the  state1_idx and state2_idx PES
   * of type electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx             Index of the first PES that this NACME is
   * affecting.
   * @param state2_idx             Index of the second PES that this NACME is
   * affecting.
   * @param electronic_state_type  Which is the spin quantum number of the two
   * PES that this term is affecting.
   * @param NACME                  NACME evaluated on the simulation grid, _x.
   */
  Eigen::Ref<Eigen::VectorXd> get_NACME(int state1_idx, int state2_idx,
                                        std::string electronic_state_type,
                                        Eigen::Ref<Eigen::VectorXd> NACME) {
    // check if this key already exist, if so,  return the corresponding NACME
    for (std::tuple<int, int, Eigen::VectorXd> &nacme :
         _electronic_NACME[electronic_state_type]) {
      if (std::get<0>(nacme) == state1_idx &&
          std::get<1>(nacme) == state2_idx) {
        return std::get<2>(nacme);
      }
    }
  }
  /**
   * @brief Allows getting a TDM coupling the  state1_idx and state2_idx PES of
   * type electronic_state_type. Being s the  spin quantum number, the string
   * electronic_state_type can be any choice between "Singlet"(s=0),
   * "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx             Index of the first PES that this TDM is
   * affecting.
   * @param state2_idx             Index of the second PES that this TDM is
   * affecting.
   * @param electronic_state_type  Which is the spin quantum number of the two
   * PES that this term is affecting.
   * @param NACME                  NACME evaluated on the simulation grid, _x.
   */
  Eigen::Ref<Eigen::VectorXd> get_TDM(int state1_idx, int state2_idx,
                                      std::string electronic_state_type,
                                      Eigen::Ref<Eigen::VectorXd> TDM) {
    // check if this key already exist, if so,  return the corresponding TDM
    for (std::tuple<int, int, Eigen::VectorXd> &tdm :
         _electronic_TDM[electronic_state_type]) {
      if (std::get<0>(tdm) == state1_idx && std::get<1>(tdm) == state2_idx) {
        return std::get<2>(tdm);
      }
    }
  }
  /**
   * @brief Allows getting a SOC coupling the  state1_idx of type
   * electronic_state1_type and state2_idx PES of type electronic_state2_type.
   * Being s the  spin quantum number, the string electronic_state_type can be
   * any choice between "Singlet"(s=0), "Doublet"(s=1) or "Triplet"(s=2).
   *
   * @param state1_idx               Index of the first PES that this TDM is
   * affecting.
   * @param state2_idx               Index of the second PES that this TDM is
   * affecting.
   * @param electronic_state1_type   Type of electronic state of the first PES
   * that this term is coupling.
   * @param electronic_state2_type   Type of electronic state of the second PES
   * that this term is coupling.
   * @param SOC                      SOC evaluated on the simulation grid, _x.
   */
  Eigen::Ref<Eigen::VectorXd> get_SOC(int state1_idx, int state2_idx,
                                      std::string electronic_state1_type,
                                      std::string electronic_state2_type,
                                      Eigen::Ref<Eigen::VectorXd> SOC) {
    // check if this key already exist, if so, return the corresponding SOC.
    for (std::tuple<int, int, Eigen::VectorXd> &soc :
         _electronic_SOC[std::tuple<std::string, std::string>{
             electronic_state1_type, electronic_state2_type}]) {
      if (std::get<0>(soc) == state1_idx && std::get<1>(soc) == state2_idx) {
        return std::get<2>(soc);
      }
    }
  }

  // Controllers
  void step_forward() {
    if (_pn.size() != _perturbation_signal_samples.size())
      throw std::runtime_error("step_forward:: Iconsistent vector sizes.");

    // loop throuught the fractals weights using the symmetric approximant
    // until completing one step forward.
    std::vector<std::tuple<double, double>> _suzuki_data;
    for (int i = 0; i < _pn.size(); ++i) {
      _suzuki_data.push_back(
          std::tuple<double, double>{_pn[i], _perturbation_signal_samples[i]});
    }
   for(auto &&[pn,epsilon]: _suzuki_data) 
     symmetrized_apprroximant(_dt*pn, epsilon);
  }

private:
  // simlation time
  double simulation_time = 0;
  // Is the wave packet in position representation?
  bool _position_representation;
  // Is the wave packet in momentum representation?
  bool _momentum_representation;

  // Number of points in the grid.
  int _n_grid_points;
  // discretization of the coordinate
  Eigen::VectorXd _x;
  // indices for the grid elements
  std::vector<int> _grid_idx;
  // Kinetic energy discretized on the simlation grid
  Eigen::VectorXcd _exp_T;
  //  Space discretization step.
  double _dx;
  //  Momentum discretization step.
  double _dk;
  // Time discretization step.
  double _dt;
  // Kinetic energy operator
  Eigen::VectorXd _kin;
  // hbar
  double _hbar = 1.0; // using atomic units by default.
                      // PES that we are considering.
                      // Number of singlets
  int _n_singlets;
  // Number of doublets
  int _n_doublets;
  // Number of triplets
  int _n_triplets;
  // set true if the setup of the electronic structre is coomplete
  bool _electronic_structure_ready;
  // electronic Hamiltonian
  Eigen::MatrixXcd _electronic_H;
  // Handler for bookkeeping the blocking of the Hamiltonian
  std::map<std::tuple<std::string, std::string>, Eigen::MatrixXcd> _eH_Block;

  // electronic wave packets per electronic state type
  std::vector<std::tuple<int, Eigen::VectorXcd>> _singlet_psi;
  std::vector<std::tuple<int, Eigen::VectorXcd>> _doublet_psi;
  std::vector<std::tuple<int, Eigen::VectorXcd>> _triplet_psi;
  std::map<std::string, std::vector<std::tuple<int, Eigen::VectorXcd>>>
      _electronic_wave_packet{{"Singlet", _singlet_psi},
                              {"Doublet", _doublet_psi},
                              {"Triplet", _triplet_psi}};
  // Zero-based lists of unique indices for the electronic states
  // List of singlets
  std::vector<int> _singlet_list;
  // List of doblets
  std::vector<int> _doublet_list;
  // List of triplets
  std::vector<int> _triplet_list;
  // keep track if every state have being loaded
  std::vector<std::tuple<int, bool>> _singlet_PES_loaded;
  std::vector<std::tuple<int, bool>> _doublet_PES_loaded;
  std::vector<std::tuple<int, bool>> _triplet_PES_loaded;
  std::map<std::string, std::vector<std::tuple<int, bool>>>
      _electronic_PES_loaded{{"Singlet", _singlet_PES_loaded},
                             {"Doublet", _doublet_PES_loaded},
                             {"Triplet", _triplet_PES_loaded}};
  // Container holding the potential energy surfaces (PES)
  // one vector per state type
  std::vector<std::tuple<int, Eigen::VectorXd>> _singlet_PES;
  std::vector<std::tuple<int, Eigen::VectorXd>> _doublet_PES;
  std::vector<std::tuple<int, Eigen::VectorXd>> _triplet_PES;
  // Handlers for bookkeeping the PES
  std::map<std::string, std::vector<std::tuple<int, Eigen::VectorXd>>>
      _electronic_PES{{"Singlet", _singlet_PES},
                      {"Doublet", _doublet_PES},
                      {"Triplet", _triplet_PES}};
  // Container holding the dipole-moments (DM)
  // one vector per state type
  std::vector<std::tuple<int, Eigen::VectorXd>> _singlet_DM;
  std::vector<std::tuple<int, Eigen::VectorXd>> _doublet_DM;
  std::vector<std::tuple<int, Eigen::VectorXd>> _triplet_DM;
  // keep track if every DM have being loaded
  std::vector<std::tuple<int, bool>> _singlet_DM_loaded;
  std::vector<std::tuple<int, bool>> _doublet_DM_loaded;
  std::vector<std::tuple<int, bool>> _triplet_DM_loaded;
  // Handlers for bookkeeping the DM
  std::map<std::string, std::vector<std::tuple<int, Eigen::VectorXd>>>
      _electronic_DM{{"Singlet", _singlet_DM},
                     {"Doublet", _doublet_DM},
                     {"Triplet", _triplet_DM}};
  std::map<std::string, std::vector<std::tuple<int, bool>>>
      _electronic_DM_loaded{{"Singlet", _singlet_DM_loaded},
                            {"Doublet", _doublet_DM_loaded},
                            {"Triplet", _triplet_DM_loaded}};
  // Container holding the non-adiabatic coupling matrix elements (NACMEs)
  // The indices of the two states that the NACME is coupling should be of the
  // same type
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_NACME;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_NACME;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _triplet_NACME;
  // keep track if every NACME have being loaded
  std::vector<std::tuple<int, int, bool>> _singlet_NACME_loaded;
  std::vector<std::tuple<int, int, bool>> _doublet_NACME_loaded;
  std::vector<std::tuple<int, int, bool>> _triplet_NACME_loaded;
  // Handlers for bookkeeping the NACME
  std::map<std::string, std::vector<std::tuple<int, int, Eigen::VectorXd>>>
      _electronic_NACME{{"Singlet", _singlet_NACME},
                        {"Doublet", _doublet_NACME},
                        {"Triplet", _triplet_NACME}};
  std::map<std::string, std::vector<std::tuple<int, int, bool>>>
      _electronic_NACME_loaded{{"Singlet", _singlet_NACME_loaded},
                               {"Doublet", _doublet_NACME_loaded},
                               {"Triplet", _triplet_NACME_loaded}};
  // Containers holding the transition dipole moments (TDM)
  // The indices of the two states that the TDM is coupling should be of the
  // same type
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_TDM;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_TDM;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _triplet_TDM;
  // keep track if every TDM have being loaded
  std::vector<std::tuple<int, int, bool>> _singlet_TDM_loaded;
  std::vector<std::tuple<int, int, bool>> _doublet_TDM_loaded;
  std::vector<std::tuple<int, int, bool>> _triplet_TDM_loaded;
  // Handlers for bookkeeping the TDM
  std::map<std::string, std::vector<std::tuple<int, int, Eigen::VectorXd>>>
      _electronic_TDM{{"Singlet", _singlet_TDM},
                      {"Doublet", _doublet_TDM},
                      {"Triplet", _triplet_TDM}};
  std::map<std::string, std::vector<std::tuple<int, int, bool>>>
      _electronic_TDM_loaded{{"Singlet", _singlet_TDM_loaded},
                             {"Doublet", _doublet_TDM_loaded},
                             {"Triplet", _triplet_TDM_loaded}};
  // Container holding the SOCs
  // The indices of the two states that the SOC is coupling should be of the
  // different type.
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_doublet_SOC;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_triplet_SOC;
  std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_triplet_SOC;
  // keep track if every SOC have being loaded
  std::vector<std::tuple<int, int, bool>> _singlet_doublet_SOC_loaded;
  std::vector<std::tuple<int, int, bool>> _singlet_triplet_SOC_loaded;
  std::vector<std::tuple<int, int, bool>> _doublet_triplet_SOC_loaded;
  // Handlers for bookkeeping the SOC
  std::map<std::tuple<std::string, std::string>,
           std::vector<std::tuple<int, int, Eigen::VectorXd>>>
      _electronic_SOC{{{"Singlet", "Doublet"}, _singlet_doublet_SOC},
                      {{"Singlet", "Triplet"}, _singlet_triplet_SOC},
                      {{"Doublet", "Triplet"}, _doublet_triplet_SOC}};
  std::map<std::tuple<std::string, std::string>,
           std::vector<std::tuple<int, int, bool>>>
      _electronic_SOC_loaded{
          {{"Singlet", "Doublet"}, _singlet_doublet_SOC_loaded},
          {{"Singlet", "Triplet"}, _singlet_triplet_SOC_loaded},
          {{"Doublet", "Triplet"}, _doublet_triplet_SOC_loaded}};

  // Quantum dynamics related containers
  // Wave packet propagation mode
  std::map<std::string, int> propagation_mode{{"forward", 0}, {"backward", 1}};
  int _prop_mode;
  // Wave packet propagation scheme
  // only unitary and  symplectic schemes will be implemented.
  std::map<std::string, int> propagation_method{
      {"implicit midpoint", 2}, {"Crank-Nicolson", 1}, {"Suzuki-fractal", 0}};
  // desired order of accuuracy to achive using Suzuki fractal decomposition
  // method.
  int _suzuki_fractal_order;
  // weights for the Suzuki fractal decomposition method.
  std::vector<double> _pn;
  // list of times to evaluate the time dependent perturbation to be used with
  // the Suzuki fractal decomposition method.
  std::vector<double> _suzuki_fractal_perturbation_eval;
  // list of samples of the time-dependent perturbation signal
  std::vector<double> _perturbation_signal_samples;
  int _prop_method;
  // set true if the setup of the propagation scheme is complete
  bool _propagations_scheme_ready;
  // Internal methods

  /**
   * @brief Changes the representation of the wave packet to momentum.
   */
  void _transform_to_momentum_representation() {
    // if wave packet is ialready in momentum representation, do nothing.
    if (_momentum_representation && !_position_representation) {
      return;
    } else {
      // transform to momntum representation all the components of the
      // wavepacket
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["Singlet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["Doublet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["triplet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
      _momentum_representation = true;
      _position_representation = false;
    }
  }
  /**
   * @brief Changes the representation of the wave packet to poosition.
   */
  void _transform_to_position_representation() {
    // if wave packet is already in position  representation, do nothing.
    if (_position_representation && !_momentum_representation) {
      return;
    } else {
      // transform to momentum representation all the components of the
      // wavepacket
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["Singlet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::inv_fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["Doublet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::inv_fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
#pragma omp parallel for
      for (std::tuple<int, Eigen::VectorXcd> &wp :
           _electronic_wave_packet["triplet"]) {
        std::shared_ptr<Eigen::VectorXcd> tmp_psi(
            new Eigen::VectorXcd(_n_grid_points));
        *tmp_psi = std::get<1>(wp);
        tmp_psi = PROCTOR::inv_fft1(tmp_psi);
        tmp_psi->normalize();
        std::get<1>(wp) = *tmp_psi;
      }
      _position_representation = true;
      _momentum_representation = false;
    }
  }
  /**
   * @brief Returrns the coefficients (Eq. (3) in the reference bellow ) to
   * perform the fractal decomposition of an exponential operator.
   *
   * Suzuki, Masuo. "Fractal decomposition of exponential operators with
   * applications to many-body theories and Monte Carlo simulations." Physics
   * Letters A 146.6 (1990): 319-323.
   *
   * @param k  Half of the total order of the desired approximant.
   *
   * @return list of coefficients to accomplish the of 2kth order approximant of
   * the exponential operator.
   */
  std::vector<double> _suzuki_fractal_decomposition_coeffs(int k) {
    // recursive generation of the coefficients in Eq. (3)
    if (k < 0)
      throw std::runtime_error("Simulation1D::_suzuki_fractal_decomposition_"
                               "coeffs: k should be a positive integer.");
    if (k == 1) {
      return std::vector<double>{1.0};
    } else {
      std::vector<double> output;
      /* Eq. (42)*/
      double p_k = 1.0 / (4.0 - std::pow(4.0, (1.0 / (2.0 * k - 1.0))));
      /*
      // This actually match the coefficients in the reference
      std::cout << std::setprecision(18);
      std::cout<< "p["<< k <<"]= "<<p_k<<std::endl;
      */
      std::vector<double> k_th_coeffs = {p_k, p_k, 1.0 - 4.0 * p_k, p_k, p_k};
      for (auto v : k_th_coeffs)
        for (auto sfdc : _suzuki_fractal_decomposition_coeffs(k - 1))
          output.push_back(v * sfdc);
      return output;
    }
  }
  /**
   * @brief Returns a vector with the exponential of the kinetic energy part of
   * the evolution  operator evaluated on a grid in  the momenttum
   * representation. This routine evaluates the formula:
   *
   * e^{(x)T},
   *
   * with
   *
   * x=-i*_hbar*dt
   *
   * corresponding to the first factor of Eq.(5) in the reference [1].
   *
   * [1] Suzuki, Masuo. "Fractal decomposition of exponential operators with
   * applications to many-body theories and Monte Carlo simulations." Physics
   * Letters A 146.6 (1990): 319-323.
   *
   *
   * @param dt
   *
   * @return
   */
  Eigen::VectorXcd _update_kinetic_exp_operator(double dt) {
    _dt = dt;
    // populating
    double L = _x(_n_grid_points - 1) - _x(0);
    // step-size in k
    _dk = 2.0 * M_PI / L;
    // discretization of the corresponding moomeentum
    std::vector<double> _k;
    // filling the momentum grid
    _k.resize(_n_grid_points);
    // populating _k with valuues between -_n_grid_points / 2 and _n_grid_points
    // / 2-1
    std::iota(_k.begin(), _k.end(), (double)(0));
    // _k scaled by _dk
    // fixing capture issues
    double dK = _dk;
    int N = _n_grid_points;
    // see Spectral Methods in MATLAB. Lloyd N. Trefethen, pag 24.
    std::transform(_k.begin(), _k.end(), _k.begin(), [N, dK](double i) {
      return i < N / 2 ? i * dK : (i - N) * dK;
    });
    // __exp_T = e^(-i*dt*T/2)
    Eigen::VectorXcd __exp_T;
    __exp_T.resize(_n_grid_points);
    for (int j = 0; j < _n_grid_points; ++j) {
      double theta = -_k[j] * _k[j] / (2.0) * (_dt / (2 * _hbar));
      __exp_T(j) = std::complex<double>(cos(theta), sin(theta));
    }
    return __exp_T;
  }
  /**
   * @brief Symmetrized approximant to the timee-evolution opertator
   * Current implementation is based on the following references:
   *
   * [1] Alvarellos, José, and Horia Metiu.
   * "The evolution of the wave function in a curve crossing problem computed by
   * a fast Fourier transform method." The Journal of chemical physics 88.8
   * (1988): 4957-4966.
   *
   * [2] Hatano, Naomichi, and Masuo Suzuki.
   * "Finding exponential product formulas of higher orders." Quantum annealing
   * and other optimization methods. Springer, Berlin, Heidelberg, 2005. 37-68.
   *
   *
   * @param __dt       Time-step for setting up the approximant.
   * @param __epsilon  Electric field compnent of the external light pulse that
   * is interacting with the wave packet.
   */
  void symmetrized_apprroximant(double __dt, double __epsilon) {
    if (_n_singlets > 0 && _n_doublets > 0 && _n_triplets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Singlet", "Doublet", "Triplet"};
      std::vector<int> block_sizes = {_n_singlets, _n_doublets, _n_triplets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{
          {"Singlet", 0},
          {"Doublet", _n_singlets},
          {"Triplet", _n_singlets + _n_doublets}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //
    } else if (_n_singlets > 0 && _n_doublets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Singlet", "Doublet"};
      std::vector<int> block_sizes = {_n_singlets, _n_doublets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{
          {"Singlet", 0}, {"Doublet", _n_singlets}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //

    } else if (_n_singlets > 0 && _n_triplets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Singlet", "Triplet"};
      std::vector<int> block_sizes = {_n_singlets, _n_triplets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{
          {"Singlet", 0}, {"Triplet", _n_singlets}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //

    } else if (_n_doublets > 0 && _n_triplets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Doublet", "Triplet"};
      std::vector<int> block_sizes = {_n_doublets, _n_triplets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{
          {"Doublet", 0}, {"Triplet", _n_doublets}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //

    } else if (_n_triplets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Triplet"};
      std::vector<int> block_sizes = {_n_triplets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{{"Triplet", 0}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //

    } else if (_n_doublets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Doublet"};
      std::vector<int> block_sizes = {_n_doublets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{{"Doublet", 0}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //

    } else if (_n_singlets > 0) {
      // => BEGIN <= //
      // list of states to consider
      std::vector<std::string> _state_types = {"Singlet"};
      std::vector<int> block_sizes = {_n_singlets};
      // beginning of the block for each state type
      std::vector<std::tuple<std::string, int>> _blocking{{"Singlet", 0}};

      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      Eigen::VectorXcd __exp_T = _update_kinetic_exp_operator(__dt);
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => Propagation corresponding to the potential energy operator <= //
#pragma omp parallel for
      for (auto xi : _grid_idx) {
        // storage for the local wave function
        Eigen::VectorXcd _psi_loc =
            Eigen::VectorXcd::Zero(_n_singlets + _n_doublets + _n_triplets);
        // gathering the local wave function
        for (std::tuple<std::string, int> block : _blocking)
          for (std::tuple<int, Eigen::VectorXcd> &wp :
               _electronic_wave_packet[std::get<0>(block)]) {
            _psi_loc(std::get<1>(block) + std::get<0>(wp)) =
                std::get<1>(wp)(xi);
          }
        // storage for the local nonadiabatic potential
        Eigen::MatrixXcd _V_loc =
            Eigen::MatrixXcd::Zero(_n_singlets + _n_doublets + _n_triplets,
                                   _n_singlets + _n_doublets + _n_triplets);
        // gathering the local Hamiltonian
        for (std::tuple<std::string, int> block1 : _blocking) {
          // PES
          for (std::tuple<int, Eigen::VectorXd> &pes1 :
               _electronic_PES[std::get<0>(block1)]) {
            // if the PES is loaded, count it
            if (std::get<1>(_electronic_PES_loaded[std::get<0>(block1)]
                                                  [std::get<0>(pes1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(pes1),
                     std::get<1>(block1) + std::get<0>(pes1)) +=
                  std::get<1>(pes1)(xi);
            }
          }
          // DM
          for (std::tuple<int, Eigen::VectorXd> &dm1 :
               _electronic_DM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_DM_loaded[std::get<0>(block1)]
                                                 [std::get<0>(dm1)]))
              _V_loc(std::get<1>(block1) + std::get<0>(dm1),
                     std::get<1>(block1) + std::get<0>(dm1)) +=
                  __epsilon * std::get<1>(dm1)(xi);
          }
          // NACME
          for (std::tuple<int, int, Eigen::VectorXd> &nacme1 :
               _electronic_NACME[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_NACME_loaded[std::get<0>(block1)]
                                                    [std::get<0>(nacme1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(nacme1),
                     std::get<1>(block1) + std::get<1>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(nacme1),
                     std::get<1>(block1) + std::get<0>(nacme1)) +=
                  std::get<2>(nacme1)(xi);
            }
          }
          // TDM
          for (std::tuple<int, int, Eigen::VectorXd> &tdm1 :
               _electronic_TDM[std::get<0>(block1)]) {
            if (std::get<1>(_electronic_TDM_loaded[std::get<0>(block1)]
                                                  [std::get<0>(tdm1)])) {
              _V_loc(std::get<1>(block1) + std::get<0>(tdm1),
                     std::get<1>(block1) + std::get<1>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
              _V_loc(std::get<1>(block1) + std::get<1>(tdm1),
                     std::get<1>(block1) + std::get<0>(tdm1)) +=
                  __epsilon * std::get<2>(tdm1)(xi);
            }
          }
          // SOC
          for (std::tuple<std::string, int> block2 : _blocking) {
            if (std::get<1>(block1) != std::get<1>(block2)) {
              for (std::tuple<int, int, Eigen::VectorXd> &soc1 :
                   _electronic_SOC[std::tuple<std::string, std::string>{
                       std::get<0>(block1), std::get<0>(block2)}]) {
                _V_loc(std::get<1>(block1) + std::get<0>(soc1),
                       std::get<1>(block1) + std::get<1>(soc1)) +=
                    std::get<2>(soc1)(xi);
                _V_loc(std::get<1>(block1) + std::get<1>(soc1),
                       std::get<1>(block1) + std::get<0>(soc1)) +=
                    std::get<2>(soc1)(xi);
              }
            }
          }
          // Local Hamiltonian is complete. multiplying it by  -i*dt
          _V_loc *= -std::complex<double>(0.0, 1.0) * __dt;
          // |psi_loc> -> e^{-i*V*dt} |psi_loc>
          _psi_loc = _V_loc.exp() * _psi_loc;
          // copying the results back to the global wave packet.
          for (std::tuple<std::string, int> block : _blocking)
            for (std::tuple<int, Eigen::VectorXcd> &wp :
                 _electronic_wave_packet[std::get<0>(block)]) {
              std::get<1>(wp)(xi) =
                  _psi_loc(std::get<1>(block) + std::get<0>(wp));
            }
        }
      }
      // => From position to momentum representation <= //
      _transform_to_momentum_representation();
      // => Propagation corresponding to half kinetic energy operator in
      // momentum representation <= //
      for (auto state : _state_types)
        for (std::tuple<int, Eigen::VectorXcd> &wp :
             _electronic_wave_packet[state]) {
          std::get<1>(wp).array() *= __exp_T.array();
        }
      // => From position to momentum representation <= //
      _transform_to_position_representation();
      // => BEGIN <= //
    }
  }
};
} // namespace PROCTOR
#ifdef __cplusplus
} // extern "C"
#endif
