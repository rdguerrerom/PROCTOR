#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>
//#include <pybind11/eigen.h>
//#include <pybind11/stl.h>
#include "level0.hpp"
#include <map>
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
  // Geetters
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
   * @brief Contructor for an empty simulation. The initial state shall be
   loaded using a
   * different mechanism.

   *
   * @param[in] x             Simulation grid for the coordinate provided by the
   user
   * @param[in] dx            Grid spacing
   * @param[in] dt            Time-step fpr the simulation
   * @param[in] n_grid_points Number of grid points on the simulation grid
   * @param[in] n_singlets    Number of singlet states
   * @param[in] n_doublets    Number of doublet states
   * @param[in] n_triplets    Number of triplet states
   */
  Simulation1D(const Eigen::VectorXd &x, double dx, double dt,
               int n_grid_points, int n_singlets, int n_doublets,
               int n_triplets)
      : _x(x), _dx(dx), _dt(dt), _n_grid_points(n_grid_points),
        _n_singlets(n_singlets), _n_doublets(n_doublets),
        _n_triplets(n_triplets) {
    // checking that the information makes sense

    // resizing containers
    int total_electronic_states = _n_singlets + _n_doublets + _n_triplets;

    // populating
    double L = _x(_n_grid_points - 1) - x(0);
    // step-size in k
    _dk = 2.0 * M_PI / L;
    // filling the momentum grid
    _k.resize(_n_grid_points);
    // populating _k with valuues between -_n_grid_points / 2 and _n_grid_points
    // / 2-1
    std::iota(_k.begin(), _k.end(), (double)(-_n_grid_points / 2));
    // _k scaled by _dk
    // fixing capture issues
    double dK = _dk;
    int N = _n_grid_points;
    std::transform(_k.begin(), _k.end(), _k.begin(),
                   [N, dK](double i) { return i * dK; });
  }
  // Desstructor
  ~Simulation1D() {}

  // Setters
  // Allows loading the initial condition
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
  // Allows load a new PES
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
  // Allows loading a new DM
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
  // Allows loading a new NACME
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
  // Allows loading a new TDM
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
  // Allows loading a new SOC
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
    //std::cout << std::get<0>(
    //                 _electronic_SOC[std::tuple<std::string, std::string>{
    //                                     electronic_state1_type,
    //                                     electronic_state2_type}]
    //                     .back())
    //          << std::endl;
    //std::cout << std::get<1>(
    //                 _electronic_SOC[std::tuple<std::string, std::string>{
    //                                     electronic_state1_type,
    //                                     electronic_state2_type}]
    //                     .back())
    //          << std::endl;
    //std::cout << std::get<2>(
    //                 _electronic_SOC[std::tuple<std::string, std::string>{
    //                                     electronic_state1_type,
    //                                     electronic_state2_type}]
    //                     .back())
    //          << std::endl;
  }
  // Getters
  /**
   * @brief Check if the PES setup is correct
   *
   * @return True, if the setup is correct. False if not.
   */
  bool electronic_PES_complete() {
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
  bool electronic_DM_complete() {
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
  bool electronic_TDM_complete() {
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
  bool electronic_NACME_complete() {
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

private:
  // Number of points in the grid.
  int _n_grid_points;
  // discretization of the coordinate
  Eigen::VectorXd _x;
  // discretization of the corresponding moomeentum
  std::vector<double> _k;
  //  Space discretization step.
  double _dx;
  //  Momentum discretization step.
  double _dk;
  // Time discretization step.
  double _dt;
  // Kinetic energy operator
  Eigen::VectorXd _kin;
  // hbar
  double _hbar;
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
      {"implicit midpoint", 0}, {"Crank-Nicolson", 1}, {"Suzuki-Trotter", 2}};
  int _prop_method;
  // set true if the setup of the propagation scheme is complete
  bool _propagations_scheme_ready;
};
} // namespace PROCTOR
#ifdef __cplusplus
} // extern "C"
#endif
