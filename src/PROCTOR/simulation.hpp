#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
//#include <pybind11/eigen.h>
//#include <pybind11/stl.h>
#include "level0.hpp"
#include <map>
#include <tuple>

#ifdef __cplusplus
extern "C" {
#endif

  class Simulation0D {
    public:
      // Constructor
      Simulation0D() {}
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
      Simulation1D() {}
      /**
       * @brief Contructor for an empty simulation. The initial state shall be loaded using a 
       * different mechanism.
      
       *
       * @param x             Simulation grid for the coordinate provided by the user
       * @param dx            Grid spacing
       * @param dt            Time-step fpr the simulation 
       * @param n_grid_points Number of grid points on the simulation grid
       * @param n_singlets    Number of singlet states  
       * @param n_doublets    Number of doublet states
       * @param n_triplets    Number of triplet states
       */
      Simulation1D(std::vector<double> x, double dx, double dt, int n_grid_points, int n_singlets,
          int n_doublets, int n_triplets)
        :_x(x), _dx(dx), _dt(dt), _n_grid_points(n_grid_points), _n_singlets(n_singlets), _n_doublets(n_doublets),
        _n_triplets(n_triplets) {
          // checking that the information makes sense
          
          // resizing containers

          // populating 
          double L = _x.back() - x.front();
          // step-size in k
          _dk = 2.0*M_PI/L;
          // filling the momentum grid
          _k.resize(_n_grid_points);
          // populating _k with valuues between -_n_grid_points / 2 and _n_grid_points / 2-1
          std::iota(_k.begin(), _k.end(), (double)(-_n_grid_points / 2));
          // _k scaled by _dk
          // fixing capture issues
          double dK = _dk;
          int N=_n_grid_points;
          std::transform(_k.begin(), _k.end(), _k.begin(),
              [N, dK](double i) { return i * dK; });



        }
      // Desstructor
      ~Simulation1D() {}
      // Setters
      void set_PES(int state_idx, std::string electrronic_state_type,
          Eigen::VectorXd pes);

      // Getters

    private:
      // Number of points in the grid.
      int _n_grid_points;
      // discretization of the coordinate
      std::vector<double> _x;
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
      // Zero-based lists of unique indices for the electronic states
      // List of singlets
      std::vector<int> _singlets_list;
      // List of doblets
      std::vector<int> _doublets_list;
      // List of triplets
      std::vector<int> _triplets_list;
      // Container holding the potential energy surfaces (PES)
      // one vector per state type
      std::vector<std::tuple<int, Eigen::VectorXd>> _singlet_pes;
      std::vector<std::tuple<int, Eigen::VectorXd>> _doublet_pes;
      std::vector<std::tuple<int, Eigen::VectorXd>> _triplet_pes;
      // Container holding the dipole-moments (DM)
      // one vector per state type
      std::vector<std::tuple<int, Eigen::VectorXd>> _singlet_dm;
      std::vector<std::tuple<int, Eigen::VectorXd>> _doublet_dm;
      std::vector<std::tuple<int, Eigen::VectorXd>> _triplet_dm;
      // Container holding the non-adiabatic coupling matrix elements (NACMEs)
      // The indices of the two states that the NACME is coupling should be of the
      // same type
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_nacme;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_nacme;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _triplet_nacme;
      // Containers holding the transition dipole moments (TDM)
      // The indices of the two states that the TDM is coupling should be of the
      // same type
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_tdm;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_tdm;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _triplet_tdm;
      // Container holding the SOCs
      // The indices of the two states that the SOC is coupling should be of the
      // different type.
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_doublet_socs;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _singlet_triplet_socs;
      std::vector<std::tuple<int, int, Eigen::VectorXd>> _doublet_triplet_socs;

      // Quantum dynamics related containers
      // Wave packet propagation mode
      std::map<std::string, int> propagation_mode{{"forward", 0}, {"backward", 1}};
      int _prop_mode;
      // Wave packet propagation scheme
      // only unitary and  symplectic schemes will be implemented.
      std::map<std::string, int> propagation_method{
        {"implicit midpoint", 0}, {"Crank-Nicolson", 1}, {"Suzuki-Trotter", 2}};
      int _prop_method;
  };

#ifdef __cplusplus
} // extern "C"
#endif
