/*
  Copyright (C) 2011 - 2020 by the authors of the ASPECT code.

  This file is part of ASPECT.

  ASPECT is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2, or (at your option)
  any later version.

  ASPECT is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with ASPECT; see the file LICENSE.  If not see
  <http://www.gnu.org/licenses/>.
*/

#ifndef _aspect_boundary_traction_crustal_flow_h
#define _aspect_boundary_traction_crustal_flow_h

#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/shared_tria.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <csignal>

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/utilities.h>

namespace aspect
{
  namespace Postprocess
  {
    using namespace dealii;

    /**
     * A class that implements traction boundary conditions based on ...
     *
     * @ingroup BoundaryTractions
     */
    template <int dim>
    class CrustalFlow : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        /**
         * Constructor.
         */
        CrustalFlow ();

        /*
         * Destructor
         */
        ~CrustalFlow ();

        /**
         * Initialization function. This function is called once at the
         * beginning of the program. Checks preconditions.
         */
        void
        initialize () override;

        /**
        * A function that is called at the beginning of each time step to
        * indicate what the model time is for which the boundary values will
        * next be evaluated. For the current class, the function passes to
        * the parsed function what the current time is.
        */
        void update () override;

        /**
         * Return the boundary traction as a function of position. The
         * (outward) normal vector to the domain is also provided as
         * a second argument.
         */
        std::pair<std::string,std::string>
        execute (TableHandler &statistics) override;
        // Tensor<1,dim>
        // boundary_traction (const types::boundary_id boundary_indicator,
        //                    const Point<dim> &p,
        //                    const Tensor<1,dim> &normal) const override;

        // void save (std::map<std::string, std::string> &status_strings) const override;
        // void load (const std::map<std::string, std::string> &status_strings) override;

      private:
        static constexpr unsigned int boundarydim = dim - 1;
        parallel::distributed::Triangulation<boundarydim, dim> triangulation;
        DoFHandler<boundarydim, dim> dof_handler;
        FESystem<boundarydim, dim> fe;
        AffineConstraints<double> constraints;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        dealii::LinearAlgebraTrilinos::MPI::SparseMatrix system_matrix;
        dealii::LinearAlgebraTrilinos::MPI::Vector locally_relevant_solution;
        dealii::LinearAlgebraTrilinos::MPI::Vector old_locally_relevant_solution;
        dealii::LinearAlgebraTrilinos::MPI::Vector system_rhs;

        const FEValuesExtractors::Vector u_extractor;
        const FEValuesExtractors::Scalar p_extractor;
        const FEValuesExtractors::Scalar h_extractor;
        const FEValuesExtractors::Scalar s_extractor;

        void setup_dofs ();
        void assemble_system (const double dt);
        void solve ();
        void refine_mesh ();
        void output_results (const unsigned int timestep,
                             const double time);
        double get_dt (const double max_dt);

        double RHO_C=2650;
        double RHO_M=3300;
        double RHO_S=3550;

        types::boundary_id surface_boundary_id;

        static constexpr double PI = 3.14159;
        static constexpr double ETA = 1.0;
    };
  }
}

#endif
