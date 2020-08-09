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


#include <aspect/boundary_traction/crustal_flow.h>
#include <aspect/utilities.h>
#include <aspect/global.h>

namespace aspect
{
  namespace BoundaryTraction
  {
    template <int dim>
    CrustalFlow<dim>::CrustalFlow ()
      :
      triangulation (MPI_COMM_WORLD,
                     typename Triangulation<boundarydim, dim>::MeshSmoothing (
                       Triangulation<boundarydim, dim>::smoothing_on_refinement | Triangulation<boundarydim, dim>::smoothing_on_coarsening),
                     parallel::distributed::Triangulation<boundarydim, dim>::mesh_reconstruction_after_repartitioning),
      dof_handler (triangulation),
      fe (FE_Q<boundarydim, dim> (2), boundarydim /* Crustal flow velocity */,
          FE_Q<boundarydim, dim> (1), 1 /* P -- Crustal thickness */,
          FE_Q<boundarydim, dim> (1), 1 /* h -- Elastic plate deflection */,
          FE_Q<boundarydim, dim> (1), 1 /* s -- Overburden load */),
      u_extractor (0),
      p_extractor (boundarydim),
      h_extractor (boundarydim+1),
      s_extractor (boundarydim+2)
    {}

    template <int dim>
    CrustalFlow<dim>::~CrustalFlow ()
    {
      dof_handler.clear ();
    }

    template <int dim>
    void
    CrustalFlow<dim>::initialize ()
    {
      surface_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");

      {
        GridGenerator::hyper_sphere(triangulation);
        triangulation.refine_global(3);
        setup_dofs();

        // std::set<types::boundary_id> boundary_ids;
        // boundary_ids.insert(surface_boundary_id);
        // GridGenerator::extract_boundary_mesh(this->get_triangulation(),
        //                                      triangulation,
        //                                      boundary_ids);
      }
    }

    template <int dim>
    void
    CrustalFlow<dim>::update()
    {
      double model_timestep = this->get_timestep();
      double dt;
      double time = 0;
      do
        {
          dt = get_dt();
          assemble_system(dt);
          solve();
          time += dt;
        }
      while (time < model_timestep);
    }


    template <int dim>
    Tensor<1,dim>
    CrustalFlow<dim>::
    boundary_traction (const types::boundary_id,
                       const Point<dim> &,
                       const Tensor<1,dim> &normal_vector) const
    {
      double traction = 0;
      return traction * normal_vector;
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    setup_dofs()
    {
      dof_handler.distribute_dofs (fe);
      {
        locally_owned_dofs = dof_handler.locally_owned_dofs ();
        DoFTools::extract_locally_relevant_dofs (dof_handler,
                                                 locally_relevant_dofs);
      }

      {
        constraints.clear ();
        constraints.reinit (locally_relevant_dofs);
        DoFTools::make_hanging_node_constraints (dof_handler, constraints);
        VectorTools::interpolate_boundary_values (dof_handler, 0,
                                                  Functions::ZeroFunction<dim>(5),
                                                  constraints,
                                                  fe.component_mask(p_extractor)
                                                  | fe.component_mask(h_extractor)
                                                  | fe.component_mask(s_extractor));

        std::vector<Point<dim> >  constraint_locations;
        std::vector<unsigned int> constraint_component_indices;
        std::vector<double>       constraint_values;

        for (unsigned int i = 0; i < constraint_locations.size(); ++i)
          {
            const Point<dim> constrain_where = constraint_locations[i]; // TODO
            const unsigned int constraint_component_index = constraint_component_indices[i];
            const double constraint_value = constraint_values[i];

            // Find nearest quadrature point.
            double min_local_distance = std::numeric_limits<double>::max();
            unsigned int local_best_dof_index;

            const std::vector<Point<boundarydim> > points = fe.get_unit_support_points();
            const Quadrature<boundarydim> quadrature (points);
            FEValues<boundarydim, dim> fe_values (fe, quadrature, update_quadrature_points);
            typename DoFHandler<boundarydim,dim>::active_cell_iterator cell;
            for (cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell)
              {
                if (! cell->is_artificial())
                  {
                    fe_values.reinit(cell);
                    std::vector<unsigned int> local_dof_indices(fe.dofs_per_cell);
                    cell->get_dof_indices (local_dof_indices);

                    for (unsigned int q=0; q<quadrature.size(); q++)
                      {
                        // If it's okay to constrain this DOF
                        if (constraints.can_store_line(local_dof_indices[q]) &&
                            !constraints.is_constrained(local_dof_indices[q]))
                          {
                            const unsigned int c_idx = fe.system_to_component_index(q).first;
                            if (c_idx == constraint_component_index)
                              {
                                const Point<dim> p = fe_values.quadrature_point(q);
                                const double distance = constrain_where.distance(p);
                                if (distance < min_local_distance)
                                  {
                                    min_local_distance = distance;
                                    local_best_dof_index = local_dof_indices[q];
                                  }
                              }
                          }
                      }
                  }
              }
            const double global_nearest = Utilities::MPI::min (min_local_distance, MPI_COMM_WORLD);
            if (min_local_distance == global_nearest)
              {
                constraints.add_line (local_best_dof_index);
                constraints.set_inhomogeneity (local_best_dof_index, constraint_value);
              }
          }
        constraints.close ();
      }

      DynamicSparsityPattern dsp (locally_relevant_dofs);
      DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern (dsp,
                                                  dof_handler.n_locally_owned_dofs_per_processor (),
                                                  MPI_COMM_WORLD,
                                                  locally_relevant_dofs);
      system_matrix.reinit (locally_owned_dofs, locally_owned_dofs, dsp, MPI_COMM_WORLD);

      system_rhs.reinit (locally_owned_dofs, MPI_COMM_WORLD);
      locally_relevant_solution.reinit (locally_owned_dofs, locally_relevant_dofs,
                                        MPI_COMM_WORLD);
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    assemble_system(const double dt)
    {
      const QGauss<boundarydim> quadrature_formula (5);
      FEValues<boundarydim,dim> fe_values (fe, quadrature_formula,
                                           update_values | update_JxW_values | update_gradients
                                           | update_quadrature_points);
      const unsigned int dofs_per_cell = fe.dofs_per_cell;
      const unsigned int n_q_points = quadrature_formula.size ();
      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<Tensor<1,dim> > phi_u (dofs_per_cell);
      std::vector<double> phi_p (dofs_per_cell);
      std::vector<double> phi_h (dofs_per_cell);
      std::vector<double> phi_s (dofs_per_cell);



      std::vector<double> div_phi_u (dofs_per_cell);

      std::vector<double> old_h_values (n_q_points);
      std::vector<double> old_s_values (n_q_points);

      typename DoFHandler<boundarydim,dim>::active_cell_iterator cell =
        dof_handler.begin_active (), endc = dof_handler.end ();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned ())
          {
            cell_matrix = 0;
            cell_rhs = 0;
            fe_values.reinit (cell);

            fe_values[h_extractor].get_function_values (locally_relevant_solution,
                                                        old_h_values);
            fe_values[s_extractor].get_function_values (locally_relevant_solution,
                                                        old_s_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k] = fe_values[u_extractor].value (k,q);
                    phi_p[k] = fe_values[p_extractor].value (k,q);
                    phi_h[k] = fe_values[h_extractor].value (k,q);
                    phi_s[k] = fe_values[s_extractor].value (k,q);
                    div_phi_u[k] = fe_values[u_extractor].divergence (k,q);
                  }

                Point<dim> loc = fe_values.quadrature_point (q);
                const double sigma_zz = RHO_C * old_h_values[q] + RHO_S * old_s_values[q];
                const double emplacement = 1e-7*std::cos(PI*loc[0]);
                const double h_n = old_h_values[q] + 1;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  {
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                      {
                        cell_matrix (i, j) += (

                                                2 * PI * ETA * phi_u[i] * phi_u[j]
                                                - (h_n*h_n) * div_phi_u[i]*phi_p[j]

                                                + phi_p[i]*div_phi_u[j]

                                                + phi_h[i] * (phi_h[j] + dt*(2.0/3.0)*div_phi_u[j])

                                                + phi_s[i] * phi_s[j]
                                              ) * fe_values.JxW (q) ;
                      }
                    cell_rhs (i) += (

                                      phi_p[i] * sigma_zz/2.0

                                      + phi_h[i] * old_h_values[q]

                                      + phi_s[i] * (old_s_values[q] + dt*emplacement)
                                    ) * fe_values.JxW (q);
                  }
              }
            cell->get_dof_indices (local_dof_indices);
            constraints.distribute_local_to_global (cell_matrix, cell_rhs,
                                                    local_dof_indices, system_matrix, system_rhs);
          }
      system_matrix.compress (VectorOperation::add);
      system_rhs.compress (VectorOperation::add);
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    solve()
    {
      dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution (
        locally_owned_dofs, MPI_COMM_WORLD);

      SolverControl cn;
      TrilinosWrappers::SolverDirect solver (cn);
      try
        {
          solver.solve (system_matrix, distributed_solution, system_rhs);
          constraints.distribute (distributed_solution);
          locally_relevant_solution = distributed_solution;
        }
      catch (const std::exception &exc)
        {
          if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            {
              AssertThrow(false, ExcMessage (
                            std::string ("The direct Stokes solver failed with error:\n\n")
                            + exc.what ()));
            }
          else
            {
              throw QuietException ();
            }
        }
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    refine_mesh()
    {
      // parallel::distributed::SolutionTransfer<boundarydim,
      //          TrilinosWrappers::MPI::Vector, DoFHandler<boundarydim,dim>>
      //          solutionTx (dof_handler);
      //
      // {
      //   Vector<float> estimated_error_per_cell (triangulation.n_active_cells ());
      //   KellyErrorEstimator<boundarydim, dim>::estimate (dof_handler, QGauss<boundarydim-1> (3),
      //                                                    typename FunctionMap<dim>::type (), locally_relevant_solution,
      //                                                    estimated_error_per_cell,
      //                                                    fe.component_mask(u_extractor) | fe.component_mask(p_extractor),
      //                                                    nullptr, 0, triangulation.locally_owned_subdomain ());
      //   parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction (
      //     triangulation, estimated_error_per_cell, 0.5, 0.3);
      //
      //   // Limit refinement to min/max levels
      //   // if (triangulation.n_levels () > ( 5 ))
      //   //   for (typename Triangulation<boundarydim, dim>::active_cell_iterator cell =
      //   //          triangulation.begin_active ( 5 );
      //   //        cell != triangulation.end (); ++cell)
      //   //     cell->clear_refine_flag ();
      //   // for (typename Triangulation<boundarydim, dim>::active_cell_iterator cell =
      //   //        triangulation.begin_active ( 4 );
      //   //      cell != triangulation.end_active ( 4 ); ++cell)
      //   //   cell->clear_coarsen_flag ();
      //
      //   // Transfer solution onto new mesh
      //   std::vector<const TrilinosWrappers::MPI::Vector *> solution (1);
      //   solution[0] = &locally_relevant_solution;
      //   triangulation.prepare_coarsening_and_refinement ();
      //   solutionTx.prepare_for_coarsening_and_refinement (solution);
      //
      //   triangulation.execute_coarsening_and_refinement ();
    // }
    //
    // setup_dofs ();
    //
    // {
    //   TrilinosWrappers::MPI::Vector distributed_solution (system_rhs);
    //   std::vector<TrilinosWrappers::MPI::Vector *> tmp (1);
    //   tmp[0] = &(distributed_solution);
    //   solutionTx.interpolate (tmp);
    //   constraints.distribute (distributed_solution);
    //   locally_relevant_solution = distributed_solution;
    // }
  }

  template <int dim>
  void
  CrustalFlow<dim>::
  output_results (const unsigned int timestep,
                  const double time)
  {
    const std::string output_directory = this->get_output_directory();

    DataOut<boundarydim,DoFHandler<boundarydim,dim>> data_out;
    data_out.attach_dof_handler (dof_handler);

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(0);

    for (unsigned int i=0; i<dim; ++i)
      {
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
      }
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    std::vector<std::string> solution_name(0);
    for (unsigned int i=0; i<dim; ++i)
      {
        solution_name.push_back("Velocity");
      }
    solution_name.push_back ("Pressure");
    solution_name.push_back ("Crustal_Thickness");
    solution_name.push_back ("Sill_Thickness");

    data_out.add_data_vector (locally_relevant_solution, solution_name,
                              DataOut<boundarydim,DoFHandler<boundarydim,dim>>::type_dof_data,
                              data_component_interpretation);

    data_out.build_patches ();
    std::ofstream output (
      (output_directory + "/crustal_flow-" + Utilities::int_to_string (timestep, 5) + "."
       + Utilities::int_to_string (triangulation.locally_owned_subdomain (), 4)
       + ".vtu").c_str ());
    data_out.write_vtu (output);
    if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes (MPI_COMM_WORLD);
             ++i)
          {
            filenames.push_back ("crustal_flow-"
                                 + Utilities::int_to_string (timestep, 5) + "."
                                 + Utilities::int_to_string (i, 4) + ".vtu");
          }

        const std::string pvtu_master_filename =
          output_directory + "/crustal_flow-"
          + Utilities::int_to_string (timestep, 4) + ".pvtu";
        std::ofstream pvtu_master (pvtu_master_filename);
        data_out.write_pvtu_record (pvtu_master, filenames);

        static std::vector<std::pair<double, std::string>> times_and_names;
        times_and_names.push_back (std::pair<double, std::string> (time, pvtu_master_filename));
        std::ofstream pvd_output ("crustal_flow.pvd");
        DataOutBase::write_pvd_record (pvd_output, times_and_names);
        std::ofstream pvd_output2 (output_directory+".pvd");
        DataOutBase::write_pvd_record (pvd_output2, times_and_names);
      }
  }

  template <int dim>
  double
  CrustalFlow<dim>::
  get_dt()
  {
    const unsigned int velocity_degree = 2;
    const QIterated<boundarydim> quadrature_formula (QTrapez<1> (), velocity_degree);
    const unsigned int n_q_points = quadrature_formula.size ();
    FEValues<boundarydim,dim> fe_values (fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim> > velocity_values (n_q_points);
    double max_local_cfl = 0;

    typename DoFHandler<boundarydim,dim>::active_cell_iterator cell =
      dof_handler.begin_active (), endc = dof_handler.end ();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned ())
        {
          fe_values.reinit (cell);
          fe_values[u_extractor].get_function_values (locally_relevant_solution,
                                                      velocity_values);
          double max_local_velocity = 1e-10;
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              max_local_velocity = std::max (max_local_velocity,
                                             velocity_values[q].norm ());
            }
          max_local_cfl = std::max (max_local_cfl,
                                    max_local_velocity / cell->diameter ());
        }
    const double CFL = Utilities::MPI::max (max_local_cfl, MPI_COMM_WORLD);

    const double MAX_DT = 1.0;
    return std::min (MAX_DT, 0.1 / CFL);
  }
}
}

// explicit instantiations
namespace aspect
{
  namespace BoundaryTraction
  {
    ASPECT_REGISTER_BOUNDARY_TRACTION_MODEL(CrustalFlow, "Crustal flow",
                                            "Implementation of a model in which the boundary "
                                            "traction is given in terms of ")
  }
}
