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


#include <aspect/postprocess/crustal_flow.h>
#include <aspect/utilities.h>
#include <aspect/global.h>

namespace aspect
{
  namespace Postprocess
  {
    template <int spacedim>
    CrustalFlow<spacedim>::CrustalFlow ()
      :
      triangulation (MPI_COMM_WORLD,
                     typename Triangulation<dim, spacedim>::MeshSmoothing (
                       Triangulation<dim, spacedim>::smoothing_on_refinement | Triangulation<dim, spacedim>::smoothing_on_coarsening),
                       parallel::distributed::Triangulation<dim, spacedim>::mesh_reconstruction_after_repartitioning),
      flow_dof_handler (triangulation),
      flow_fe (FE_Q<dim, spacedim> (2), dim /* Crustal flow velocity */,
          FE_Q<dim, spacedim> (1), 1 /* P -- Crustal thickness */,
          FE_Q<dim, spacedim> (1), 1 /* h -- Elastic plate deflection */,
          FE_Q<dim, spacedim> (1), 1 /* s -- Overburden load */),
      u_extractor (0),
      p_extractor (dim),
      h_extractor (dim+1),
      s_extractor (dim+2),
      flexure_dof_handler (triangulation),
      flexure_fe (FE_Q<dim, spacedim> (2), 1),
      w_extractor (0)
    {}

    template <int spacedim>
    CrustalFlow<spacedim>::~CrustalFlow ()
    {
      flow_dof_handler.clear ();
      flexure_dof_handler.clear ();
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::initialize ()
    {
      surface_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");

      const std::string vis_directory = this->get_output_directory() + "/crustal_flow";
      Utilities::create_directory (vis_directory,
                                   this->get_mpi_communicator(),
                                   true);

      {
        GridGenerator::hyper_sphere(triangulation);
        triangulation.refine_global(3);

        // std::set<types::boundary_id> boundary_ids;
        // boundary_ids.insert(surface_boundary_id);
        // GridGenerator::extract_boundary_mesh(this->get_triangulation(),
        //                                      triangulation,
        //                                      boundary_ids);

        setup_flow_dofs ();
        setup_flexure_dofs ();

        for (unsigned int i=0; i<2; ++i)
        {
          assemble_flexure_system ();
          solve_flexure ();

          assemble_flow_system (0);
          solve_flow ();
          refine_mesh ();
        }
      }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::update()
    {}


    // template <int spacedim>
    // Tensor<1,spacedim>
    // CrustalFlow<spacedim>::
    // boundary_traction (const types::boundary_id,
    //                    const Point<spacedim> &,
    //                    const Tensor<1,spacedim> &normal_vector) const
    template <int spacedim>
    std::pair<std::string,std::string>
    CrustalFlow<spacedim>::
    execute (TableHandler &)
    {
      double geodynamic_timestep = this->get_timestep();
      double dt = 0;
      int crustal_flow_timestep = 0;
      double crustal_flow_time = 0;
      do
        {
          // This might only work after solve_flow() has happened at least
          // once. Which it has been, if there is any initial refinement.
          dt = get_dt (geodynamic_timestep-crustal_flow_time);

          assemble_flexure_system ();
          solve_flexure ();

          assemble_flow_system (dt);
          solve_flow ();

          crustal_flow_time += dt;
          crustal_flow_timestep += 1;
        }
      while (crustal_flow_time < geodynamic_timestep);

      output_results (this->get_timestep_number(), this->get_time());

      refine_mesh ();

      return std::make_pair (std::string ("Crustal flow timesteps"),
                             Utilities::int_to_string(crustal_flow_timestep));
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    setup_flexure_dofs ()
    {
      flexure_dof_handler.distribute_dofs (flexure_fe);
      {
        locally_owned_flexure_dofs = flexure_dof_handler.locally_owned_dofs ();
        DoFTools::extract_locally_relevant_dofs (flexure_dof_handler,
                    locally_relevant_flexure_dofs);
      }

      {
        flexure_constraints.clear ();
        flexure_constraints.reinit (locally_relevant_flexure_dofs);
        DoFTools::make_hanging_node_constraints (flexure_dof_handler,
          flexure_constraints);
        VectorTools::interpolate_boundary_values (flexure_dof_handler, 0,
                                                  Functions::ZeroFunction<spacedim>(1),
                                                  flexure_constraints,
                                                  flexure_fe.component_mask (w_extractor));
        flexure_constraints.close ();
      }

      {
        TrilinosWrappers::SparsityPattern dsp(locally_owned_flexure_dofs,
                                              locally_owned_flexure_dofs,
                                              locally_relevant_flexure_dofs,
                                              MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(flexure_dof_handler,dsp,flexure_constraints);
        dsp.compress();
        flexure_matrix.reinit(dsp);

        // DynamicSparsityPattern dsp (locally_relevant_flow_dofs);
        // DoFTools::make_sparsity_pattern (flow_dof_handler, dsp, flow_constraints, false);
        // SparsityTools::distribute_sparsity_pattern (dsp,
        //                                             flow_dof_handler.n_locally_owned_dofs_per_processor (),
        //                                             MPI_COMM_WORLD,
        //                                             locally_relevant_flow_dofs);
        // flow_matrix.reinit (locally_owned_flow_dofs, locally_owned_flow_dofs, dsp, MPI_COMM_WORLD);

        flexure_rhs.reinit (locally_owned_flexure_dofs, MPI_COMM_WORLD);
        locally_relevant_flexure_solution.reinit (locally_owned_flexure_dofs,
                                                  locally_relevant_flexure_dofs,
                                                  MPI_COMM_WORLD);
      }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    setup_flow_dofs()
    {
      flow_dof_handler.distribute_dofs (flow_fe);
      {
        locally_owned_flow_dofs = flow_dof_handler.locally_owned_dofs ();
        DoFTools::extract_locally_relevant_dofs (flow_dof_handler,
                                                 locally_relevant_flow_dofs);
      }

      {
        flow_constraints.clear ();
        flow_constraints.reinit (locally_relevant_flow_dofs);
        DoFTools::make_hanging_node_constraints (flow_dof_handler, flow_constraints);
        VectorTools::interpolate_boundary_values (flow_dof_handler, 0,
                                                  Functions::ZeroFunction<spacedim>(5),
                                                  flow_constraints,
                                                  flow_fe.component_mask(p_extractor)
                                                  | flow_fe.component_mask(h_extractor)
                                                  | flow_fe.component_mask(s_extractor));

        std::vector<Point<spacedim> >  constraint_locations;
        std::vector<unsigned int> constraint_component_indices;
        std::vector<double>       constraint_values;

        for (unsigned int i = 0; i < constraint_locations.size(); ++i)
          {
            const Point<spacedim> constrain_where = constraint_locations[i]; // TODO
            const unsigned int constraint_component_index = constraint_component_indices[i];
            const double constraint_value = constraint_values[i];

            // Find nearest quadrature point.
            double min_local_distance = std::numeric_limits<double>::max();
            unsigned int local_best_dof_index;

            const std::vector<Point<dim> > points = flow_fe.get_unit_support_points();
            const Quadrature<dim> quadrature (points);
            FEValues<dim, spacedim> flow_fe_values (flow_fe, quadrature, update_quadrature_points);
            typename DoFHandler<dim,spacedim>::active_cell_iterator cell;
            for (cell = flow_dof_handler.begin_active(); cell != flow_dof_handler.end(); ++cell)
              {
                if (! cell->is_artificial())
                  {
                    flow_fe_values.reinit(cell);
                    std::vector<unsigned int> local_dof_indices(flow_fe.dofs_per_cell);
                    cell->get_dof_indices (local_dof_indices);

                    for (unsigned int q=0; q<quadrature.size(); q++)
                      {
                        // If it's okay to constrain this DOF
                        if (flow_constraints.can_store_line(local_dof_indices[q]) &&
                            !flow_constraints.is_constrained(local_dof_indices[q]))
                          {
                            const unsigned int c_idx = flow_fe.system_to_component_index(q).first;
                            if (c_idx == constraint_component_index)
                              {
                                const Point<spacedim> p = flow_fe_values.quadrature_point(q);
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
                flow_constraints.add_line (local_best_dof_index);
                flow_constraints.set_inhomogeneity (local_best_dof_index, constraint_value);
              }
          }
        flow_constraints.close ();
      }

      {
        TrilinosWrappers::SparsityPattern dsp(locally_owned_flow_dofs,
                                              locally_owned_flow_dofs,
                                              locally_relevant_flow_dofs,
                                              MPI_COMM_WORLD);
        DoFTools::make_sparsity_pattern(flow_dof_handler,dsp,flow_constraints);
        dsp.compress();
        flow_matrix.reinit(dsp);

        // DynamicSparsityPattern dsp (locally_relevant_flow_dofs);
        // DoFTools::make_sparsity_pattern (flow_dof_handler, dsp, flow_constraints, false);
        // SparsityTools::distribute_sparsity_pattern (dsp,
        //                                             flow_dof_handler.n_locally_owned_dofs_per_processor (),
        //                                             MPI_COMM_WORLD,
        //                                             locally_relevant_flow_dofs);
        // flow_matrix.reinit (locally_owned_flow_dofs, locally_owned_flow_dofs, dsp, MPI_COMM_WORLD);

        flow_rhs.reinit (locally_owned_flow_dofs, MPI_COMM_WORLD);
        locally_relevant_flow_solution.reinit (locally_owned_flow_dofs, locally_relevant_flow_dofs,
                                          MPI_COMM_WORLD);
      }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    assemble_flexure_system ()
    {
      const QGauss<dim> quadrature_formula (1);
      FEValues<dim,spacedim> flexure_fe_values (flexure_fe, quadrature_formula,
                                           update_values | update_JxW_values
                                           | update_hessians
                                           | update_quadrature_points);
     FEValues<dim,spacedim> flow_fe_values (flow_fe, quadrature_formula,
                                            update_values);
      const unsigned int dofs_per_cell = flexure_fe.dofs_per_cell;
      const unsigned int n_q_points = quadrature_formula.size ();
      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs (dofs_per_cell);
      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<double> phi_w (dofs_per_cell);
      std::vector<Tensor<2,spacedim> > hessian_phi_w (dofs_per_cell);
      std::vector<double> latest_h_values (n_q_points);
      std::vector<double> latest_s_values (n_q_points);

      typename DoFHandler<dim,spacedim>::active_cell_iterator cell =
        flexure_dof_handler.begin_active (), endc = flexure_dof_handler.end ();
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned ())
            {
              cell_matrix = 0;
              cell_rhs = 0;
              flexure_fe_values.reinit (cell);
              flow_fe_values.reinit (cell);

              flow_fe_values[h_extractor].get_function_values (
                locally_relevant_flow_solution, latest_h_values);
              flow_fe_values[s_extractor].get_function_values (
                locally_relevant_flow_solution, latest_s_values);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  //Point<spacedim> loc = flexure_fe_values.quadrature_point (q);
                  for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                      phi_w[k] = flexure_fe_values[w_extractor].value (k,q);
                      hessian_phi_w[k] = flexure_fe_values[w_extractor].hessian (k,q);
                    }

                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      // const Tensor<2,spacedim> hessian_i =
                      //   flexure_fe_values.shape_hessian(i, q);
                      for (unsigned int j = 0; j < dofs_per_cell; ++j)
                        {
                          // const Tensor<2,spacedim> hessian_i =
                          //   flexure_fe_values.shape_hessian(j, q);
                          cell_matrix (i, j) += (
                            //RIGIDITY * scalar_product(hessian_i, hessian_j)
                            RIGIDITY * scalar_product(hessian_phi_w[i],
                                                      hessian_phi_w[j])
                          ) * flexure_fe_values.JxW (q);
                        }

                      cell_rhs (i) += (
                        phi_w[i] * 1.0
                      ) * flexure_fe_values.JxW (q);
                    }
                }
            }
        }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    assemble_flow_system(const double dt)
    {
      const QGauss<dim> quadrature_formula (5);
      FEValues<dim,spacedim> flow_fe_values (flow_fe, quadrature_formula,
                                           update_values | update_JxW_values | update_gradients
                                           | update_quadrature_points);
      const unsigned int dofs_per_cell = flow_fe.dofs_per_cell;
      const unsigned int n_q_points = quadrature_formula.size ();
      FullMatrix<double> cell_matrix (dofs_per_cell, dofs_per_cell);
      Vector<double> cell_rhs (dofs_per_cell);

      std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

      std::vector<Tensor<1,spacedim> > phi_u (dofs_per_cell);
      std::vector<double> phi_p (dofs_per_cell);
      std::vector<double> phi_h (dofs_per_cell);
      std::vector<double> phi_s (dofs_per_cell);

      std::vector<double> div_phi_u (dofs_per_cell);

      std::vector<double> old_h_values (n_q_points);
      std::vector<double> old_s_values (n_q_points);

      typename DoFHandler<dim,spacedim>::active_cell_iterator cell =
        flow_dof_handler.begin_active (), endc = flow_dof_handler.end ();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned ())
          {
            cell_matrix = 0;
            cell_rhs = 0;
            flow_fe_values.reinit (cell);

            flow_fe_values[h_extractor].get_function_values (locally_relevant_flow_solution,
                                                        old_h_values);
            flow_fe_values[s_extractor].get_function_values (locally_relevant_flow_solution,
                                                        old_s_values);

            for (unsigned int q = 0; q < n_q_points; ++q)
              {
                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                  {
                    phi_u[k] = flow_fe_values[u_extractor].value (k,q);
                    phi_p[k] = flow_fe_values[p_extractor].value (k,q);
                    phi_h[k] = flow_fe_values[h_extractor].value (k,q);
                    phi_s[k] = flow_fe_values[s_extractor].value (k,q);
                    div_phi_u[k] = flow_fe_values[u_extractor].divergence (k,q);
                  }

                Point<spacedim> loc = flow_fe_values.quadrature_point (q);
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
                                              ) * flow_fe_values.JxW (q) ;
                      }
                    cell_rhs (i) += (

                                      phi_p[i] * sigma_zz/2.0

                                      + phi_h[i] * old_h_values[q]

                                      + phi_s[i] * (old_s_values[q] + dt*emplacement)
                                    ) * flow_fe_values.JxW (q);
                  }
              }
            cell->get_dof_indices (local_dof_indices);
            flow_constraints.distribute_local_to_global (cell_matrix, cell_rhs,
                                                    local_dof_indices, flow_matrix, flow_rhs);
          }
      flow_matrix.compress (VectorOperation::add);
      flow_rhs.compress (VectorOperation::add);
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    solve_flexure ()
    {
      dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution (
        locally_owned_flexure_dofs, MPI_COMM_WORLD);

      SolverControl cn;
      TrilinosWrappers::SolverDirect solver (cn);
      try
        {
          solver.solve (flexure_matrix, distributed_solution, flexure_rhs);
          flexure_constraints.distribute (distributed_solution);
          locally_relevant_flexure_solution = distributed_solution;
        }
      catch (const std::exception &exc)
        {
          if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            {
              AssertThrow(false, ExcMessage (
                            std::string ("The flexure solver failed with error:\n\n")
                            + exc.what ()));
            }
          else
            {
              throw QuietException ();
            }
        }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    solve_flow()
    {
      dealii::LinearAlgebraTrilinos::MPI::Vector distributed_solution (
        locally_owned_flow_dofs, MPI_COMM_WORLD);

      SolverControl cn;
      TrilinosWrappers::SolverDirect solver (cn);
      try
        {
          solver.solve (flow_matrix, distributed_solution, flow_rhs);
          flow_constraints.distribute (distributed_solution);
          locally_relevant_flow_solution = distributed_solution;
        }
      catch (const std::exception &exc)
        {
          if (Utilities::MPI::this_mpi_process (MPI_COMM_WORLD) == 0)
            {
              AssertThrow(false, ExcMessage (
                            std::string ("The flow solver failed with error:\n\n")
                            + exc.what ()));
            }
          else
            {
              throw QuietException ();
            }
        }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    refine_mesh()
    {
      parallel::distributed::SolutionTransfer<dim,
               TrilinosWrappers::MPI::Vector, DoFHandler<dim,spacedim>>
               solutionTx (flow_dof_handler);

      {
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells ());
        KellyErrorEstimator<dim, spacedim>::estimate (flow_dof_handler, QGauss<dim-1> (3),
                                                         std::map<types::boundary_id, const Function<spacedim> *>(),
                                                         locally_relevant_flow_solution,
                                                         estimated_error_per_cell,
                                                         flow_fe.component_mask(u_extractor) | flow_fe.component_mask(p_extractor),
                                                         nullptr, 0, triangulation.locally_owned_subdomain ());
        GridRefinement::refine_and_coarsen_fixed_fraction (
          triangulation, estimated_error_per_cell, 0.5, 0.3);

        // Limit refinement to min/max levels
        // if (triangulation.n_levels () > ( 5 ))
        //   for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
        //          triangulation.begin_active ( 5 );
        //        cell != triangulation.end (); ++cell)
        //     cell->clear_refine_flag ();
        // for (typename Triangulation<dim, spacedim>::active_cell_iterator cell =
        //        triangulation.begin_active ( 4 );
        //      cell != triangulation.end_active ( 4 ); ++cell)
        //   cell->clear_coarsen_flag ();

        // Transfer solution onto new mesh
        std::vector<const TrilinosWrappers::MPI::Vector *> solution (1);
        solution[0] = &locally_relevant_flow_solution;
        triangulation.prepare_coarsening_and_refinement ();
        solutionTx.prepare_for_coarsening_and_refinement (solution);

        triangulation.execute_coarsening_and_refinement ();
      }

      setup_flow_dofs ();

      {
        TrilinosWrappers::MPI::Vector distributed_solution (flow_rhs);
        std::vector<TrilinosWrappers::MPI::Vector *> tmp (1);
        tmp[0] = &(distributed_solution);
        solutionTx.interpolate (tmp);
        flow_constraints.distribute (distributed_solution);
        locally_relevant_flow_solution = distributed_solution;
      }
    }

    template <int spacedim>
    void
    CrustalFlow<spacedim>::
    output_results (const unsigned int timestep,
                    const double time)
    {
      const std::string output_directory = this->get_output_directory();
      const std::string vis_directory = output_directory + "/crustal_flow";

      // Because the problem is split into two semi-independent finite element
      // spaces, we need to make it possible to extract solution values from
      // both. First we will combine the two finite element systems into one
      // joint system.
      const FESystem<dim, spacedim> joint_fe (flow_fe, 1, flexure_fe, 1);
      DoFHandler<dim, spacedim> joint_dof_handler (triangulation);
      joint_dof_handler.distribute_dofs(joint_fe);
      Assert(joint_dof_handler.n_dofs() ==
              flow_dof_handler.n_dofs() + flexure_dof_handler.n_dofs(),
            ExcInternalError());

      IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
      DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                              locally_relevant_joint_dofs);
      TrilinosWrappers::MPI::Vector locally_relevant_joint_solution;
      locally_relevant_joint_solution.reinit(locally_relevant_joint_dofs,
                                             MPI_COMM_WORLD);

      TrilinosWrappers::MPI::Vector joint_solution;
      joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                            MPI_COMM_WORLD);

      {
        std::vector<types::global_dof_index> local_joint_dof_indices(
          joint_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_flow_dof_indices(
          flow_fe.dofs_per_cell);
        std::vector<types::global_dof_index> local_flexure_dof_indices(
          flexure_fe.dofs_per_cell);

        typename DoFHandler<dim, spacedim>::active_cell_iterator
        joint_cell = joint_dof_handler.begin_active(),
        joint_endc = joint_dof_handler.end(),
        flow_cell = flow_dof_handler.begin_active(),
        flexure_cell = flexure_dof_handler.begin_active();

        for (; joint_cell != joint_endc;
              ++joint_cell, ++flow_cell, ++flexure_cell)
        {
          if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            flow_cell->get_dof_indices(local_flow_dof_indices);
            flexure_cell->get_dof_indices(local_flexure_dof_indices);

            for (unsigned int i=0; i < joint_fe.dofs_per_cell; ++i)
            {
              if (joint_fe.system_to_base_index(i).first.first == 0)
              {
                Assert(joint_fe.system_to_base_index(i).second <
                         local_flow_dof_indices.size(),
                       ExcInternalError());
                locally_relevant_joint_solution(local_joint_dof_indices[i]) =
                  locally_relevant_flow_solution(
                    local_flow_dof_indices
                      [joint_fe.system_to_base_index(i).second]);
              }
              else
              {
                Assert(joint_fe.system_to_base_index(i).first.first == 1,
                       ExcInternalError());
                Assert(joint_fe.system_to_base_index(i).second <
                         local_flexure_dof_indices.size(),
                       ExcInternalError());
                locally_relevant_joint_solution(local_joint_dof_indices[i]) =
                  locally_relevant_flexure_solution(
                    local_flexure_dof_indices
                      [joint_fe.system_to_base_index(i).second]);
              }
            }
          }
        }
      }

      // Now proceed with data output using the joint solution space.
      DataOut<dim,DoFHandler<dim,spacedim>> data_out;
      data_out.attach_dof_handler (joint_dof_handler);

      std::vector<DataComponentInterpretation::DataComponentInterpretation> data_component_interpretation(0);
      std::vector<std::string> solution_name(0);
      for (unsigned int i=0; i<dim; ++i)
        {
          // data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
          data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
          solution_name.push_back("Velocity_"+Utilities::int_to_string(i));
        }
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

      solution_name.push_back ("Pressure");
      solution_name.push_back ("Crustal_Thickness");
      solution_name.push_back ("Sill_Thickness");
      solution_name.push_back ("Flexure");

      data_out.add_data_vector (locally_relevant_joint_solution, solution_name,
                                DataOut<dim,DoFHandler<dim,spacedim>>::type_dof_data,
                                data_component_interpretation);

      data_out.build_patches ();
      std::ofstream output (
        (vis_directory + "/crustal_flow-"
         + Utilities::int_to_string (timestep, 5) + "."
         + Utilities::int_to_string (triangulation.locally_owned_subdomain (), 4)
         + ".vtu").c_str ());
      data_out.write_vtu (output);
      if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator()) == 0)
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
            "crustal_flow/crustal_flow-"
            + Utilities::int_to_string (timestep, 4) + ".pvtu";
          std::ofstream pvtu_master (output_directory + "/" + pvtu_master_filename);
          data_out.write_pvtu_record (pvtu_master, filenames);

          static std::vector<std::pair<double, std::string>> times_and_names;
          times_and_names.push_back (std::pair<double, std::string> (time, pvtu_master_filename));
          std::ofstream pvd_output (output_directory + "/crustal_flow.pvd");
          DataOutBase::write_pvd_record (pvd_output, times_and_names);
        }
    }

    template <int spacedim>
    double
    CrustalFlow<spacedim>::
    get_dt(const double max_dt)
    {
      const unsigned int velocity_degree = 2;
      const QIterated<dim> quadrature_formula (QTrapez<1> (), velocity_degree);
      const unsigned int n_q_points = quadrature_formula.size ();
      FEValues<dim,spacedim> flow_fe_values (flow_fe, quadrature_formula, update_values);
      std::vector<Tensor<1, spacedim> > velocity_values (n_q_points);
      double max_local_cfl = 0;

      typename DoFHandler<dim,spacedim>::active_cell_iterator cell =
        flow_dof_handler.begin_active (), endc = flow_dof_handler.end ();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned ())
          {
            flow_fe_values.reinit (cell);
            flow_fe_values[u_extractor].get_function_values (locally_relevant_flow_solution,
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

      return std::min (max_dt, 0.1 / CFL);
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace Postprocess
  {
    ASPECT_REGISTER_POSTPROCESSOR(CrustalFlow, "crustal flow",
                                  "Implementation of a model in which the boundary "
                                  "traction is given in terms of ")
  }
}
