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
          FE_Q<boundarydim, dim> (1), 1 /* Crustal thickness */,
          FE_Q<boundarydim, dim> (2), 1 /* Elastic plate deflection */,
          FE_Q<boundarydim, dim> (1), 1 /* Overburden load */),
      u_extractor (0),
      h_extractor (boundarydim),
      w_extractor (boundarydim+1),
      s_extractor (boundarydim+2)
    {}


    template <int dim>
    void
    CrustalFlow<dim>::initialize ()
    {
      surface_boundary_id =
        this->get_geometry_model().translate_symbolic_boundary_name_to_id("top");
    }


    template <int dim>
    void
    CrustalFlow<dim>::update()
    {
      std::set<types::boundary_id> boundary_ids;
      boundary_ids.insert(surface_boundary_id);
      GridGenerator::extract_boundary_mesh(this->get_triangulation(),
                                           triangulation,
                                           boundary_ids);

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
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    assemble_system(const double)
    {
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    solve()
    {
    }

    template <int dim>
    void
    CrustalFlow<dim>::
    refine_mesh()
    {
    }

    template <int dim>
    double
    CrustalFlow<dim>::
    get_dt()
    {
      return 1000 * 3.14e7;
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
