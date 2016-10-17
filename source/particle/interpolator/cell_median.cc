/*
  Copyright (C) 2016 by the authors of the ASPECT code.

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
 along with ASPECT; see the file doc/COPYING.  If not see
 <http://www.gnu.org/licenses/>.
 */

#include <aspect/particle/interpolator/cell_median.h>
#include <aspect/postprocess/tracers.h>
#include <aspect/simulator.h>

#include <deal.II/grid/grid_tools.h>

namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      template <int dim>
      std::vector<std::vector<double> >
      CellMedian<dim>::properties_at_points(const std::multimap<types::LevelInd, Particle<dim> > &particles,
                                             const std::vector<Point<dim> > &positions,
                                             const typename parallel::distributed::Triangulation<dim>::active_cell_iterator &cell) const
      {
        const Postprocess::Tracers<dim> *tracer_postprocessor = this->template find_postprocessor<Postprocess::Tracers<dim> >();

        const std::multimap<aspect::Particle::types::LevelInd, aspect::Particle::Particle<dim> > &ghost_particles =
          tracer_postprocessor->get_particle_world().get_ghost_particles();

        typename parallel::distributed::Triangulation<dim>::active_cell_iterator found_cell;

        if (cell == typename parallel::distributed::Triangulation<dim>::active_cell_iterator())
          {
            // We can not simply use one of the points as input for find_active_cell_around_point
            // because for vertices of mesh cells we might end up getting ghost_cells as return value
            // instead of the local active cell. So make sure we are well in the inside of a cell.
            Assert(positions.size() > 0,
                   ExcMessage("The particle property interpolator was not given any "
                              "positions to evaluate the particle properties at."));

            const Point<dim> approximated_cell_midpoint = std::accumulate (positions.begin(), positions.end(), Point<dim>())
                                                          / static_cast<double> (positions.size());

            found_cell =
              (GridTools::find_active_cell_around_point<> (this->get_mapping(),
                                                           this->get_triangulation(),
                                                           approximated_cell_midpoint)).first;
          }
        else
          found_cell = cell;

        const types::LevelInd cell_index = std::make_pair(found_cell->level(),found_cell->index());

        const std::pair<typename std::multimap<types::LevelInd, Particle<dim> >::const_iterator,
              typename std::multimap<types::LevelInd, Particle<dim> >::const_iterator> particle_range =
                (cell->is_locally_owned())
                ?
                particles.equal_range(cell_index)
                :
                ghost_particles.equal_range(cell_index);

        const unsigned int n_particles = std::distance(particle_range.first,particle_range.second);
        const unsigned int n_properties = particles.begin()->second.get_properties().size();
        std::vector<double> cell_medians (n_properties, 0.0);

        int j=0;
        if (n_particles > 0)
          {
            std::vector<std::vector<double> > cell_properties (n_properties, std::vector<double> (n_particles));

            for (typename std::multimap<types::LevelInd, Particle<dim> >::const_iterator particle = particle_range.first;
                 particle != particle_range.second; ++particle, ++j)
              {
                const std::vector<double> &particle_properties = particle->second.get_properties();

                for (unsigned int i = 0; i < n_properties; ++i)
                  cell_properties[i][j] = particle_properties[i];
              }

            const size_t midpoint = n_particles/2;

            for (unsigned int i = 0; i < n_properties; ++i)
              {
                std::nth_element(cell_properties[i].begin(),
                                 cell_properties[i].begin() + midpoint,
                                 cell_properties[i].end());
                cell_medians[i] = cell_properties[i][midpoint];
              }
          }
        // If there are no particles in this cell use the median of the
        // neighboring cells.
        else
          {
            std::vector<std::vector<double> > cell_properties (n_properties, std::vector<double> ());

            std::vector<typename parallel::distributed::Triangulation<dim>::active_cell_iterator> neighbors;
            GridTools::get_active_neighbors<parallel::distributed::Triangulation<dim> >(found_cell,neighbors);

            unsigned int non_empty_neighbors = 0;
            for (unsigned int i=0; i<neighbors.size(); ++i)
              {
                // Only recursively call this function if the neighbor cell contains
                // particles (else we end up in an endless recursion)
                if ((neighbors[i]->is_locally_owned())
                    && (particles.count(std::make_pair(neighbors[i]->level(),neighbors[i]->index())) == 0))
                  continue;
                else if ((!neighbors[i]->is_locally_owned())
                         && (ghost_particles.count(std::make_pair(neighbors[i]->level(),neighbors[i]->index())) == 0))
                  continue;

                std::vector<double> neighbor_properties = properties_at_points(particles,
                                                                               std::vector<Point<dim> > (1,neighbors[i]->center(true,false)),
                                                                               neighbors[i])[0];

                for (unsigned int i = 0; i < n_properties; ++i)
                  cell_properties[i].push_back(neighbor_properties[i]);

                ++non_empty_neighbors;
              }

            AssertThrow(non_empty_neighbors != 0,
                        ExcMessage("A cell and all of its neighbors do not contain any particles. "
                                   "The 'cell median' interpolation scheme does not support this case."));

            for (unsigned int i = 0; i < n_properties; ++i)
              {
                size_t midpoint = cell_properties[i].size() / 2;
                std::nth_element(cell_properties[i].begin(),
                                 cell_properties[i].begin() + midpoint,
                                 cell_properties[i].end());
                cell_medians[i] = cell_properties[i][midpoint];
              }
          }

        return std::vector<std::vector<double> > (positions.size(),cell_medians);
      }
    }
  }
}


// explicit instantiations
namespace aspect
{
  namespace Particle
  {
    namespace Interpolator
    {
      ASPECT_REGISTER_PARTICLE_INTERPOLATOR(CellMedian,
                                            "cell median",
                                            "Return the median of all tracer properties in the given cell.")
    }
  }
}
