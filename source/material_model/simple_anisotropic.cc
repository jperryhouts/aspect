/*
  Copyright (C) 2011 - 2015 by the authors of the ASPECT code.

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


#include <aspect/material_model/simple_anisotropic.h>

using namespace dealii;

namespace aspect
{
  namespace MaterialModel
  {

    template <int dim>
    void
    SimpleAnisotropic<dim>::
    evaluate(const MaterialModelInputs<dim> &in,
             MaterialModelOutputs<dim> &out) const
    {
      Simple<dim>::evaluate (in, out);
      for (unsigned int i=0; i < in.position.size(); ++i)
        out.consitutive_tensors[i] = C;
    }


    template <int dim>
    void
    SimpleAnisotropic<dim>::declare_parameters (ParameterHandler &prm)
    {

      Simple<dim>::declare_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Simple anisotropic model");
        {
          if (dim == 2)
            prm.declare_entry ("Viscosity tensor",
                               "1, 0, 0,"
                               "0, 1, 0,"
                               "0, 0,.5",
                               Patterns::List(Patterns::Double()),
                               "Viscosity-scaling tensor in Voigt notation.");
          else
            prm.declare_entry ("Viscosity tensor",
                               "1, 0, 0, 0, 0, 0,"
                               "0, 1, 0, 0, 0, 0,"
                               "0, 0, 1, 0, 0, 0,"
                               "0, 0, 0,.5, 0, 0,"
                               "0, 0, 0, 0,.5, 0,"
                               "0, 0, 0, 0, 0,.5",
                               Patterns::List(Patterns::Double()),
                               "Viscosity-scaling tensor in Voigt notation.");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }



    template <int dim>
    void
    SimpleAnisotropic<dim>::parse_parameters (ParameterHandler &prm)
    {

      Simple<dim>::parse_parameters (prm);

      prm.enter_subsection("Material model");
      {
        prm.enter_subsection("Simple anisotropic model");
        {
          const int size_voigt = (dim == 3 ? 6 : 3);
          const int n_constitutive_components = size_voigt*size_voigt;
          const std::vector<double> tmp_tensor =
            Utilities::string_to_double(Utilities::split_string_list(prm.get ("Viscosity tensor")));
          Assert(tmp_tensor.size() == n_constitutive_components,
                 ExcMessage("Constitutive voigt matrix must have 9 components in 2D, or 36 components in 3d"));

          std::vector<std::vector<double> > voigt_visc_tensor (size_voigt);
          for (unsigned int i=0; i<size_voigt; ++i)
            {
              voigt_visc_tensor[i].resize(size_voigt);
              for (unsigned int j=0; j<size_voigt; ++j)
                voigt_visc_tensor[i][j] = tmp_tensor[i*size_voigt+j];
            }

          // Voigt indices (For mapping back to real tensor)
          const unsigned int vi3d0[] = {0, 1, 2, 1, 0, 0};
          const unsigned int vi3d1[] = {0, 1, 2, 2, 2, 1};
          const unsigned int vi2d0[] = {0, 1, 0};
          const unsigned int vi2d1[] = {0, 1, 1};

          // Fill the constitutive tensor with values from the Voigt tensor
          for (unsigned int i=0; i<size_voigt; ++i)
            for (unsigned int j=0; j<size_voigt; ++j)
              if (dim == 2)
                C[vi2d0[i]][vi2d1[i]][vi2d0[j]][vi2d1[j]] = voigt_visc_tensor[i][j];
              else
                C[vi3d0[i]][vi3d1[i]][vi3d0[j]][vi3d1[j]] = voigt_visc_tensor[i][j];
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  }
}

// explicit instantiations
namespace aspect
{
  namespace MaterialModel
  {
    ASPECT_REGISTER_MATERIAL_MODEL(SimpleAnisotropic,
                                   "simple anisotropic",
                                   "A material model that has constant values "
                                   "for all coefficients but the density and viscosity. The defaults for all "
                                   "coefficients are chosen to be similar to what is believed to be correct "
                                   "for Earth's mantle. All of the values that define this model are read "
                                   "from a section ``Material model/Simple anisotropic model'' in the input file, see "
                                   "Section~\\ref{parameters:Material_20model/Simple_20model}."
                                   "\n\n"
                                   "This model uses the following set of equations for the two coefficients that "
                                   "are non-constant: "
                                   "\\begin{align}"
                                   "  \\eta(p,T,\\mathfrak c) &= \\tau(T) \\zeta(\\mathfrak c) \\eta_0, \\\\"
                                   "  \\rho(p,T,\\mathfrak c) &= \\left(1-\\alpha (T-T_0)\\right)\\rho_0 + \\Delta\\rho \\; c_0,"
                                   "\\end{align}"
                                   "where $c_0$ is the first component of the compositional vector "
                                   "$\\mathfrak c$ if the model uses compositional fields, or zero otherwise. "
                                   "\n\n"
                                   "The temperature pre-factor for the viscosity formula above is "
                                   "defined as "
                                   "\\begin{align}"
                                   "  \\tau(T) &= H\\left(e^{\\beta (T-T_0)/T_0}\\right),"
                                   "  \\qquad\\qquad H(x) = \\begin{cases}"
                                   "                            10^{-2} & \\text{if}\\; x<10^{-2}, \\\\"
                                   "                            x & \\text{if}\\; 10^{-2}\\le x \\le 10^2, \\\\"
                                   "                            10^{2} & \\text{if}\\; x>10^{2}, \\\\"
                                   "                         \\end{cases}"
                                   "\\end{align} "
                                   "where $\\beta$ corresponds to the input parameter ``Thermal viscosity exponent'' "
                                   "and $T_0$ to the parameter ``Reference temperature''. If you set $T_0=0$ "
                                   "in the input file, the thermal pre-factor $\\tau(T)=1$."
                                   "\n\n"
                                   "The compositional pre-factor for the viscosity is defined as "
                                   "\\begin{align}"
                                   "  \\zeta(\\mathfrak c) &= \\xi^{c_0}"
                                   "\\end{align} "
                                   "if the model has compositional fields and equals one otherwise. $\\xi$ "
                                   "corresponds to the parameter ``Composition viscosity prefactor'' in the "
                                   "input file."
                                   "\n\n"
                                   "Finally, in the formula for the density, $\\Delta\\rho$ "
                                   "corresponds to the parameter ``Density differential for compositional field 1''."
                                   "\n\n"
                                   "Note that this model uses the formulation that assumes an incompressible "
                                   "medium despite the fact that the density follows the law "
                                   "$\\rho(T)=\\rho_0(1-\\beta(T-T_{\\text{ref}}))$. "
                                   "\n\n"
                                   "\\note{Despite its name, this material model is not exactly ``simple'', "
                                   "as indicated by the formulas above. While it was originally intended "
                                   "to be simple, it has over time acquired all sorts of temperature "
                                   "and compositional dependencies that weren't initially intended. "
                                   "Consequently, there is now a ``simpler'' material model that now fills "
                                   "the role the current model was originally intended to fill.}")
  }
}
