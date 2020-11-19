# Copyright (C) 2020 by the authors of the ASPECT code.
#
# This file is part of ASPECT.
#
# ASPECT is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# ASPECT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ASPECT; see the file LICENSE.  If not see
# <http://www.gnu.org/licenses/>.

SET(LIBTCC_DIR "" CACHE PATH "An optional hint to a LIBTCC installation")
SET_IF_EMPTY(LIBTCC_DIR "$ENV{LIBTCC_DIR}")

FIND_PATH(LIBTCC_INCLUDE_DIR
  NAMES libtcc.h
  HINTS ${LIBTCC_DIR}
  PATH_SUFFIXES libtcc include
        )

FIND_LIBRARY(LIBTCC_LIBRARY
  NAMES tcc
  HINTS ${LIBTCC_DIR}
  PATH_SUFFIXES lib${LIB_SUFFIX} lib64 lib
  )

# SET(_header "${LIBTCC_INCLUDE_DIR}/libtcc.h")

SET(LIBTCC_VERSION "unknown")


IF(LIBTCC_INCLUDE_DIR AND LIBTCC_LIBRARY)
  SET(LIBTCC_FOUND TRUE)
  SET(LIBTCC_LIBRARIES ${LIBTCC_LIBRARY})
  SET(LIBTCC_INCLUDE_DIRS ${LIBTCC_INCLUDE_DIR})
ELSE()
  SET(LIBTCC_FOUND FALSE)
ENDIF()