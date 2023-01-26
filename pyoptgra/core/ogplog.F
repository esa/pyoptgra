      SUBROUTINE OGPLOG (LUPLOG, BOSVER)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE WRITING IN PYGMO LOG FORMAT
C ======================================================================
C INP | LUPLOG           | I*4 | LOGICAL UNIT FOR WRITING PYGMO LOG
C ----------------------------------------------------------------------
C INP | BOSVER           | I*4 | VERBOSITY LEVEL
C     |                  |     | -> 0=NO OUTPUT
C     |                  |     | -> 1 OUTPUT EVERY ITERATION
C     |                  |     | -> 2 OUTPUT EVERY 2ND ITERATION
C     |                  |     | -> N OUTPUT EVERY NTH ITERATION
C ======================================================================
C 2023/01/25 | W. MARTENS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
C Yes, we stay true to the original vintage F77 style with our
C variable names just to confuse future developers :P
      INTEGER(4)   LUPLOG
      INTEGER(4)   BOSVER
C ======================================================================
      LOGLUP = LUPLOG
      VERBOS = BOSVER
      FEVALS = 0 ! initialize number of cost fun evaluations
      PYGFLA = 0  ! pygmo output status flag: 0: continue iterating, 1: final output
C ======================================================================
      RETURN
      END
