      SUBROUTINE OGDIST (MAXVAR, SNDVAR)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE OPTIMISATION CONTROL PARAMETERS
C ======================================================================
C INP | MAXVAR           | R*8 | MAXIMUM DISTANCE PER ITERATION
C     |                  |     | -> SCALED
C ----------------------------------------------------------------------
C INP | SNDVAR           | R*8 | PERTURBATION FOR SND ORDER DERIVATIVES
C     |                  |     | -> SCALED
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
      CHARACTER*160 SCCS_INFO(2)
      DATA SCCS_INFO
     &/'%Z%%P% ',
     &'%I%      %E%     MAN_FDD\n'/
C ======================================================================
#include "ogdata.inc"
C ======================================================================
      REAL*8       MAXVAR
      REAL*8       SNDVAR
C ======================================================================
      VARMAX = MAXVAR
      VARSND = SNDVAR
C ======================================================================
      RETURN
      END