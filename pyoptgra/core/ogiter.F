      SUBROUTINE OGITER (ITEMAX, ITECOR, ITEOPT, ITEDIV, ITECNV)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE OPTIMISATION CONTROL PARAMETERS
C ======================================================================
C INP | ITEMAX           | I*4 | MAXIMUM NUMBER OF ITERATIONS
C ----------------------------------------------------------------------
C INP | MAXVAR           | R*8 | MAXIMUM DISANCE PER ITERATION
C     |                  |     | -> SCALED
C ----------------------------------------------------------------------
C INP | SNDVAR           | R*8 | PERTURBATION FOR SND ORDER DERIVATIVES
C     |                  |     | -> SCALED
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   ITEMAX, ITECOR, ITEOPT, ITEDIV, ITECNV
      REAL(8)      MAXVAR
      REAL(8)      SNDVAR
      CHARACTER    STR*256
C ======================================================================
      MAXITE = ITEMAX
      CORITE = ITECOR
      OPTITE = ITEOPT
      DIVITE = ITEDIV
      CNVITE = ITECNV
      IF (CORITE .GT. MAXITE) CORITE = MAXITE
      IF (OPTITE .GT. MAXITE) OPTITE = MAXITE
      IF (DIVITE .GT. CORITE) DIVITE = CORITE
      IF (CNVITE .GT. OPTITE) CNVITE = OPTITE
C ======================================================================
      RETURN
      END
