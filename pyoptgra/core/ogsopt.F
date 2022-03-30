      SUBROUTINE OGSOPT (OPTSEN)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C LINEAR OPTIMISATION MODE
C ======================================================================
C INP | OPTSEN           | I*4 | SENSITIVITY OPTIMISATION MODE
C     |                  |     | ->  0: NO
C     |                  |     | -> -1: INITIALISATION
C     |                  |     | -> +1: WITH CONSTRAINT CALCULATION
C     |                  |     | -> +2: WITH CONSTRAINT BIAS
C     |                  |     | -> +3: WITH CONSTRAINT CALC / NO OPTIM
C     |                  |     | -> +4: WITH CONSTRAINT BIAS / NO OPTIM
C ======================================================================
C 2021/03/30 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   OPTSEN
C ======================================================================
      SENOPT = OPTSEN
C ======================================================================
      RETURN
      END
