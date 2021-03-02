      SUBROUTINE OGCSCA (SCACON)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE CONSTRAINT + MERIT CONVERGENCE THRESHOLDS
C ======================================================================
C INP | SCACON(NUMCON+1) | R*8 | CONSTRAINTS CONVER THRESHOLD (1:NUMCON)
C     |                  |     | MERIT       CONVER THRESHOLD (1+NUMCON)
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
#include 'ogdata.inc'
C ======================================================================
      REAL*8       SCACON(NUMCON+1)
C ======================================================================
      INTEGER*4    CON
C ======================================================================
      DO CON = 1, NUMCON + 1
         CONSCA(CON) = SCACON(CON)
      ENDDO
C ======================================================================
      RETURN
      END
