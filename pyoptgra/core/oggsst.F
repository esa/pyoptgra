      SUBROUTINE OGGSST (VARSEN, QUASEN, CONSEN, ACTSEN, DERSEN)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL SENSITIVITY ANALYSIS
C Function to get sensitivity state data, necessary for serialization.
C Do not use this directly except in serialization routines
C ======================================================================
C OUT | VARSEN(NUMVAR)   | I*4 | STORED VARIABLES VALUE
C OUT | QUASEN(NUMCON+1) | R*8 | STORED CONSTRAINTS CORRECTION VECTOR
C OUT | CONSEN(NUMCON+1) | R*8 | STORED CONSTRAINTS VALUE
C OUT | ACTSEN(NUMCON+1) | R*8 | STORED CONSTRAINTS ACTIVE
C OUT | DERSEN(NUMCON+1, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     | 
C ======================================================================
C 2008/01/16 | J. SCHOENMAEKERS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(4)      VARSEN(NUMVAR)
      REAL(8)      QUASEN(NUMCON+1) 
      REAL(8)      CONSEN(NUMCON+1) 
      INTEGER(8)   ACTSEN(NUMCON+1) 
      REAL(8)      DERSEN(NUMCON+1,NUMVAR)

C ======================================================================
      INTEGER(4)    VAR, CON
C ======================================================================
C Variable values saved for sensitivity
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         VARSEN(VAR) = SENVAR(VAR)
      ENDDO

      DO CON = 1, NUMCON+1
         QUASEN(CON) = SENQUA(CON)
         CONSEN(CON) = SENCON(CON)
         ACTSEN(CON) = SENACT(CON)
         DO VAR = 1, NUMVAR
            DERSEN(CON,VAR) = SENDER(CON,VAR)
         ENDDO
      ENDDO
C ======================================================================
      RETURN
      END