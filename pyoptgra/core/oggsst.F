      SUBROUTINE OGGSST (VARSEN, QUASEN, CONSEN, ACTSEN, DERSEN,
     &                   ACTSAV, CONSAV, REDSAV, DERSAV, ACTNUM)
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
C OUT | ACTSAV(NUMCON+1) | I*4 | STORED ACTIVE CONSTRAINTS
C OUT | CONSAV(NUMCON+4) | I*4 | STORED ACTIVE CONSTRAINTS
C OUT | REDSAV(NUMCON+3, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     |     
C OUT | DERSAV(NUMCON+3, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     |     
C OUT | ACTNUM           | I*4 | NUMBER OF ACTIVE CONSTRAINTS
C ======================================================================
C 2021/07/19 | M. von Looz | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      REAL(8)      VARSEN(NUMVAR)
      REAL(8)      QUASEN(NUMCON+1) 
      REAL(8)      CONSEN(NUMCON+1) 
      INTEGER(4)   ACTSEN(NUMCON+1)
      REAL(8)      DERSEN(NUMCON+1,NUMVAR)
      INTEGER(4)   ACTSAV(NUMCON+1)
      INTEGER(4)   CONSAV(NUMCON+4)
      REAL(8)      REDSAV(NUMCON+3,NUMVAR)
      REAL(8)      DERSAV(NUMCON+3,NUMVAR)
C ======================================================================
      INTEGER(4)    ACTNUM
      INTEGER(4)    VAR, CON
C ======================================================================
C Variable values saved for sensitivity
C ----------------------------------------------------------------------
      ACTNUM = NUMACT

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
C Temporary status saved of which constraints are active
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON+1
         ACTSAV(CON) = ACTCON(CON)
      ENDDO

      DO CON = 1, NUMCON+4
         CONSAV(CON) = CONACT(CON)
      ENDDO

      DO CON = 1, NUMCON+3
         DO VAR = 1, NUMVAR
            REDSAV(CON,VAR) = CONRED(CON,VAR)
            DERSAV(CON,VAR) = CONDER(CON,VAR)
         ENDDO
      ENDDO
C ======================================================================
      RETURN
      END