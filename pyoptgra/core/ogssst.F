      SUBROUTINE OGSSST (VARSEN, QUASEN, CONSEN, ACTSEN, DERSEN, 
     &                   ACTSAV, CONSAV, REDSAV, DERSAV, ACTNUM)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL SENSITIVITY ANALYSIS
C Function to get sensitivity state data, necessary for serialization.
C Do not use this directly except in serialization routines
C ======================================================================
C INP | VARSEN(NUMVAR)   | I*4 | STORED VARIABLES VALUE
C INP | QUASEN(NUMCON+1) | R*8 | STORED CONSTRAINTS CORRECTION VECTOR
C INP | CONSEN(NUMCON+1) | R*8 | STORED CONSTRAINTS VALUE
C INP | ACTSEN(NUMCON+1) | R*8 | STORED CONSTRAINTS ACTIVE
C INP | DERSEN(NUMCON+1, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     | 
C INP | ACTSAV(NUMCON+1) | I*4 | STORED ACTIVE CONSTRAINTS
C INP | CONSAV(NUMCON+4) | I*4 | STORED ACTIVE CONSTRAINTS
C INP | REDSAV(NUMCON+3, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     | 
C INP | DERSAV(NUMCON+3, | R*8 | STORED DERIVATIVE
C                NUMVAR) |     |     
C INP | ACTNUM           | I*4 | NUMBER OF ACTIVE CONSTRAINTS
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
      NUMACT = ACTNUM

      DO VAR = 1, NUMVAR
         SENVAR(VAR) = VARSEN(VAR)
      ENDDO

      DO CON = 1, NUMCON+1
         SENQUA(CON) = QUASEN(CON)
         SENCON(CON) = CONSEN(CON)
         SENACT(CON) = ACTSEN(CON)
         DO VAR = 1, NUMVAR
            SENDER(CON,VAR) = DERSEN(CON,VAR)
         ENDDO
      ENDDO
C ======================================================================
C Temporary status saved of which constraints are active
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON+1
         ACTCON(CON) = ACTSAV(CON)
      ENDDO

      DO CON = 1, NUMCON+4
         CONACT(CON) = CONSAV(CON)
      ENDDO

      DO CON = 1, NUMCON+3
         DO VAR = 1, NUMVAR
            CONRED(CON,VAR) = REDSAV(CON,VAR)
         ENDDO
      ENDDO

      DO CON = 1, NUMCON
         DO VAR = 1, NUMVAR
            CONDER(CON,VAR) = DERSAV(CON,VAR)
         ENDDO
      ENDDO
C ======================================================================
      RETURN
      END