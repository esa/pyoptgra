      SUBROUTINE OGPWRI_END (OBJVAL, NUMVIO, CONVIO)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C WRITE OPTIMIZATION END RESULT IN PYGMO FORMAT
C ======================================================================
C INP | OBJVAL           | R*8 | OBJECTIVE VALUE
C ----------------------------------------------------------------------
C INP | NUMVIO           | I*4 | NUMBER OF VIOLATED CONSTRAINTS
C ----------------------------------------------------------------------
C INP | CONVIO           | R*8 | TOTAL CONSTRAINT VIOLATION
C ======================================================================
C 2023/01/25 | W. MARTENS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      CHARACTER STR*256, FEAS*2, FMT*24
      REAL(8) OBJVAL, CONVIO, X
      INTEGER(4) NUMVIO
C ======================================================================
      IF (PYGFLA .EQ. 0) THEN
            RETURN
      ENDIF
C Write termination message
      WRITE(LOGLUP, '("")')
      WRITE(LOGLUP, '("Final values after iteration        ", I10:)') 
     &NUMITE
      WRITE(LOGLUP, '("Final objective value:              ", F10.4)')
     &OBJVAL
      WRITE(LOGLUP, '("Final constraint violation:         ", F10.4)')
     &CONVIO      
      WRITE(LOGLUP, '("Final num. of violated constraints: ",I10)') 
     &NUMVIO 
      IF (PYGFLA .EQ. 1) THEN
            WRITE(LOGLUP, 
     &      '("Successful termination: Optimal solution found.")')
      ELSE IF (PYGFLA .EQ. 2) THEN
            WRITE(LOGLUP, 
     &      '("Successful termination: Constraints matched.")')
      ELSE IF (PYGFLA .EQ. 3) THEN
            WRITE(LOGLUP, '("Not converged.")') 
      ELSE IF (PYGFLA .EQ. 4) THEN
            WRITE(LOGLUP, '("Problem appears infeasible.")')
      ENDIF
      WRITE(LOGLUP, '("")')

C ======================================================================
      RETURN
      END

    