      SUBROUTINE OGPWRI (OBJVAL, NUMVIO, CONVIO)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C WRITE OPTIMIZATION LOG IN PYGMO FORMAT
C ======================================================================
C INP | OBJVAL           | R*8 | OBJECTIVE VALUE
C ----------------------------------------------------------------------
C INP | NUMVIO           | I*4 | NUMBER OF VIOLATED CONSTRAINTS
C ======================================================================
C INP | CONVIO           | R*8 | TOTAL CONSTRAINT VIOLATION
C ======================================================================
C 2023/01/25 | W. MARTENS | NEW
C ======================================================================
      IMPLICIT NONE
C ======================================================================
#include <ogdata.inc>
C ======================================================================
      CHARACTER STR*256, FEAS*2
      REAL(8) OBJVAL, CONVIO, X
      INTEGER(4)VER, RET, NUMVIO, mexprintf, mexevalstring
C ======================================================================
C Increase counter for cost function evaluations
      FEVALS = FEVALS + 1
      IF (VERBOS .EQ. 0) THEN
            RETURN
      ENDIF
C Every 50 lines print the column names.
      IF (mod(REAL(FEVALS-1D0) / REAL(VERBOS), 50D0) .EQ. 0D0) THEN
            WRITE(LOGLUP, '(A10,A15,A15,A15,A2)')
     &           "objevals:", "objval:", "violated:", "viol. norm:"
      ENDIF
      IF (VERBOS .NE. 0 .AND. MOD(FEVALS, VERBOS) == 0D0) THEN
            IF (CONVIO > 0D0) THEN
                  FEAS = " i"
            ELSE
                  FEAS = "  "
            ENDIF
    
C Write the log line
            WRITE(LOGLUP, '(I10,F15.4,I15,F15.6,A2)')
     &            FEVALS, OBJVAL, NUMVIO, CONVIO, FEAS
C            flush (LOGLUP)
      ENDIF


C ======================================================================
      RETURN
      END

    