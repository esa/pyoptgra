      SUBROUTINE OGPWRI (OBJVAL, NUMVIO, CONVIO)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C WRITE OPTIMIZATION LOG IN PYGMO FORMAT
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
      INTEGER(4)VER, RET, NUMVIO, mexprintf, mexevalstring, ICONVIO
C ======================================================================
      IF (VERBOS .EQ. 0) THEN
            RETURN
      ENDIF
C Print header
      IF (FEVALS .EQ. 0) THEN
            CALL OGPWRI_START()
      ENDIF
C Increase counter for cost function evaluations
      FEVALS = FEVALS + 1
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
    
C Write the log line (different format depending on violation size)
      IF (CONVIO == 0D0) THEN
            FMT = '(I10,F15.4,I15,I15,A2)'
            WRITE(LOGLUP, FMT) FEVALS, OBJVAL, NUMVIO, INT(CONVIO), FEAS
      ELSE IF (CONVIO .GT. 1D-3) THEN
            FMT = '(I10,F15.4,I15,F15.6,A2)'
            WRITE(LOGLUP, FMT) FEVALS, OBJVAL, NUMVIO, CONVIO, FEAS
      ELSE
            FMT = '(I10,F15.4,I15,E15.6,A2)'
            WRITE(LOGLUP, FMT) FEVALS, OBJVAL, NUMVIO, CONVIO, FEAS
      ENDIF
      ENDIF

C Write final summary
      IF (PYGFLA .NE. 0) THEN
            CALL OGPWRI_END (OBJVAL, NUMVIO, CONVIO)
      ENDIF
C ======================================================================
      RETURN
      END

    