      SUBROUTINE OGPWRI_START()
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C WRITE OPTIMIZATION LOG IN PYGMO FORMAT
C ======================================================================
C INP | VARDER           | I*4 | DERIVATIVES COMPUTATION MODE
C     |                  |     | -> 0: VALUES ONLY
C     |                  |     | -> 1: USER DEFINED
C     |                  |     | -> 2: NUMERIC WITH DOUBLE DIFFERENCING
C     |                  |     | -> 3: NUMERIC WITH SINGLE DIFFERENCING
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
      WRITE(LOGLUP, '("OPTGRA plugin for pagmo/pygmo:")') 
      IF     (VARDER .EQ. 0) THEN
            WRITE(LOGLUP, '("")')
      ELSEIF (VARDER .EQ. 1 .OR. VARDER .EQ. -1) THEN
            WRITE(LOGLUP,
     &'("    User-defined gradients")')
      ELSEIF (VARDER .EQ. 2) THEN
            WRITE(LOGLUP,
     &'("    Numerical gradients by double differencing")')
      ELSEIF (VARDER .EQ. 3) THEN
            WRITE(LOGLUP,
     &'("    Numerical gradients by single differencing")')
      ENDIF

      IF (OPTMET .EQ. 3) THEN
            WRITE(LOGLUP,
     &'("    Conjugate gradient method")')
      ELSEIF (OPTMET .EQ. 2) THEN
            WRITE(LOGLUP,
     &'("    Spectral conjugate gradient method")')
      ELSEIF (OPTMET .EQ. 1) THEN
            WRITE(LOGLUP,
     &'("    Modified spectral conjugate gradient method")')
      ELSEIF (OPTMET .EQ. 0) THEN
            WRITE(LOGLUP,
     &'("    Steepest descent method")')
      ENDIF

      WRITE(LOGLUP, '("")')

C ======================================================================
      RETURN
      END

    