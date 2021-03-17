      SUBROUTINE OGINIT (VARNUM, CONNUM)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C ALLOCATION OF ARRAYS AND INITIALISATION OF PARAMETERS
C ======================================================================
C INP | VARNUM           | I*4 | NUMBER OF VARIABLES
C ----------------------------------------------------------------------
C INP | CONNUM           | I*4 | NUMBER OF CONSTRAINTS
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
#include "ogdata.inc"
C ======================================================================
      INTEGER*4    VARNUM
      INTEGER*4    CONNUM
C ======================================================================
      INTEGER*4    VAR, CON
C ======================================================================
C VARIABLES
C ----------------------------------------------------------------------
      NUMVAR = VARNUM
C ----------------------------------------------------------------------
      ALLOCATE    (VARVAL(NUMVAR))
      ALLOCATE    (VARTYP(NUMVAR))
      ALLOCATE    (VARSCA(NUMVAR))
      ALLOCATE    (VARSTR(NUMVAR))
      ALLOCATE    (VARLEN(NUMVAR))
      ALLOCATE    (VARREF(NUMVAR))
      ALLOCATE    (VARDES(NUMVAR))
      ALLOCATE    (VARGRD(NUMVAR))
      ALLOCATE    (VARDIR(NUMVAR))
      ALLOCATE    (FUNVAR(NUMVAR))
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         VARVAL(VAR) = 0D0
         VARTYP(VAR) = 0
         VARSCA(VAR) = 1D0
         VARSTR(VAR) = ""
         VARLEN(VAR) = 0
         VARREF(VAR) = 0D0
         VARDES(VAR) = 0D0
         VARGRD(VAR) = 0D0
         VARDIR(VAR) = 0D0
         FUNVAR(VAR) = 0D0
      ENDDO
C ======================================================================
C CONSTRAINTS
C ----------------------------------------------------------------------
      NUMCON = CONNUM
C ----------------------------------------------------------------------
      ALLOCATE    (CONVAL(NUMCON+1))
      ALLOCATE    (CONTYP(NUMCON+1))
      ALLOCATE    (CONPRI(NUMCON+1))
      ALLOCATE    (CONSCA(NUMCON+1))
      ALLOCATE    (CONSTR(NUMCON+1))
      ALLOCATE    (CONLEN(NUMCON+1))
      ALLOCATE    (CONREF(NUMCON+1))
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON+1
         CONVAL(CON) = 0D0
         CONTYP(CON) = 0
         CONPRI(CON) = 1
         CONSCA(CON) = 1D0
         CONSTR(CON) = ""
         CONLEN(CON) = 0
         CONREF(CON) = 0D0
      ENDDO
C ======================================================================
C CONTROL
C ----------------------------------------------------------------------
      MAXITE = 10
      CORITE = 10
      VARMAX = 10
      VARSND = 1
      OPTMET = 2
C ======================================================================
C DERIVATIVES
C ----------------------------------------------------------------------
      VARDER = 1
C ----------------------------------------------------------------------
      ALLOCATE    (VARPER(NUMVAR))
C ----------------------------------------------------------------------
      DO VAR = 1, NUMVAR
         VARPER(VAR) = 1D-03
      ENDDO
C ======================================================================
C LOG FILE
C ----------------------------------------------------------------------
      LOGLUN = 6
      LOGLEV = 1
C ======================================================================
C MATLAB CONSOLE
C ----------------------------------------------------------------------
      MATLEV = 0
C ======================================================================
C TABLE FILE
C ----------------------------------------------------------------------
      TABLUN = 6
      TABLEV = 0
C ======================================================================
C WORKING VECTORS
C ----------------------------------------------------------------------
      ALLOCATE    (ACTCON(NUMCON+1))
      ALLOCATE    (CONFIX(NUMCON))
      ALLOCATE    (CONACT(NUMCON+4))
      ALLOCATE    (CONDER(NUMCON+3,NUMVAR))
      ALLOCATE    (CONRED(NUMCON+3,NUMVAR))
      ALLOCATE    (CONOPT(NUMCON+1))
C ----------------------------------------------------------------------
      NUMACT = 0
      ACTCON = 0
      CONACT = 0
      CONFIX = 0
      CONDER = 0D0
      CONRED = 0D0
      CONOPT = 0
C ======================================================================
      RETURN
      END
