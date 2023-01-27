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
#include <ogdata.inc>
C ======================================================================
      INTEGER(4)   VARNUM
      INTEGER(4)   CONNUM
C ======================================================================
      INTEGER(4)   VAR, CON
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
      ALLOCATE    (SENVAR(NUMVAR))
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
         SENVAR(VAR) = 0D0
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
      ALLOCATE    (SENQUA(NUMCON+1))
      ALLOCATE    (SENCON(NUMCON+1))
      ALLOCATE    (SENDEL(NUMCON+1))
      ALLOCATE    (SENACT(NUMCON+1))
C ----------------------------------------------------------------------
      DO CON = 1, NUMCON+1
         CONVAL(CON) = 0D0
         CONTYP(CON) = 0
         CONPRI(CON) = 1
         CONSCA(CON) = 1D0
         CONSTR(CON) = ""
         CONLEN(CON) = 0
         CONREF(CON) = 0D0
         SENQUA(CON) = 0D0
         SENCON(CON) = 0D0
         SENDEL(CON) = 0D0
         SENACT(CON) = 0
      ENDDO
C ======================================================================
C CONTROL
C ----------------------------------------------------------------------
      OPTMET = 2
      MAXITE = 10
      CORITE = 10
      OPTITE = 10
      DIVITE = 10
      CNVITE = 10
      VARMAX = 10D0
      VARSND = 1D0
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
C PYGMO LOG FILE
C ----------------------------------------------------------------------
      LOGLUP = 7
      LOGLEV = 0
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
C LINEAR OPTIMISATION MODE
C ----------------------------------------------------------------------
      SENOPT = 0
C ======================================================================
C WORKING VECTORS
C ----------------------------------------------------------------------
      ALLOCATE    (ACTCON(NUMCON+1))
      ALLOCATE    (CONFIX(NUMCON))
      ALLOCATE    (CONACT(NUMCON+4))
      ALLOCATE    (CONDER(NUMCON+3,NUMVAR))
      ALLOCATE    (CONRED(NUMCON+3,NUMVAR))
      ALLOCATE    (SENDER(NUMCON+3,NUMVAR))
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
