      SUBROUTINE OGCLOS ()
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEALLOCATION OF ARRAYS
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
C VARIABLES
C ----------------------------------------------------------------------
      DEALLOCATE  (VARVAL)
      DEALLOCATE  (VARTYP)
      DEALLOCATE  (VARSCA)
      DEALLOCATE  (VARSTR)
      DEALLOCATE  (VARLEN)
      DEALLOCATE  (VARREF)
      DEALLOCATE  (VARDES)
      DEALLOCATE  (VARGRD)
      DEALLOCATE  (VARDIR)
      DEALLOCATE  (FUNVAR)
C ======================================================================
C CONSTRAINTS
C ----------------------------------------------------------------------
      DEALLOCATE  (CONVAL)
      DEALLOCATE  (CONTYP)
      DEALLOCATE  (CONPRI)
	DEALLOCATE  (CONSCA)
      DEALLOCATE  (CONSTR)
      DEALLOCATE  (CONLEN)
      DEALLOCATE  (CONREF)
C ======================================================================
C DERIVATIVES
C ----------------------------------------------------------------------
      DEALLOCATE  (VARPER)
C ======================================================================
C WORKING VECTORS
C ----------------------------------------------------------------------
      DEALLOCATE  (ACTCON)
      DEALLOCATE  (CONFIX)
      DEALLOCATE  (CONACT)
      DEALLOCATE  (CONDER)
      DEALLOCATE  (CONRED)
      DEALLOCATE  (CONOPT)
C ======================================================================
      RETURN
      END
