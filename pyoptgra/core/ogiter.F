      SUBROUTINE OGITER (ITEMAX, ITECOR, ITEOPT, ITEDIV, ITECNV)
C ======================================================================
C NEAR-LINEAR OPTIMISATION TOOL TAILORED FOR S/C TRAJECTORY DESIGN:
C DEFINE OPTIMISATION CONTROL PARAMETERS
C ======================================================================
C INP | ITEMAX           | I*4 | MAXIMUM NUMBER OF ITERATIONS
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
      INTEGER*4    ITEMAX, ITECOR, ITEOPT, ITEDIV, ITECNV
      REAL*8       MAXVAR
      REAL*8       SNDVAR
C ======================================================================
      MAXITE = ITEMAX
      CORITE = ITECOR
      OPTITE = ITEOPT
      DIVITE = ITEDIV
      CNVITE = ITECNV
      IF (CORITE .GT. MAXITE) CORITE = MAXITE
      IF (OPTITE .GT. MAXITE) OPTITE = MAXITE
      IF (DIVITE .GT. CORITE) DIVITE = CORITE
      IF (CNVITE .GT. OPTITE) CNVITE = OPTITE
C ======================================================================
      RETURN
      END