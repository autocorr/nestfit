#include <stdlib.h>
#include <math.h>

#define SQRT_PI                 (sqrt(M_PI))           /* sqrt(pi)	*/
#define FAST_EXP_MAX_TAYLOR     3
#define FAST_EXP_NUM_BITS       8
#define ERF_TABLE_LIMIT         6.0                    /* For x>6 erf(x)-1<double precision machine epsilon, so no need to store the values for larger x. */
#define ERF_TABLE_SIZE          6145
#define BIN_WIDTH               (ERF_TABLE_LIMIT/(ERF_TABLE_SIZE-1.))
#define IBIN_WIDTH              (1./BIN_WIDTH)


extern double EXP_TABLE_2D[128][10];
extern double EXP_TABLE_3D[256][2][10];
extern double oneOver_i[FAST_EXP_MAX_TAYLOR+1];

extern double ERF_TABLE[ERF_TABLE_SIZE];

double	geterf(const double, const double);
double	FastExp(const float);
void	calcExpTableEntries(const int, const int);
void	fillErfTable(void);
