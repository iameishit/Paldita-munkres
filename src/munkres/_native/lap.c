/* Paldita Munkres v3 native kernel: dense linear assignment, n <= m.
 *
 * Shortest-augmenting-path Hungarian method with dual potentials, O(n^2 m).
 * Contract (see docs/NUMERICAL_CONTRACT.md):
 *   - c is row-major n x m, forbidden cells hold +inf (double) / INT64_MAX (int64).
 *   - NaN must be rejected by the caller. Int64 callers must guarantee
 *     max|c| * (n + m + 4) < 2^62 so no intermediate overflows.
 *   - minimises sum of c over a matching of all n rows into distinct columns.
 * Returns 4 if |c| is too large for overflow-free arithmetic (caller must use exact math).
 * Returns 0 on success, 2 if a NaN or -inf cost is found, 1 if no complete assignment exists (Hall witness in
 * hall_rows/hall_cols with counts in n_hr and n_hc), -1 on bad arguments / OOM.
 * u (n) and v (m) receive dual potentials: u_i + v_j <= c_ij, tight on matched
 * edges, v_j == 0 on unmatched columns (up to rounding for doubles).
 */
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* The algorithm lives in NAME##_core, which never allocates: the caller supplies the scratch arrays
 * (u, v and sr must be zeroed; the others are overwritten). NAME allocates one workspace per call;
 * the batch entry points allocate it once and reuse it for every problem. */
#define DEFINE_KERNEL(NAME, T, INFV, ISINF, ISBAD, LIMIT)                                 \
static int NAME##_core(int n, int m, const T *c, int *row_to_col, T *u_out, T *v_out,     \
         int *hall_rows, int *hall_cols, int *n_hr, int *n_hc,                           \
         T *u, T *v, T *shortest, int *row4col, int *col4row, int *path,                 \
         int *remaining, int *sc_list, char *sr) {                                       \
    int rc = 0; double big = 0.0;                                                        \
    *n_hr = 0; *n_hc = 0;                                                                \
    for (int j = 0; j < m; j++) row4col[j] = -1;                                         \
    for (int i = 0; i < n; i++) col4row[i] = -1;                                         \
    /* warm start: row reduction + greedy matching on tight cells */                     \
    for (int i = 0; i < n; i++) {                                                        \
        const T *row = c + (size_t)i * m;                                                \
        T low = INFV; int any = 0;                                                       \
        for (int j = 0; j < m; j++) { T x = row[j]; if (ISBAD(x)) { rc = 2; goto done; } if (!ISINF(x)) { double ax = (double)x; if (ax < 0) ax = -ax; if (ax > big) big = ax; if (!any || x < low) { low = x; any = 1; } } } \
        if (!any) { hall_rows[0] = i; *n_hr = 1; *n_hc = 0; rc = 1; goto done; }         \
        u[i] = low;                                                                      \
        for (int j = 0; j < m; j++)                                                      \
            if (!ISINF(row[j]) && row4col[j] < 0 && row[j] == low) { row4col[j] = i; col4row[i] = j; break; } \
    }                                                                                    \
    if (big * ((double)n + (double)m + 4.0) >= (LIMIT)) { rc = 4; goto done; }           \
    for (int cur = 0; cur < n; cur++) {                                                  \
        if (col4row[cur] >= 0) continue;                                                 \
        int nrem = m, nsc = 0, nsr = 0;                                                  \
        for (int k = 0; k < m; k++) { remaining[k] = m - 1 - k; shortest[k] = INFV; }    \
        T minval = 0; int i = cur, sink = -1;                                            \
        int *sr_rows = hall_rows; /* scratch for visited rows (n slots) */               \
        while (sink < 0) {                                                               \
            sr[i] = 1; sr_rows[nsr++] = i;                                               \
            T lowest = INFV; int index = -1;                                             \
            const T *row = c + (size_t)i * m; T ui = u[i];                               \
            for (int k = 0; k < nrem; k++) {                                             \
                int j = remaining[k]; T x = row[j];                                      \
                if (!ISINF(x)) {                                                         \
                    T r = minval + x - ui - v[j];                                        \
                    if (r < shortest[j]) { path[j] = i; shortest[j] = r; }               \
                }                                                                        \
                T sj = shortest[j];                                                      \
                if (!ISINF(sj) && (index < 0 || sj < lowest || (sj == lowest && row4col[j] < 0))) { lowest = sj; index = k; } \
            }                                                                            \
            if (index < 0) {                                                             \
                for (int a = 0; a < nsc; a++) hall_cols[a] = sc_list[a];                 \
                *n_hr = nsr; *n_hc = nsc; rc = 1;                                        \
                for (int a = 0; a < nsr; a++) sr[sr_rows[a]] = 0;                        \
                goto done;                                                               \
            }                                                                            \
            minval = lowest; int j = remaining[index];                                   \
            if (row4col[j] < 0) sink = j;                                                \
            else { i = row4col[j]; sc_list[nsc++] = j; remaining[index] = remaining[--nrem]; } \
        }                                                                                \
        u[cur] += minval;                                                                \
        for (int a = 0; a < nsc; a++) {                                                  \
            int j = sc_list[a]; u[row4col[j]] += minval - shortest[j];                   \
            v[j] -= minval - shortest[j];                                                \
        }                                                                                \
        for (int a = 0; a < nsr; a++) sr[sr_rows[a]] = 0;                                \
        int j = sink;                                                                    \
        for (;;) {                                                                       \
            int ii = path[j]; row4col[j] = ii; int tmp = col4row[ii]; col4row[ii] = j; j = tmp; \
            if (ii == cur) break;                                                        \
        }                                                                                \
    }                                                                                    \
    for (int i = 0; i < n; i++) row_to_col[i] = col4row[i];                              \
    if (u_out) for (int i = 0; i < n; i++) u_out[i] = u[i];                              \
    if (v_out) for (int j = 0; j < m; j++) v_out[j] = v[j];                              \
done:                                                                                    \
    return rc;                                                                           \
}                                                                                        \
                                                                                         \
int NAME(int n, int m, const T *c, int *row_to_col, T *u_out, T *v_out,                  \
         int *hall_rows, int *hall_cols, int *n_hr, int *n_hc) {                         \
    if (n < 0 || m < n || !c || !row_to_col) return -1;                                  \
    *n_hr = 0; *n_hc = 0;                                                                \
    if (n == 0) return 0;                                                                \
    T *u = (T *)calloc((size_t)n, sizeof(T));                                            \
    T *v = (T *)calloc((size_t)m, sizeof(T));                                            \
    T *shortest = (T *)malloc((size_t)m * sizeof(T));                                    \
    int *row4col = (int *)malloc((size_t)m * sizeof(int));                               \
    int *col4row = (int *)malloc((size_t)n * sizeof(int));                               \
    int *path = (int *)malloc((size_t)m * sizeof(int));                                  \
    int *remaining = (int *)malloc((size_t)m * sizeof(int));                             \
    int *sc_list = (int *)malloc((size_t)m * sizeof(int));                               \
    char *sr = (char *)calloc((size_t)n, 1);                                             \
    int rc = -1;                                                                         \
    if (u && v && shortest && row4col && col4row && path && remaining && sc_list && sr)  \
        rc = NAME##_core(n, m, c, row_to_col, u_out, v_out, hall_rows, hall_cols, n_hr, n_hc, \
                         u, v, shortest, row4col, col4row, path, remaining, sc_list, sr); \
    free(u); free(v); free(shortest); free(row4col); free(col4row); free(path); free(remaining); free(sc_list); free(sr); \
    return rc;                                                                           \
}                                                                                        \
                                                                                         \
/* Batch: B independent n x m problems (contiguous), one call, ONE workspace for all of them. \
 * status[b] is the kernel return code of problem b (0 ok, 1 infeasible, 2 NaN/-inf, 4 range);   \
 * out_r2c is B x n. Returns 0, or -1 on bad arguments / allocation failure. */                \
int NAME##_batch(int B, int n, int m, const T *c, int *out_r2c, int *status) {           \
    if (B < 0 || n < 0 || m < n || (B > 0 && (!c || !out_r2c || !status))) return -1;    \
    if (B == 0 || n == 0) { for (int b = 0; b < B; b++) status[b] = 0; return 0; }       \
    T *u = (T *)malloc((size_t)n * sizeof(T));                                           \
    T *v = (T *)malloc((size_t)m * sizeof(T));                                           \
    T *shortest = (T *)malloc((size_t)m * sizeof(T));                                    \
    int *row4col = (int *)malloc((size_t)m * sizeof(int));                               \
    int *col4row = (int *)malloc((size_t)n * sizeof(int));                               \
    int *path = (int *)malloc((size_t)m * sizeof(int));                                  \
    int *remaining = (int *)malloc((size_t)m * sizeof(int));                             \
    int *sc_list = (int *)malloc((size_t)m * sizeof(int));                               \
    char *sr = (char *)malloc((size_t)n);                                                \
    int *hr = (int *)malloc(((size_t)n + 1) * sizeof(int));                              \
    int *hc = (int *)malloc(((size_t)m + 1) * sizeof(int));                              \
    int rc = -1;                                                                         \
    if (u && v && shortest && row4col && col4row && path && remaining && sc_list && sr && hr && hc) { \
        int nh1, nh2;                                                                    \
        for (int b = 0; b < B; b++) {                                                    \
            memset(u, 0, (size_t)n * sizeof(T)); memset(v, 0, (size_t)m * sizeof(T)); memset(sr, 0, (size_t)n); \
            status[b] = NAME##_core(n, m, c + (size_t)b * n * m, out_r2c + (size_t)b * n, NULL, NULL, hr, hc, &nh1, &nh2, \
                                    u, v, shortest, row4col, col4row, path, remaining, sc_list, sr); \
        }                                                                                \
        rc = 0;                                                                          \
    }                                                                                    \
    free(u); free(v); free(shortest); free(row4col); free(col4row); free(path); free(remaining); free(sc_list); free(sr); free(hr); free(hc); \
    return rc;                                                                           \
}

#define ISINF_D(x) (isinf(x))
#define ISINF_I(x) ((x) == INT64_MAX)
#define ISBAD_D(x) (isnan(x) || (x) == -INFINITY)
#define ISBAD_I(x) (0)
DEFINE_KERNEL(paldita_lap_f64, double, INFINITY, ISINF_D, ISBAD_D, 1.7e308)
DEFINE_KERNEL(paldita_lap_i64, int64_t, INT64_MAX, ISINF_I, ISBAD_I, 4.0e18)

int paldita_abi_version(void) { return 2; }
