/* CPython extension `munkres._lapc`: thin buffer-protocol wrapper over lap.c (no ctypes, no numpy).
 *
 * Every buffer is validated before the GIL is released: element size, alignment, writability,
 * C-contiguity and length (computed with overflow checks). A caller can therefore hand over any
 * object and get an exception, never an out-of-bounds or misaligned access. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include "lap.c"

typedef struct { Py_buffer v; int ok; } Buf;

/* a * b into *out; returns 0 on success, -1 on overflow (portable: no compiler builtins). */
static int mul_size(Py_ssize_t a, Py_ssize_t b, Py_ssize_t *out) {
    if (a < 0 || b < 0) return -1;
    if (a != 0 && b > PY_SSIZE_T_MAX / a) return -1;
    *out = a * b;
    return 0;
}

/* Export `o` as a C-contiguous buffer of `count` elements of `itemsize` bytes (at least `count`).
 * `o == None` is allowed and leaves b->ok == 0 (callers that require the buffer check b->ok). */
static int grab(PyObject *o, Buf *b, int writable, Py_ssize_t count, Py_ssize_t itemsize, const char *name) {
    b->ok = 0;
    if (o == Py_None) return 0;
    int flags = PyBUF_C_CONTIGUOUS | (writable ? PyBUF_WRITABLE : 0);
    if (PyObject_GetBuffer(o, &b->v, flags) != 0) return -1;
    b->ok = 1;
    Py_ssize_t need;
    if (b->v.itemsize != itemsize) {
        PyErr_Format(PyExc_TypeError, "buffer '%s' must have %zd-byte items, got %zd-byte items", name, itemsize, b->v.itemsize);
    } else if (count > 0 && ((uintptr_t)b->v.buf) % (uintptr_t)itemsize != 0) {  /* empty buffers are never dereferenced */
        PyErr_Format(PyExc_ValueError, "buffer '%s' is not aligned to %zd bytes", name, itemsize);
    } else if (mul_size(count, itemsize, &need) != 0) {
        PyErr_Format(PyExc_OverflowError, "buffer '%s' size overflows", name);
    } else if (b->v.len < need) {
        PyErr_Format(PyExc_ValueError, "buffer '%s' too small (%zd < %zd bytes)", name, b->v.len, need);
    } else {
        return 0;
    }
    PyBuffer_Release(&b->v);
    b->ok = 0;
    return -1;
}
static void drop(Buf *b) { if (b->ok) PyBuffer_Release(&b->v); }

static int count_cells(int n, int m, Py_ssize_t *cells) { return mul_size((Py_ssize_t)n, (Py_ssize_t)m, cells); }

/* solve(kind, n, m, cost, r2c, u, v, hall) -> int code
 * kind 0: float64, 1: int64. r2c, hall are int32; hall needs n + m + 4 entries:
 * [n_hr, n_hc, rows(n+1), cols(m+1)]. u (n) and v (m) are optional dual outputs. */
static PyObject *py_solve(PyObject *self, PyObject *args) {
    (void)self;
    int kind, n, m; PyObject *oc, *or2c, *ou, *ov, *oh;
    if (!PyArg_ParseTuple(args, "iiiOOOOO", &kind, &n, &m, &oc, &or2c, &ou, &ov, &oh)) return NULL;
    if (n < 0 || m < n || (kind != 0 && kind != 1)) { PyErr_SetString(PyExc_ValueError, "bad arguments"); return NULL; }
    Py_ssize_t cells;
    if (count_cells(n, m, &cells) != 0) { PyErr_SetString(PyExc_OverflowError, "problem size overflows"); return NULL; }
    Buf c = {0}, r = {0}, u = {0}, v = {0}, h = {0};
    int rc = -1;
    if (grab(oc, &c, 0, cells, 8, "cost")) goto out;
    if (!c.ok) { PyErr_SetString(PyExc_TypeError, "cost required"); goto out; }
    if (grab(or2c, &r, 1, n, 4, "r2c")) goto out;
    if (!r.ok) { PyErr_SetString(PyExc_TypeError, "r2c required"); goto out; }
    if (grab(ou, &u, 1, n, 8, "u") || grab(ov, &v, 1, m, 8, "v")) goto out;
    if (grab(oh, &h, 1, (Py_ssize_t)n + m + 4, 4, "hall")) goto out;
    if (!h.ok) { PyErr_SetString(PyExc_TypeError, "hall required"); goto out; }
    int *hall = (int *)h.v.buf;
    Py_BEGIN_ALLOW_THREADS
    if (kind == 0)
        rc = paldita_lap_f64(n, m, (const double *)c.v.buf, (int *)r.v.buf, u.ok ? (double *)u.v.buf : NULL,
                             v.ok ? (double *)v.v.buf : NULL, hall + 2, hall + 3 + n, &hall[0], &hall[1]);
    else
        rc = paldita_lap_i64(n, m, (const int64_t *)c.v.buf, (int *)r.v.buf, u.ok ? (int64_t *)u.v.buf : NULL,
                             v.ok ? (int64_t *)v.v.buf : NULL, hall + 2, hall + 3 + n, &hall[0], &hall[1]);
    Py_END_ALLOW_THREADS
out:
    drop(&c); drop(&r); drop(&u); drop(&v); drop(&h);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLong(rc);
}

/* solve_batch(B, n, m, cost, r2c, status[, kind]) -> 0   (kind 0: float64 (default), 1: int64) */
static PyObject *py_solve_batch(PyObject *self, PyObject *args) {
    (void)self;
    int B, n, m, kind = 0; PyObject *oc, *or2c, *os;
    if (!PyArg_ParseTuple(args, "iiiOOO|i", &B, &n, &m, &oc, &or2c, &os, &kind)) return NULL;
    if (B < 0 || n < 0 || m < n || (kind != 0 && kind != 1)) { PyErr_SetString(PyExc_ValueError, "bad arguments"); return NULL; }
    Py_ssize_t cells, total_cells, total_out;
    if (count_cells(n, m, &cells) != 0 || mul_size(cells, (Py_ssize_t)B, &total_cells) != 0 ||
        mul_size((Py_ssize_t)B, (Py_ssize_t)n, &total_out) != 0) {
        PyErr_SetString(PyExc_OverflowError, "batch size overflows"); return NULL;
    }
    Buf c = {0}, r = {0}, s = {0};
    int rc = -1;
    if (grab(oc, &c, 0, total_cells, 8, "cost")) goto out;
    if (!c.ok) { PyErr_SetString(PyExc_TypeError, "cost required"); goto out; }
    if (grab(or2c, &r, 1, total_out, 4, "r2c")) goto out;
    if (!r.ok) { PyErr_SetString(PyExc_TypeError, "r2c required"); goto out; }
    if (grab(os, &s, 1, B, 4, "status")) goto out;
    if (!s.ok) { PyErr_SetString(PyExc_TypeError, "status required"); goto out; }
    Py_BEGIN_ALLOW_THREADS
    if (kind == 0) rc = paldita_lap_f64_batch(B, n, m, (const double *)c.v.buf, (int *)r.v.buf, (int *)s.v.buf);
    else rc = paldita_lap_i64_batch(B, n, m, (const int64_t *)c.v.buf, (int *)r.v.buf, (int *)s.v.buf);
    Py_END_ALLOW_THREADS
out:
    drop(&c); drop(&r); drop(&s);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLong(rc);
}


/* lsa(kind, n, m, cost, out_cols, maximize) -> rc
 * Fast path for the ndarray API: solves one C-contiguous n x m (n <= m) float64/int64 problem and writes
 * the column of each row straight into `out_cols` (an array of n pointer-sized ints, i.e. numpy intp).
 * Returns 0 on success; any other value (1 infeasible, 2 NaN/-inf, 4 range, -1 allocation failure,
 * -2 unsupported platform) means "use the general path", which produces the exact diagnostics.
 * For int64 the kernel's INT64_MAX "forbidden" marker and INT64_MIN (not negatable) are rejected (rc 4). */
static PyObject *py_lsa(PyObject *self, PyObject *args) {
    (void)self;
    int kind, n, m, maximize; PyObject *oc, *oo;
    if (!PyArg_ParseTuple(args, "iiiOOp", &kind, &n, &m, &oc, &oo, &maximize)) return NULL;
    if (n < 0 || m < n || (kind != 0 && kind != 1)) { PyErr_SetString(PyExc_ValueError, "bad arguments"); return NULL; }
    if (sizeof(Py_ssize_t) != 8) return PyLong_FromLong(-2);
    Py_ssize_t cells;
    if (count_cells(n, m, &cells) != 0) { PyErr_SetString(PyExc_OverflowError, "problem size overflows"); return NULL; }
    Buf c = {0}, o = {0};
    long rc = -1;
    int *ib = NULL; void *neg = NULL;
    if (grab(oc, &c, 0, cells, 8, "cost") || grab(oo, &o, 1, n, 8, "out_cols")) goto out;
    if (!c.ok || !o.ok) { PyErr_SetString(PyExc_TypeError, "cost and out_cols required"); goto out; }
    if (n == 0) { rc = 0; goto out; }
    const void *src = c.v.buf;
    if (kind == 1) {
        const int64_t *p = (const int64_t *)src;
        for (Py_ssize_t k = 0; k < cells; k++) if (p[k] == INT64_MAX || p[k] == INT64_MIN) { rc = 4; goto out; }
    }
    if (maximize) {
        neg = malloc((size_t)cells * 8);
        if (!neg) goto out;
        if (kind == 0) { const double *p = (const double *)src; double *q = (double *)neg; for (Py_ssize_t k = 0; k < cells; k++) q[k] = -p[k]; }
        else { const int64_t *p = (const int64_t *)src; int64_t *q = (int64_t *)neg; for (Py_ssize_t k = 0; k < cells; k++) q[k] = -p[k]; }
        src = neg;
    }
    ib = (int *)malloc(((size_t)2 * n + (size_t)m + 4) * sizeof(int));
    if (!ib) goto out;
    {
        int *r2c = ib, *hall = ib + n, nh1 = 0, nh2 = 0;
        int release = cells >= 2048;  /* releasing the GIL costs more than a tiny solve */
        PyThreadState *ts = NULL;
        if (release) ts = PyEval_SaveThread();
        rc = kind == 0 ? paldita_lap_f64(n, m, (const double *)src, r2c, NULL, NULL, hall, hall + 1 + n, &nh1, &nh2)
                       : paldita_lap_i64(n, m, (const int64_t *)src, r2c, NULL, NULL, hall, hall + 1 + n, &nh1, &nh2);
        if (release) PyEval_RestoreThread(ts);
        if (rc == 0) { int64_t *dst = (int64_t *)o.v.buf; for (int i = 0; i < n; i++) dst[i] = r2c[i]; }
    }
out:
    free(ib); free(neg);
    drop(&c); drop(&o);
    if (PyErr_Occurred()) return NULL;
    return PyLong_FromLong(rc);
}

static PyObject *py_abi(PyObject *self, PyObject *noargs) { (void)self; (void)noargs; return PyLong_FromLong(paldita_abi_version()); }

static PyMethodDef methods[] = {
    {"solve", py_solve, METH_VARARGS, "solve(kind, n, m, cost, r2c, u, v, hall) -> code"},
    {"solve_batch", py_solve_batch, METH_VARARGS, "solve_batch(B, n, m, cost, r2c, status[, kind]) -> 0"},
    {"lsa", py_lsa, METH_VARARGS, "lsa(kind, n, m, cost, out_cols, maximize) -> code (ndarray fast path)"},
    {"abi_version", py_abi, METH_NOARGS, "kernel ABI version"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef moddef = {PyModuleDef_HEAD_INIT, "_lapc", "Paldita native LAP kernel", -1, methods, NULL, NULL, NULL, NULL};
PyMODINIT_FUNC PyInit__lapc(void) { return PyModule_Create(&moddef); }
