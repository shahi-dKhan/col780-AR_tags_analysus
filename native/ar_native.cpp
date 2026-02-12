#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace py = pybind11;

static inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// border_mode: 0=constant, 1=replicate
// interpolation: 0=nearest, 1=bilinear
py::array warp_perspective_u8(
    py::array src,
    py::array M_in,
    int out_h,
    int out_w,
    int interpolation,
    bool warp_inverse_map,
    int border_mode,
    py::object border_value_obj
) {
    if (!(src.ndim() == 2 || src.ndim() == 3)) {
        throw std::runtime_error("src must have 2 or 3 dimensions");
    }
    if (src.dtype().kind() != 'u' || src.dtype().itemsize() != 1) {
        throw std::runtime_error("src must be uint8");
    }

    auto M_arr = py::array::ensure(M_in);
    if (!M_arr || M_arr.ndim() != 2 || M_arr.shape(0) != 3 || M_arr.shape(1) != 3) {
        throw std::runtime_error("M must be a 3x3 array");
    }

    // Read M as double
    double M[3][3];
    {
        py::buffer_info bi = M_arr.request();
        if (bi.format != py::format_descriptor<double>::format() &&
            bi.format != py::format_descriptor<float>::format()) {
            // Allow float32/float64
            // We'll copy via Python buffer cast by reading element-wise
        }
        // element-wise safe read
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                M[r][c] = py::float_(M_arr.attr("__getitem__")(py::make_tuple(r, c)));
            }
        }
    }

    // OpenCV semantics: if warp_inverse_map==false, OpenCV inverts the input matrix.
    // Here we expect M_in to be the same matrix you pass to cv2.warpPerspective.
    // If warp_inverse_map is false, compute inverse so we map dst->src.
    double A[3][3];
    if (warp_inverse_map) {
        for (int r = 0; r < 3; r++) for (int c = 0; c < 3; c++) A[r][c] = M[r][c];
    } else {
        // Invert 3x3
        const double a00 = M[0][0], a01 = M[0][1], a02 = M[0][2];
        const double a10 = M[1][0], a11 = M[1][1], a12 = M[1][2];
        const double a20 = M[2][0], a21 = M[2][1], a22 = M[2][2];

        const double b00 = (a11 * a22 - a12 * a21);
        const double b01 = (a02 * a21 - a01 * a22);
        const double b02 = (a01 * a12 - a02 * a11);
        const double b10 = (a12 * a20 - a10 * a22);
        const double b11 = (a00 * a22 - a02 * a20);
        const double b12 = (a02 * a10 - a00 * a12);
        const double b20 = (a10 * a21 - a11 * a20);
        const double b21 = (a01 * a20 - a00 * a21);
        const double b22 = (a00 * a11 - a01 * a10);

        const double det = a00 * b00 + a01 * b10 + a02 * b20;
        if (std::abs(det) < 1e-12) {
            throw std::runtime_error("Matrix is singular");
        }
        const double inv_det = 1.0 / det;

        A[0][0] = b00 * inv_det;
        A[0][1] = b01 * inv_det;
        A[0][2] = b02 * inv_det;
        A[1][0] = b10 * inv_det;
        A[1][1] = b11 * inv_det;
        A[1][2] = b12 * inv_det;
        A[2][0] = b20 * inv_det;
        A[2][1] = b21 * inv_det;
        A[2][2] = b22 * inv_det;
    }

    const int src_h = (int)src.shape(0);
    const int src_w = (int)src.shape(1);
    const int channels = (src.ndim() == 3) ? (int)src.shape(2) : 1;

    // border value
    std::vector<double> border_val((size_t)channels, 0.0);
    if (channels == 1) {
        border_val[0] = py::float_(border_value_obj);
    } else {
        if (py::isinstance<py::sequence>(border_value_obj)) {
            py::sequence seq = border_value_obj;
            if ((int)py::len(seq) != channels) {
                throw std::runtime_error("border_value length must equal channels");
            }
            for (int c = 0; c < channels; c++) border_val[c] = py::float_(seq[c]);
        } else {
            double v = py::float_(border_value_obj);
            for (int c = 0; c < channels; c++) border_val[c] = v;
        }
    }

    // allocate output
    py::array out;
    if (channels == 1) {
        out = py::array_t<uint8_t>({out_h, out_w});
    } else {
        out = py::array_t<uint8_t>({out_h, out_w, channels});
    }

    py::buffer_info src_bi = src.request();
    py::buffer_info out_bi = out.request();

    const uint8_t* src_ptr = static_cast<const uint8_t*>(src_bi.ptr);
    uint8_t* out_ptr = static_cast<uint8_t*>(out_bi.ptr);

    const ssize_t src_stride_y = src_bi.strides[0];
    const ssize_t src_stride_x = src_bi.strides[1];
    const ssize_t src_stride_c = (channels == 1) ? 0 : src_bi.strides[2];

    const ssize_t out_stride_y = out_bi.strides[0];
    const ssize_t out_stride_x = out_bi.strides[1];
    const ssize_t out_stride_c = (channels == 1) ? 0 : out_bi.strides[2];

    auto read_pixel = [&](int y, int x, int c) -> double {
        if (border_mode == 1) {
            y = clamp_int(y, 0, src_h - 1);
            x = clamp_int(x, 0, src_w - 1);
            const uint8_t* p = src_ptr + y * src_stride_y + x * src_stride_x + c * src_stride_c;
            return (double)(*p);
        }
        // constant
        if (y < 0 || x < 0 || y >= src_h || x >= src_w) {
            return border_val[(size_t)c];
        }
        const uint8_t* p = src_ptr + y * src_stride_y + x * src_stride_x + c * src_stride_c;
        return (double)(*p);
    };

    const double h00 = A[0][0], h01 = A[0][1], h02 = A[0][2];
    const double h10 = A[1][0], h11 = A[1][1], h12 = A[1][2];
    const double h20 = A[2][0], h21 = A[2][1], h22 = A[2][2];

    py::gil_scoped_release release;

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            const double denom = h20 * x + h21 * y + h22;
            if (std::abs(denom) < 1e-12) {
                // write border
                if (channels == 1) {
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    *op = (uint8_t)std::clamp((int)std::lround(border_val[0]), 0, 255);
                } else {
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    for (int c = 0; c < channels; c++) {
                        op[c * out_stride_c] = (uint8_t)std::clamp((int)std::lround(border_val[(size_t)c]), 0, 255);
                    }
                }
                continue;
            }
            const double sx = (h00 * x + h01 * y + h02) / denom;
            const double sy = (h10 * x + h11 * y + h12) / denom;

            if (interpolation == 0) {
                const int ix = (int)std::floor(sx);
                const int iy = (int)std::floor(sy);
                if (channels == 1) {
                    const double v = read_pixel(iy, ix, 0);
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    *op = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
                } else {
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    for (int c = 0; c < channels; c++) {
                        const double v = read_pixel(iy, ix, c);
                        op[c * out_stride_c] = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
                    }
                }
            } else {
                const int ix = (int)std::floor(sx);
                const int iy = (int)std::floor(sy);
                const double dx = sx - ix;
                const double dy = sy - iy;

                if (channels == 1) {
                    const double p00 = read_pixel(iy, ix, 0);
                    const double p10 = read_pixel(iy, ix + 1, 0);
                    const double p01 = read_pixel(iy + 1, ix, 0);
                    const double p11 = read_pixel(iy + 1, ix + 1, 0);
                    const double v = (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p10 + (1 - dx) * dy * p01 + dx * dy * p11;
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    *op = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
                } else {
                    uint8_t* op = out_ptr + y * out_stride_y + x * out_stride_x;
                    for (int c = 0; c < channels; c++) {
                        const double p00 = read_pixel(iy, ix, c);
                        const double p10 = read_pixel(iy, ix + 1, c);
                        const double p01 = read_pixel(iy + 1, ix, c);
                        const double p11 = read_pixel(iy + 1, ix + 1, c);
                        const double v = (1 - dx) * (1 - dy) * p00 + dx * (1 - dy) * p10 + (1 - dx) * dy * p01 + dx * dy * p11;
                        op[c * out_stride_c] = (uint8_t)std::clamp((int)std::lround(v), 0, 255);
                    }
                }
            }
        }
    }

    return out;
}

PYBIND11_MODULE(ar_native, m) {
    m.doc() = "Native acceleration for AR tag project";
    m.def(
        "warp_perspective_u8",
        &warp_perspective_u8,
        py::arg("src"),
        py::arg("M"),
        py::arg("out_h"),
        py::arg("out_w"),
        py::arg("interpolation") = 1,
        py::arg("warp_inverse_map") = false,
        py::arg("border_mode") = 0,
        py::arg("border_value") = 0
    );
}
