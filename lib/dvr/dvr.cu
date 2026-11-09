// Acknowledgments: https://github.com/tarashakhurana/4d-occ-forecasting
// Modified by Haisong Liu

#include <torch/extension.h>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define MAX_D 1446
#define MAX_STEP 1000

template <typename scalar_t>
__global__ void render_forward_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> sigma,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> origin,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> tindex,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> pred_dist,
    torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> coord_index) {

    const auto n = blockIdx.y;

    const auto c = blockIdx.x * blockDim.x + threadIdx.x;

    const auto M = points.size(1);
    const auto T = sigma.size(1);

    if (c < M) {
        const auto t = tindex[n][c];

        assert(T == 1 || t < T);

        // Share one occupancy grid across time indices when T == 1.
        const auto ts = (T == 1) ? 0 : t;

        // Negative time indices mark padded rays.
        if (t < 0) return;

        const int vzsize = sigma.size(2);
        const int vysize = sigma.size(3);
        const int vxsize = sigma.size(4);

        const double xo = origin[n][t][0];
        const double yo = origin[n][t][1];
        const double zo = origin[n][t][2];

        const double xe = points[n][c][0];
        const double ye = points[n][c][1];
        const double ze = points[n][c][2];

        const int vxo = int(xo);
        const int vyo = int(yo);
        const int vzo = int(zo);

        int vx = vxo;
        int vy = vyo;
        int vz = vzo;

        const double rx = xe - xo;
        const double ry = ye - yo;
        const double rz = ze - zo;
        const double ray_length = sqrt(rx * rx + ry * ry + rz * rz);

        const double dx = rx / ray_length;
        const double dy = ry / ray_length;
        const double dz = rz / ray_length;

        const int stepX = (dx >= 0) ? 1 : -1;
        const int stepY = (dy >= 0) ? 1 : -1;
        const int stepZ = (dz >= 0) ? 1 : -1;

        // Coordinates of the next voxel boundary along each axis.
        const double next_voxel_boundary_x = vx + (stepX < 0 ? 0 : 1);
        const double next_voxel_boundary_y = vy + (stepY < 0 ? 0 : 1);
        const double next_voxel_boundary_z = vz + (stepZ < 0 ? 0 : 1);

        double tMaxX = (dx!=0) ? (next_voxel_boundary_x - xo)/dx : DBL_MAX;
        double tMaxY = (dy!=0) ? (next_voxel_boundary_y - yo)/dy : DBL_MAX;
        double tMaxZ = (dz!=0) ? (next_voxel_boundary_z - zo)/dz : DBL_MAX;

        // Ray distances between successive voxel boundaries along each axis.
        const double tDeltaX = (dx!=0) ? stepX/dx : DBL_MAX;
        const double tDeltaY = (dy!=0) ? stepY/dy : DBL_MAX;
        const double tDeltaZ = (dz!=0) ? stepZ/dz : DBL_MAX;

        int3 path[MAX_D];
        double d[MAX_D];

        // Traverse the grid with the 3D DDA algorithm.
        int step = 0;  // All traversal steps, including those outside the grid.
        int count = 0;  // Visited voxels inside the grid.

        bool was_inside = false;
        while (true) {
            bool inside = (0 <= vx && vx < vxsize) &&
                (0 <= vy && vy < vysize) &&
                (0 <= vz && vz < vzsize);
            if (inside) {
                was_inside = true;
                path[count] = make_int3(vx, vy, vz);
            } else if (was_inside) {
                // A straight ray cannot re-enter the grid after leaving it.
                break;
            }

            // Record the distance at which the ray exits each visited voxel.
            double _d = 0.0;
            if (tMaxX < tMaxY) {
                if (tMaxX < tMaxZ) {
                    _d = tMaxX;
                    vx += stepX;
                    tMaxX += tDeltaX;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            } else {
                if (tMaxY < tMaxZ) {
                    _d = tMaxY;
                    vy += stepY;
                    tMaxY += tDeltaY;
                } else {
                    _d = tMaxZ;
                    vz += stepZ;
                    tMaxZ += tDeltaZ;
                }
            }
            if (inside) {
                d[count] = _d;
                count ++;
            }
            step ++;

            if (step > MAX_STEP) {
                break;
            }
        }

        assert(count <= MAX_D);

        if (count > 0) {
            // Use the first occupied cell, or the final cell if the ray is empty.
            double exp_d = d[count-1];

            const int3 &v_init = path[count-1];
            int x = v_init.x;
            int y = v_init.y;
            int z = v_init.z;

            for (int i = 0; i < count; i++) {
                const int3 &v = path[i];
                const double occ = sigma[n][ts][v.z][v.y][v.x];
                if (occ > 0.5) {
                    exp_d = d[i];

                    x = v.x;
                    y = v.y;
                    z = v.z;

                    break;
                }

            }

            pred_dist[n][c] = exp_d;

            coord_index[n][c][0] = double(x);
            coord_index[n][c][1] = double(y);
            coord_index[n][c][2] = double(z);

        }
    }
}

// Cast rays through occupancy [B, T, Dz, Dy, Dx] in voxel-grid coordinates.
// Return distances [B, R] and voxel indices [B, R, 3].
std::vector<torch::Tensor> render_forward_cuda(
    torch::Tensor sigma,
    torch::Tensor origin,
    torch::Tensor points,
    torch::Tensor tindex) {

    const auto N = points.size(0);
    const auto M = points.size(1);

    const auto device = sigma.device();

    const int threads = 1024;
    const dim3 blocks((M + threads - 1) / threads, N);


    auto pred_dist = -torch::ones({N, M}, device);

    auto coord_index = torch::zeros({N, M, 3}, device);

    AT_DISPATCH_FLOATING_TYPES(sigma.scalar_type(), "render_forward_cuda", ([&] {
                render_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
                    sigma.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
                    origin.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    points.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
                    tindex.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    pred_dist.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                    coord_index.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>());
            }));

    cudaDeviceSynchronize();

    return {pred_dist, coord_index};
}
