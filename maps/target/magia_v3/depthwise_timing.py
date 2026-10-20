"""Rounded dispatch-cycle calibration, mirrored by MAPS SDK depthwise selection.

Geometry entries, vector blocks and DMA submissions explain the measured
shape dependence; these estimates exclude inter-tile transport. Calibration
covers the isolated MobileViT stage 12/27/75/79/153/157 shards, not stage IDs.
"""


def depthwise_estimates(channels: int, height: int, width: int,
                        kernel: int, stride_w: int) -> tuple[int, int]:
    if not channels or not height or not width or not kernel:
        return 0, 0
    row = 32 <= width <= 64
    blocks = height if row else (height * width + 63) // 64
    lanes = width if row else min(height * width, 64)
    geometry = kernel * (width if row else height * width)
    odd = stride_w % 2
    vector = (6000 + geometry * (76 + 10 * odd)
              + channels * blocks * (150 + kernel * (60 + lanes + 12 * odd)))
    band_rows = max(1, 65000 // (2 * (kernel + 1) * (width + 1)))
    bands = (height + band_rows - 1) // band_rows
    prepare = kernel * bands * 900 if stride_w == 1 else kernel * height * 420
    data = height * width * (kernel + 3)
    data = data // 2 if stride_w == 1 else data * 4 // 5
    im2col = 6000 + channels * (prepare + data
                              + bands * 1050)
    return vector, im2col


def depthwise_cycles(work: object) -> int:
    output = work.output_slice.dims
    channels = output[0].length * output[1].length
    height, width = (d.length for d in output[2:])
    kernel = work.kernel_shape[0] * work.kernel_shape[1]
    vector, im2col = depthwise_estimates(channels, height, width, kernel, work.strides[1])
    eligible = (work.input_slice.dims[3].length % 2 == 0 and
                height * width % 2 == 0 and kernel <= 49)
    return vector if eligible and vector * 6 < im2col * 5 else im2col
