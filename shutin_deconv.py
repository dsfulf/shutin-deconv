
from petbox import dca
import numpy as np
from numba import jit
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy.typing import NDArray

NDFloat = NDArray[np.float64]

plt.rcParams['font.size'] = 14
plt.rcParams['figure.figsize'] = (12, 10)

# create a convolution function that does *linear* convolution


@jit
def convolve(x: NDFloat, k: NDFloat) -> NDFloat:
    # create an array of zeros to store our result
    result = np.zeros_like(x)
    b = len(k) - 1

    for i in range(len(x)):
        hi = min(b, i)  # hi is the largest element of our kernel
        lo = max(0, i - b)  # lo is the element corresponding to the smallest element of our kernel
        for j in range(i + 1):
            if hi - j < 0:
                break

            result[i] += x[lo + j] * k[hi - j]

    return result


@jit
def convolve_stepwise(x: NDFloat, k: NDFloat, start: int, end: int) -> None:
    b = len(k) - 1

    for i in range(start, min(end + 1, len(x))):
        conv = 0
        hi = min(b, i)  # hi is the largest element of our kernel
        lo = max(0, i - b)  # lo is the element corresponding to the smallest element of our kernel
        for j in range(i + 1):
            if hi - j < 0:
                break

            conv += x[lo + j] * k[hi - j]

        # this is equivalent to k[i] = 0 - conv, meaning we are solving for
        # the kernel value that gives x = 0
        k[i] -= conv


def make_kernel(time: NDFloat) -> NDFloat:
    """
    Create a kernel to put a well on production
    """
    # create a kernel that is the same length as the time array
    kernel = np.zeros_like(time)

    # pwf is intial pressure before well is put on production
    kernel[time <= 0] = 0

    # get first time index where time > 0
    idx = np.argmin(time > 0)

    # We need to create an impulse to "start" the well onto production
    kernel[idx] = 1

    return kernel


def add_shutin_deconv(
    kernel: NDFloat, rate: NDFloat, time: NDFloat,
    start_time: int, duration: int, buildup_time: int
) -> None:
    """
    Add a shut-in period to our kernel
    """
    start_idx = np.where(time == start_time)[0][0]
    end_idx = np.where(time == start_time + duration)[0][0]
    build_end_idx = np.where(time == start_time + duration + buildup_time)[0][0]
    shutin_slice = slice(start_idx, end_idx)
    build_slice = slice(end_idx, build_end_idx)

    # create a dimensionless rate array to simplify calculations
    dimen_rate = rate / np.max(rate)

    # get the sum of impulses that already exist in the kernel
    impulses = np.sum(kernel[start_idx:])

    # We perform convolution step-by-step, making the kernel equal to the
    # prior convolution result at each step, which will result in zeroes
    # when the kernel is convolved. This is effectively a deconvolution.
    convolve_stepwise(dimen_rate, kernel, start_idx, end_idx)

    # lastly, we need to "makeup" for the shut-in by adding the rate back in
    # over the shut-in period
    kernel[build_slice] += (impulses - np.sum(kernel[shutin_slice])) / buildup_time


def add_shutin(
    kernel: NDFloat, rate: NDFloat, time: NDFloat,
    start_time: int, duration: int, buildup_time: int
) -> None:
    """
    Add a shut-in period to our kernel
    """
    start_idx = np.where(time == start_time)[0][0]
    end_idx = np.where(time == start_time + duration)[0][0]
    duration_idx = np.where(time == duration)[0][0]
    build_end_idx = np.where(time == start_time + duration + buildup_time)[0][0]
    shutin_slice = slice(start_idx, end_idx)
    build_slice = slice(end_idx, build_end_idx)
    # shut-in the well

    # create a dimensionless rate array to simplify calculations
    dimen_rate = rate / np.max(rate)

    # We need to create a shut-in event, which is a negative impulse
    # with magnitude equal difference in rate between shut-in time and start time.
    # Since our array is dimensionless and is already in units of "% of initial rate",
    # we simply index the array.
    shutin_impulse = -dimen_rate[start_idx]
    kernel[start_idx] += shutin_impulse

    # we need the start of the buildup to be forced back to zero
    impulse_adj = -dimen_rate[end_idx]
    impulse_adj -= shutin_impulse * dimen_rate[duration_idx]
    kernel[end_idx] += impulse_adj

    # lastly, we need to "makeup" for the shut-in by adding the rate back in
    # over the shut-in period
    kernel[build_slice] += (-shutin_impulse - impulse_adj) / buildup_time


def main() -> None:
    # create some arbitrary decline function
    qi = 1000.0
    Di = 0.80
    b = 0.9
    Dterm = 0.06
    mh = dca.MH(qi, Di, b, Dterm)

    evaluation_time = int(round(50 * 365.25, 0))

    time = np.linspace(0, evaluation_time, evaluation_time + 1)
    rate = mh.rate(time)

    start_time = 90
    duration = 60
    buildup_time = 60

    # build simply kernel
    kernel = make_kernel(time)
    add_shutin(kernel, rate, time, start_time, duration, buildup_time)
    rate_conv = convolve(rate, kernel)

    # perform deconvolution to determine the kernel
    kernel_deconv = make_kernel(time)
    add_shutin_deconv(kernel_deconv, rate, time, start_time, duration, buildup_time)
    rate_deconv = convolve(rate, kernel_deconv)

    # more shut-in periods
    kernel_lots = make_kernel(time)
    add_shutin_deconv(kernel_lots, rate, time, start_time, duration, buildup_time)
    add_shutin_deconv(kernel_lots, rate, time, start_time + duration + 30, duration, buildup_time)
    add_shutin_deconv(kernel_lots, rate, time, start_time + duration + 200, duration, buildup_time)
    rate_lots = convolve(rate, kernel_lots)

    # plot the results
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, rate, lw=1.5, label=f'Rate, EUR = {np.sum(rate) / 1000 :,.0f} MBbl')
    ax.plot(time, rate_conv, 'o', mec='C1', mfc='w', ms=6, lw=1.5,
            label=f'Convolved Rate (Naive Kernel), EUR = {np.sum(rate_conv) / 1000:,.0f} MBbl')
    ax.plot(time, rate_deconv, 'o', mec='C2', mfc='w', ms=3, lw=1.5,
            label=f'Convolved Rate (Deconvolved Kernel), EUR = {np.sum(rate_deconv) / 1000:,.0f} MBbl')
    ax.plot(time, rate_lots, 'o', mec='C3', mfc='w', ms=2, lw=1.5,
            label=f'More Shut-Ins, EUR = {np.sum(rate_lots) / 1000:,.0f} MBbl')

    ax2 = ax.twinx()
    p_deconv = 5000 * (1 - np.cumsum(kernel_deconv)) + 1000
    p_lots = 5000 * (1 - np.cumsum(kernel_lots)) + 1000

    # ax2.plot(time, p_deconv, c='C2', lw=3, label='Deconvolved Pressure (Kernel)')
    # ax2.plot(time, p_lots, c='C3', lw=3, label='More Shut-Ins')

    ax.set(ylabel='Rate, MMcfd', xlabel='Time, days', xlim=(0, 1000), ylim=(0, None))
    ax2.set(ylabel='Pressure, psi', ylim=(0, 7000))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend([*h1, *h2], [*l1, *l2], loc='upper right', ncol=1)

    plt.savefig('shutin_deconv.png', dpi=300)


    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(time, rate, lw=1.5, label=f'Rate, EUR = {np.sum(rate) / 1000 :,.0f} MBbl')
    ax.plot(time, rate_lots, 'o', mec='C3', mfc='w', ms=2, lw=1.5,
            label=f'Convolved Rate (Deconvolved Kernel), EUR = {np.sum(rate_lots) / 1000:,.0f} MBbl')

    ax2 = ax.twinx()
    p_deconv = 8000 * (1 - np.cumsum(kernel_deconv)) + 1000
    p_lots = 8000 * (1 - np.cumsum(kernel_lots)) + 1000

    ax2.plot(time, p_lots, c='C4', lw=2, label='Deconvolved Pressure (Kernel)')

    ax.set(ylabel='Rate, MMcfd', xlabel='Time, days', xlim=(0, 1000), ylim=(0, None))
    ax2.set(ylabel='Pressure, psi', ylim=(0, 7000))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend([*h1, *h2], [*l1, *l2], loc='upper right', ncol=1)

    plt.savefig('shutin_deconv_pressure.png', dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
