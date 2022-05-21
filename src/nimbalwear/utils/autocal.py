#van Hees, V. T., Fang, Z., Langford, J., Assah, F., Mohammad, A., M da Silva, I. C., Trenell, M. I., White, T.,
# Wareham, N. J., Brage, S., Hees,  van V., &#38; Silva,  da I. (2014). Autocalibration of accelerometer data for
# free-living physical activity assessment using local gravity and temperature: an evaluation on four continents.
# J Appl Physiol, 117, 738–744. https://doi.org/10.1152/japplphysiol.00421.2014.-Wearable

import math

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def autocal(x, y, z, accel_fs, temp=None, temp_fs=None, use_temp=True, epoch_secs=None, detect_only=False, plot=False,
            quiet=False):

    accel = np.vstack((x, y, z))

    if epoch_secs is None:

        if temp_fs - int(temp_fs) == 0:
            epoch_secs = 10
        else:
            factor = round(1 / (temp_fs - int(temp_fs)))
            epoch_secs = int((int(10 / factor) + 1) * factor)

    # epoch signals
    accel_epoch_samples = int(epoch_secs * accel_fs)
    temp_epoch_samples = int(epoch_secs * temp_fs)

    accel_epochs = accel[:, :int(int(accel.shape[1] / accel_epoch_samples) * accel_epoch_samples)]
    accel_epochs = accel_epochs.reshape((3, -1, accel_epoch_samples))

    temp_epochs = temp[:int(int(temp.shape[0] / temp_epoch_samples) * temp_epoch_samples)]
    temp_epochs = temp_epochs.reshape((-1, temp_epoch_samples))

    epoch_count = min(accel_epochs.shape[1], temp_epochs.shape[0])

    accel_epochs = accel_epochs[:, :epoch_count, :]
    temp_epochs = temp_epochs[:epoch_count, :]

    # calculate epoch means and std
    accel_epoch_means = accel_epochs.mean(axis=2)
    accel_epoch_stds = accel_epochs.std(axis=2)

    temp_epoch_means = temp_epochs.mean(axis=1)

    # identify resting epochs based on accelerometer std
    rest_epoch_idx = np.where((accel_epoch_stds < 0.013).all(axis=0))[0]

    # select means of rest epochs
    accel_epoch_means_rest = accel_epoch_means[:, rest_epoch_idx]
    temp_epoch_means_rest = temp_epoch_means[rest_epoch_idx]

    # calculate mean calibration error
    rest_epoch_vm = np.sqrt(np.square(accel_epoch_means_rest).sum(axis=0))
    pre_err = round(abs(rest_epoch_vm - 1).mean() * 1000, 2)
    if not quiet:
        print("Pre-calibration error: " + str(pre_err))

    post_err = None
    iter = None

    if not detect_only:

        # autocalibrate
        input = accel_epoch_means_rest.T
        inputtemp = np.vstack((temp_epoch_means_rest, temp_epoch_means_rest, temp_epoch_means_rest)).T

        meantemp = inputtemp[:, 0].mean()

        inputtemp = inputtemp - meantemp

        offset = np.zeros(input.shape[1])
        scale = np.ones(input.shape[1])

        tempoffset = np.zeros(input.shape[1])

        weights = np.ones(input.shape[0])

        res = np.array([math.inf])

        maxiter = 1000

        tol = 1e-10

        for iter in range(maxiter):

            curr = np.multiply(input + offset, scale) + np.multiply(inputtemp, tempoffset)

            closestpoint = curr / np.sqrt(np.square(curr).sum(axis=1))[:, None]

            offsetch = np.zeros(input.shape[1])
            scalech = np.ones(input.shape[1])

            toffch = np.zeros(inputtemp.shape[1])

            for k in range(input.shape[1]):

                lm_X = np.vstack((curr[:, k], inputtemp[:, k])).T
                lm_y = closestpoint[:, k]
                lm_w = weights

                fobj = LinearRegression().fit(lm_X, lm_y, lm_w)

                offsetch[k] = fobj.intercept_

                scalech[k] = fobj.coef_[0]

                if use_temp:
                    toffch[k] = fobj.coef_[1]

                curr[:, k] = fobj.predict(lm_X)

            offset = offset + offsetch / (scale * scalech)

            if use_temp:
                tempoffset = tempoffset * scalech + toffch

            scale = scale * scalech

            res = np.append(res, 3 * (weights[:, None] * np.square(curr - closestpoint) / weights.sum()).mean())

            weights = np.minimum(1 / np.sqrt(np.square(curr - closestpoint).sum(axis=1)), 1 / 0.01)

            if (abs(res[iter + 1] - res[iter]) < tol):
                break

        temp_fill = np.repeat(temp, int(accel_fs / temp_fs))

        pts = min(accel.shape[1], len(temp_fill))
        temp_fill = temp_fill[:pts]
        accel = accel[:, :pts]

        temp_fill = np.vstack((temp_fill, temp_fill, temp_fill)).T

        calib_accel = (offset + (accel.T * scale) + ((temp_fill - meantemp) * tempoffset)).T

        # calculate vector magnitude for uncalibrated and calibrated accel
        vm = abs(np.sqrt(np.square(accel).sum(axis=0)) - 1)
        calib_vm = abs(np.sqrt(np.square(calib_accel).sum(axis=0)) - 1)

        # find means of rest epochs of calibrated data
        calib_accel_epochs = calib_accel[:,
                             :int(int(accel.shape[1] / accel_epoch_samples) * accel_epoch_samples)].reshape(
            (3, -1, accel_epoch_samples))
        calib_accel_epoch_means = calib_accel_epochs.mean(axis=2)
        calib_accel_epoch_means_rest = calib_accel_epoch_means[:, rest_epoch_idx]

        # find error of calibrated data
        calib_rest_epoch_vm = np.sqrt(np.square(calib_accel_epoch_means_rest).sum(axis=0))
        post_err = round(abs(calib_rest_epoch_vm - 1).mean() * 1000, 2)

        if not quiet:
            print("Post-calibration error: " + str(post_err))

        x = calib_accel[0]
        y = calib_accel[1]
        z = calib_accel[2]

        if plot:
            # plot
            plt.figure()
            plt.plot(vm, linewidth=0.25, color='lightcoral')
            plt.plot(calib_vm, linewidth=0.25, color='dodgerblue')
            plt.show()

    return x, y, z, pre_err, post_err, iter
