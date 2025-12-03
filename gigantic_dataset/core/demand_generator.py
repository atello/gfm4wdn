#
# Created on Wed Feb 28 2024
# Copyright (c) 2024 Andrés Tello
#


import argparse

import numpy as np
import wntr
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from enum import Enum


class Seasonal(Enum):
    PeriodicOneDay = 0
    PeriodicRand = 1

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Seasonal[s]
        except KeyError:
            raise ValueError()


class Frequency(Enum):
    Daily = 365.
    Weekly = 52.
    Monthly = 12.
    Quarterly = 4.
    Semiannually = 2.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Frequency[s]
        except KeyError:
            raise ValueError()


def create_seasonal_component(time_step, duration, num_patterns, seasonal: Seasonal, freq: Frequency):
    """
    This method creates the seasonal component for the demand time series.
    :param time_step: sampling rate in minutes
    :param duration: total duration of the demand time series in hours
    :param num_patterns: number of time series patterns to create. It should equal to the number of nodes in the WDN.
    :param seasonal: Enum: [PeriodicOneDay, PeriodicRand]. Indicates if the seasonal component is created by repeating
        'PeriodicOneDay' pattern M times until reaching the specified 'duration' or use a sinusoidal function to
        generate random patterns along the x-axis.
    :param freq: Enum: [Daily, Weekly, Monthly, Quarterly, Semiannually]. This sets the frequency of the seasonal
        pattern in the time series. Using less frequent patterns (Monthly, ...) shows an increasing or decreasing
        sinus like slope if you plot the time series for just a few days. To obtain seasonal components which are
        clearly observable Daily or Weekly frequency should be used instead.
    :return: the seasonal component for the demand time series.
    """
    steps_per_hour = int(3600 / (time_step * 60))
    num_repetitions = duration // 24
    num_samples = int(steps_per_hour * duration)

    shift_y = np.random.uniform(0, 2, size=(num_patterns, 1))
    amplitude = np.random.uniform(1, 3, size=(num_patterns, 1))

    if seasonal == Seasonal.PeriodicOneDay.value:
        # one day pattern
        x_values = np.concatenate((np.random.uniform(0, 0.2, size=(num_patterns, steps_per_hour * 6)),
                                   np.random.uniform(0.6, 1.5, size=(num_patterns, steps_per_hour * 6)),
                                   np.random.uniform(0.4, 1.2, size=(num_patterns, steps_per_hour * 6)),
                                   np.random.uniform(0, 0.5, size=(num_patterns, steps_per_hour * 6))), axis=1)
        seasonal_pattern = shift_y + amplitude * np.sin(x_values)
        seasonal_pattern = np.tile(seasonal_pattern, num_repetitions)
        seasonal_pattern = savgol_filter(seasonal_pattern, steps_per_hour * 6, 3, axis=1)

    elif seasonal == Seasonal.PeriodicRand.value:
        # random sinusoidal function
        period_cos = np.random.uniform(1.5, 3, size=(num_patterns, 1))

        x = np.linspace(0, 2 * freq * np.pi, num_samples)
        seasonal_pattern = shift_y + amplitude * np.sin(x) + np.cos(period_cos * x)
    return seasonal_pattern


def create_random_pattern(time_step: int, duration: int, num_patterns: int, seasonal: Seasonal, freq: Frequency):
    """
    This method creates N demand patterns of duration T
    :param time_step: sampling rate in minutes
    :param duration: total duration of the demand time series in hours
    :param num_patterns: number of time series patterns to create. It should equal to the number of nodes in the WDN.
    :param seasonal: Enum: [PeriodicOneDay, PeriodicRand]. Indicates if the seasonal component is created by repeating
        'PeriodicOneDay' pattern M times until reaching the specified 'duration' or use a sinusoidal function to
        generate random patterns along the x-axis.
    :param freq: Enum: [Daily, Weekly, Monthly, Quarterly, Semiannually]. This sets the frequency of the seasonal
        pattern in the time series. Using less frequent patterns (Monthly, ...) shows an increasing or decreasing
        sinus like slope if you plot the time series for just a few days. To obtain seasonal components which are
        clearly observable Daily or Weekly frequency should be used instead.
    :return: a matrix of dimension [N,T] where N='num_patterns' and T is the total number of
        time steps to satisfy the specified 'duration'
    """
    steps_per_hour = int(3600 / (time_step * 60))
    num_samples = int(steps_per_hour * duration)

    seasonal_pattern = create_seasonal_component(time_step=time_step,
                                                 duration=duration,
                                                 num_patterns=num_patterns,
                                                 seasonal=seasonal,
                                                 freq=freq)

    # # normal
    locs = np.random.uniform(0.2, 0.6, size=(num_patterns, 1))
    scales = np.random.uniform(0.1, 0.5, size=(num_patterns, 1))
    noise = np.random.normal(loc=locs, scale=scales, size=(num_patterns, num_samples))

    trend = np.random.uniform(-1 * 10 ** -(np.log10(num_samples) + 1.5),
                              1 * 10 ** -(np.log10(num_samples) + 1.5),
                              size=(num_patterns, 1)) * range(num_samples)

    patterns = noise + trend + seasonal_pattern
    win_size = steps_per_hour * 2 if steps_per_hour * 2 > 3 else 4
    patterns_smoothed = savgol_filter(patterns, win_size, 3, axis=1)
    # patterns_smoothed = np.clip(patterns_smoothed, 0., np.max(patterns))

    mins = np.min(patterns_smoothed, axis=1).reshape(-1, 1)
    maxs = np.max(patterns_smoothed, axis=1).reshape(-1, 1)
    #min_target = np.random.uniform(0, 0.4, size=(num_patterns, 1))
    #max_target = np.random.uniform(0.95, 3, size=(num_patterns, 1))

    patterns_smoothed = ((patterns_smoothed - mins) / (maxs - mins)) # * (max_target - min_target) + min_target

    return patterns_smoothed


def plot_pattern(pattern, time_step: int, num_days: int):
    """
    Simple Plot of a time series using matplotlib library
    :param pattern: the time series to be plotted
    :param time_step: the time_step for every sample in the time series in minutes.
    :param num_days: number of days to be plotted
    """
    steps_per_hour = int(3600 / (time_step * 60))
    x = np.arange(24 * steps_per_hour * num_days)
    fig, ax = plt.subplots(figsize=(20, 5))
    plt.plot(x, pattern[:len(x)])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--time_step', help="time_step for the simulation of demand in 'minutes'", default=5, type=int)
    parser.add_argument('--duration', help="duration of the time steps in 'hours'.", default=8760, type=int)
    parser.add_argument('--num_patterns', help="total number of demands to create", default=15, type=int)
    parser.add_argument('--seasonal', help="version of the method to create the seasonal component of the time series",
                        type=lambda seasonal: Seasonal[seasonal], choices=list(Seasonal), default="PeriodicRand")
    parser.add_argument('--freq', help="frequency of the seasonal component",
                        type=lambda freq: Frequency[freq], choices=list(Frequency), default="Daily")

    args = parser.parse_args()

    print(args.freq.value)

    # Seasonal components test
    # patterns = create_seasonal_component(time_step=args.time_step,
    #                                      duration=args.duration,
    #                                      num_patterns=args.num_patterns,
    #                                      seasonal=args.seasonal.value,
    #                                      freq=args.freq.value)
    # for p in patterns:
    #     plot_pattern(p, time_step=args.time_step, num_days=21)

    # Patterns (trend + seasonal + noise) test.
    patterns = create_random_pattern(time_step=args.time_step,
                                     duration=args.duration,
                                     num_patterns=args.num_patterns,
                                     seasonal=args.seasonal.value,
                                     freq=args.freq.value)
    # for p in patterns:
    #     print(f"min: {np.min(p):.2f},\tmax:{np.max(p):.2f}")
    #     plot_pattern(p, time_step=args.time_step, num_days=365)
    #     plot_pattern(p, time_step=args.time_step, num_days=120)
    #     plot_pattern(p, time_step=args.time_step, num_days=30)
    #     plot_pattern(p, time_step=args.time_step, num_days=7)

    p : np.ndarray = np.vstack(patterns) 

    print(f'p shape = {p.shape}')
    print(f'p min = {np.min(p)}')
    print(f'p max = {np.max(p)}')
    print(f'p mean = {np.mean(p)}')
    print(f'p std = {np.std(p)}')
    
    
    #normed_p = (p - np.min(p)) / (np.max(p) - np.min(p))
    #plot_pattern(p, time_step=args.time_step, num_days=7)
    #plot_pattern(normed_p, time_step=args.time_step, num_days=7)