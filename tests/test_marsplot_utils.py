import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
_argv = sys.argv
try:
    # MarsPlot parses the command line at import time
    sys.argv = ['MarsPlot']
    from bin import MarsPlot
finally:
    sys.argv = _argv


class FakeVariable:
    def __init__(self, data):
        self.data = np.ma.asarray(data)

    def __getitem__(self, key):
        return self.data[key]


class TestAreoByTime(unittest.TestCase):
    def test_average_file_areo(self):
        areo = FakeVariable(np.array([[10.0], [20.0], [30.0]]))
        np.testing.assert_array_equal(MarsPlot.areo_by_time(areo),
                                      [10.0, 20.0, 30.0])

    def test_diurn_file_uses_midnight_value(self):
        areo = FakeVariable(np.array([[[10.0], [10.5]], [[20.0], [20.5]]]))
        np.testing.assert_array_equal(MarsPlot.areo_by_time(areo),
                                      [10.0, 20.0])

    def test_single_time_step_diurn_file_keeps_time_axis(self):
        # Regression: np.squeeze dropped the time axis, which broke
        # plots of single-record diurn files (e.g., after MarsFiles -split)
        areo = FakeVariable(np.array([[[268.1], [268.4]]]))
        ls = MarsPlot.areo_by_time(areo)
        self.assertEqual(ls.shape, (1,))
        self.assertEqual(float(ls[0]), 268.1)


class TestGetTimeIndex(unittest.TestCase):
    def test_single_time_step(self):
        ti, txt = MarsPlot.get_time_index(270, np.array([268.1]))
        self.assertEqual(int(ti), 0)
        self.assertIn('268.10', txt)

    def test_scalar_time_step(self):
        ti, txt = MarsPlot.get_time_index(270, np.float64(268.1))
        self.assertEqual(int(ti), 0)
        self.assertIn('268.10', txt)


class TestTopographyForOverlay(unittest.TestCase):
    lon = np.array([45.0, 135.0, 225.0, 315.0])
    zsurf = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]])

    def test_matching_grid_is_shifted_like_the_data(self):
        # Regression: topography was never drawn on lon X lat plots
        _, var = MarsPlot.shift_data(self.lon, self.zsurf * 10.)
        topo = MarsPlot.topography_for_overlay(self.lon, self.zsurf, var)
        _, expected = MarsPlot.shift_data(self.lon, self.zsurf)
        np.testing.assert_array_equal(topo, expected)

    def test_no_fixed_file(self):
        var = np.zeros((2, 4))
        self.assertIsNone(
            MarsPlot.topography_for_overlay(self.lon, None, var))

    def test_different_grid_is_skipped(self):
        # e.g., data regridded to a different resolution
        var = np.zeros((3, 4))
        self.assertIsNone(
            MarsPlot.topography_for_overlay(self.lon, self.zsurf, var))


class TestLongitudeRange(unittest.TestCase):
    lons = np.arange(1.875, 360., 3.75)    # FV3-based MGCM grid

    def test_range_across_prime_meridian(self):
        # Regression: -45,45 averaged the 270 deg on the far side
        loni, txt = MarsPlot.get_lon_index(np.array([-45., 45.]), self.lons)
        self.assertEqual(len(loni), 25)
        self.assertAlmostEqual(float(self.lons[loni[0]]), 313.125)
        self.assertAlmostEqual(float(self.lons[loni[-1]]), 43.125)
        self.assertIn('-46.9<->43.1', txt)

    def test_range_across_dateline_keeps_direction(self):
        loni, txt = MarsPlot.get_lon_index(np.array([160., -40.]), self.lons)
        self.assertEqual(len(loni), 44)
        self.assertIn('159.4<->-39.4', txt)

    def test_range_past_180_keeps_0_360_label(self):
        # Regression: 100,200 was labeled 99.0<->-161.0
        loni, txt = MarsPlot.get_lon_index(np.array([100., 200.]), self.lons)
        self.assertEqual(len(loni), 28)
        self.assertIn('99.4<->200.6', txt)

    def test_single_lon_keeps_requested_convention(self):
        _, txt = MarsPlot.get_lon_index(np.array(200.), self.lons)
        self.assertIn('lon=200.6', txt)
        _, txt = MarsPlot.get_lon_index(np.array(-160.), self.lons)
        self.assertIn('lon=-159.4', txt)

    def test_full_circle_counts_each_longitude_once(self):
        # Regression: both ends snapped to one point, and the wrap
        # counted that point twice (Nlon + 1 indices)
        for req in ([-180., 180.], [0., 360.]):
            loni, txt = MarsPlot.get_lon_index(np.array(req), self.lons)
            np.testing.assert_array_equal(np.sort(loni),
                                          np.arange(len(self.lons)))
            self.assertIn('zonal avg', txt)

    def test_single_point_range(self):
        loni, _ = MarsPlot.get_lon_index(np.array([10., 10.]), self.lons)
        self.assertEqual(np.atleast_1d(loni).size, 1)

    def test_range_within_eastern_hemisphere(self):
        loni, _ = MarsPlot.get_lon_index(np.array([10., 50.]), self.lons)
        self.assertEqual(len(loni), 12)


class FakeAxis:
    def __init__(self, data, units=None):
        self.data = np.asarray(data)
        if units is not None:
            self.units = units

    def __getitem__(self, key):
        return self.data[key]


class FakeFile:
    def __init__(self, **variables):
        self.variables = variables


class TestVerticalLevels(unittest.TestCase):
    def test_pfull_in_mb_is_converted_to_pa(self):
        # Regression: pfull in mb was plotted on an axis labeled Pa
        f = FakeFile(pfull=FakeAxis([0.01, 1.0, 7.0], units='mb'))
        np.testing.assert_allclose(MarsPlot.vertical_levels(f, 'pfull'),
                                   [1.0, 100.0, 700.0])

    def test_pstd_in_pa_is_unchanged(self):
        f = FakeFile(pstd=FakeAxis([1000.0, 50.0], units='Pa'))
        np.testing.assert_array_equal(MarsPlot.vertical_levels(f, 'pstd'),
                                      [1000.0, 50.0])

    def test_altitude_and_no_units_are_unchanged(self):
        f = FakeFile(zstd=FakeAxis([0.0, 5000.0], units='m'),
                     zagl=FakeAxis([0.0, 15.0]))
        np.testing.assert_array_equal(MarsPlot.vertical_levels(f, 'zstd'),
                                      [0.0, 5000.0])
        np.testing.assert_array_equal(MarsPlot.vertical_levels(f, 'zagl'),
                                      [0.0, 15.0])


class TestDiurnMean(unittest.TestCase):
    # [time=1, tod=2, lat=1, lon=2]; one below-ground NaN at tod 0, lon 1.
    # Longitude first: tod 0 -> 1, tod 1 -> (3 + 5)/2 = 4, mean = 2.5.
    # Time of day first: lon 0 -> 2, lon 1 -> 5, mean = 3.5.
    var = np.array([[[[1.0, np.nan]], [[3.0, 5.0]]]])

    def test_zonal_mean_plots_average_longitude_first(self):
        for plot_type in MarsPlot.ZONAL_THEN_TIME_PLOTS:
            out = MarsPlot.diurn_mean(self.var, plot_type)
            self.assertEqual(out.shape, (1, 1, 1))
            self.assertAlmostEqual(float(np.nanmean(out)), 2.5)

    def test_other_plots_keep_longitude(self):
        out = MarsPlot.diurn_mean(self.var, "2D_lon_lat")
        self.assertEqual(out.shape, (1, 1, 2))
        np.testing.assert_array_equal(out[0, 0], [2.0, 5.0])

    def test_mean_func_keepdims(self):
        out = MarsPlot.mean_func(self.var, axis = -1, keepdims = True)
        self.assertEqual(out.shape, (1, 2, 1, 1))

    def test_area_mean_plots_average_space_first(self):
        # [time=1, tod=2, lat=2, lon=2]; a 1-lat slice behaves as above
        var = np.concatenate([self.var, self.var], axis = 2)
        lat = np.array([-1.0, 1.0])
        for plot_type in MarsPlot.AREA_THEN_TIME_PLOTS:
            out = MarsPlot.diurn_mean(var, plot_type, lat)
            self.assertEqual(out.shape, (1, 1, 1))
            self.assertAlmostEqual(float(out.squeeze()), 2.5)

    def test_meridional_mean_plots_average_latitude_first(self):
        # [time=1, tod=2, lat=2, lon=1]; NaN at tod 0, lat 1.
        # Latitude first: tod 0 -> 1, tod 1 -> 4, mean = 2.5
        var = np.array([[[[1.0], [np.nan]], [[3.0], [5.0]]]])
        lat = np.array([-1.0, 1.0])
        for plot_type in MarsPlot.MERIDIONAL_THEN_TIME_PLOTS:
            out = MarsPlot.diurn_mean(var, plot_type, lat)
            self.assertEqual(out.shape, (1, 1, 1))
            self.assertAlmostEqual(float(out.squeeze()), 2.5)


class TestAreaMean(unittest.TestCase):
    lat = np.array([0.0, 60.0])

    def test_lat_mean_is_area_weighted(self):
        var = np.array([[0.0], [1.0]])             # [lat, lon]
        w = MarsPlot.area_weights_deg((2, 1), self.lat)[:, 0]
        out = MarsPlot.lat_mean(var, self.lat, axis = 0)
        self.assertAlmostEqual(float(out[0]), w[1] / w.sum())
        self.assertLess(float(out[0]), 0.5)        # pole-ward cell is smaller

    def test_lat_mean_skips_nans(self):
        var = np.array([[2.0], [np.nan]])          # [lat, lon]
        out = MarsPlot.lat_mean(var, self.lat, axis = 0)
        self.assertEqual(float(out[0]), 2.0)

    def test_lat_mean_skips_masked_values(self):
        var = np.ma.masked_array([2.0, 7.0], mask = [False, True])
        out = MarsPlot.lat_mean(var, self.lat, axis = 0)
        self.assertEqual(float(out), 2.0)

    def test_lat_mean_all_nan_is_nan(self):
        out = MarsPlot.lat_mean(np.array([np.nan, np.nan]), self.lat, axis = 0)
        self.assertTrue(np.isnan(out))

    def test_area_mean_averages_longitude_first(self):
        # [lat=2, lon=2]; one below-ground NaN at lat 1, lon 1.
        # Longitude first: lat 0 -> 1, lat 1 -> 3; then weighted by area.
        # Latitude first would give lon 0 -> weighted(1, 3), lon 1 -> 1.
        var = np.array([[1.0, 1.0], [3.0, np.nan]])
        w = MarsPlot.area_weights_deg((2, 1), self.lat)[:, 0]
        expected = (1.0 * w[0] + 3.0 * w[1]) / w.sum()
        out = MarsPlot.area_mean(var, self.lat)
        self.assertAlmostEqual(float(out), expected)


if __name__ == '__main__':
    unittest.main()
