
import numpy as np
import pytest

from xrtm.train.core.eval.calibration import BetaScaler, PlattScaler


class TestBetaScaler:
    @pytest.fixture
    def scaler(self):
        scaler = BetaScaler()
        scaler.fitted = True
        scaler.a = 1.0
        scaler.b = 1.0
        scaler.c = 0.0
        return scaler

    def test_transform_scalar(self, scaler):
        y_prob = 0.5
        result = scaler.transform(y_prob)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # With a=1, b=1, c=0, transform(0.5) should be 0.5
        # log(0.5) - log(0.5) + 0 = 0 -> sigmoid(0) = 0.5
        assert result == 0.5

    def test_transform_list(self, scaler):
        y_prob = [0.2, 0.5, 0.8]
        result = scaler.transform(y_prob)
        assert isinstance(result, list)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

        assert result[1] == 0.5

    def test_transform_numpy_array(self, scaler):
        y_prob = np.array([0.2, 0.5, 0.8])
        result = scaler.transform(y_prob)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[1] == 0.5

    def test_transform_not_fitted(self):
        scaler = BetaScaler()
        # Not fitted, should return input as is
        assert scaler.transform(0.5) == 0.5
        assert scaler.transform([0.5]) == [0.5]

    def test_transform_edge_cases(self, scaler):
        # 0.0 and 1.0 should be clipped
        y_prob = [0.0, 1.0]
        result = scaler.transform(y_prob)
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0

class TestPlattScaler:
    @pytest.fixture
    def scaler(self):
        scaler = PlattScaler()
        scaler.fitted = True
        scaler.a = 1.0
        scaler.b = 0.0
        return scaler

    def test_transform_scalar(self, scaler):
        y_prob = 0.5
        result = scaler.transform(y_prob)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # log(0.5 / 0.5) = log(1) = 0
        # 1.0 * 0 + 0 = 0
        # sigmoid(0) = 0.5
        assert result == 0.5

    def test_transform_list(self, scaler):
        y_prob = [0.2, 0.5, 0.8]
        result = scaler.transform(y_prob)
        assert isinstance(result, list)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

        assert result[1] == 0.5

    def test_transform_numpy_array(self, scaler):
        y_prob = np.array([0.2, 0.5, 0.8])
        result = scaler.transform(y_prob)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[1] == 0.5
