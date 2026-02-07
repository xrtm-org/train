
import numpy as np

from xrtm.train.core.eval.calibration import BetaScaler, PlattScaler


class TestPlattScaler:
    def test_transform_scalar(self):
        scaler = PlattScaler(a=1.0, b=0.0, fitted=True)
        # identity transform
        result = scaler.transform(0.5)
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # With a=1, b=0, logit(0.5)=0, 1/(1+exp(0))=0.5
        assert np.isclose(result, 0.5)

    def test_transform_list(self):
        scaler = PlattScaler(a=1.0, b=0.0, fitted=True)
        result = scaler.transform([0.5, 0.8])
        assert isinstance(result, list)
        assert len(result) == 2
        assert np.isclose(result[0], 0.5)

    def test_transform_numpy(self):
        scaler = PlattScaler(a=1.0, b=0.0, fitted=True)
        arr = np.array([0.5, 0.8])
        # The type hint says Union[float, List[float]], but let's check if it handles numpy array
        # The implementation uses np.array(y_prob), which handles it.
        # However, the return type is documented as Union[float, List[float]].
        # The implementation returns p_calib.tolist() if not scalar.
        result = scaler.transform(arr)
        assert isinstance(result, list)
        assert len(result) == 2
        assert np.isclose(result[0], 0.5)

    def test_not_fitted(self):
        scaler = PlattScaler()
        assert scaler.transform(0.5) == 0.5
        assert scaler.transform([0.5]) == [0.5]


class TestBetaScaler:
    def test_transform_scalar(self):
        scaler = BetaScaler(a=1.0, b=1.0, c=0.0, fitted=True)
        # identity transform for beta calibration with a=1, b=1, c=0
        # logit = 1*log(p) - 1*log(1-p) + 0 = log(p/(1-p)) -> same as identity
        result = scaler.transform(0.5)
        assert isinstance(result, float)
        assert np.isclose(result, 0.5)

    def test_transform_list(self):
        scaler = BetaScaler(a=1.0, b=1.0, c=0.0, fitted=True)
        result = scaler.transform([0.5, 0.8])
        assert isinstance(result, list)
        assert np.isclose(result[0], 0.5)

    def test_transform_numpy(self):
        scaler = BetaScaler(a=1.0, b=1.0, c=0.0, fitted=True)
        arr = np.array([0.5, 0.8])
        result = scaler.transform(arr)
        assert isinstance(result, list)
        assert np.isclose(result[0], 0.5)
