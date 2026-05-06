from __future__ import annotations
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cdm_reader_mapper.common.iterators import ParquetStreamReader
from cdm_reader_mapper.core._utilities import (
    SubscriptableMethod,
    _copy,
    combine_attribute_values,
    method,
    reader_method,
)


def test_copy_dict():
    d = {"a": 1}
    assert d == _copy(d)


def test_copy_pandas():
    df = pd.DataFrame([{"a": 1, "b": 2}])
    pd.testing.assert_frame_equal(df, _copy(df))

    series = pd.Series([1, 2])
    pd.testing.assert_series_equal(series, _copy(series))


def test_copy_parquetstremareader():
    psr = ParquetStreamReader([pd.DataFrame([{"a": 1, "b": 2}]), pd.DataFrame([{"a": 3, "b": 4}])])
    copy = _copy(psr)

    pd.testing.assert_frame_equal(psr.read(), copy.read())


def test_copy_list():
    ll = [1, 2]
    assert ll == _copy(ll)


def test_copy_int():
    ii = 1
    assert ii == _copy(ii)


def test_method_callable():
    def f(x, y):
        return x + y

    result = method(f, 2, 3)

    assert result == 5


def test_method_subscriptable():
    data = {("a", "b"): 42}

    result = method(data, "a", "b")

    assert result == 42


def test_method_raises():
    class Dummy:
        pass

    obj = Dummy()

    with pytest.raises(ValueError, match="Attribute is neither callable nor subscriptable."):
        method(obj, 1)


@pytest.fixture
def test_psr():
    df1 = pd.DataFrame({"a": 1, "b": 2}, index=[0])
    df2 = pd.DataFrame({"a": 3, "b": 4}, index=[1])
    return ParquetStreamReader([df1, df2])


def test_reader_method_callable(test_psr):
    result = reader_method(
        test_psr,
        "sum",
        axis=1,
        process_kwargs={"non_data_output": "acc"},
    )

    expected = pd.Series({0: 3, 1: 7})

    pd.testing.assert_series_equal(result.read(), expected)


def test_reader_method_subscriptable(test_psr):
    result = reader_method(
        test_psr,
        "__getitem__",
        "a",
    )

    expected = pd.Series({0: 1, 1: 3}, name="a")

    pd.testing.assert_series_equal(result.read(), expected)


def test_reader_method_none(test_psr):
    result = reader_method(
        test_psr,
        "get",
        "false_attr",
    )

    assert result is None


def test_combine_attribute_values_index():
    first = pd.Index([1, 2])
    iterator = [
        SimpleNamespace(attr=pd.Index([2, 3])),
        SimpleNamespace(attr=pd.Index([3, 4])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    expected = pd.Index([1, 2, 3, 4])
    pd.testing.assert_index_equal(result, expected)


def test_combine_attribute_values_numeric():
    first = 10
    iterator = [SimpleNamespace(attr=5), SimpleNamespace(attr=3)]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == 18


def test_combine_attribute_values_tuple():
    first = (2, 5)
    iterator = [SimpleNamespace(attr=(3, 5)), SimpleNamespace(attr=(4, 5))]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == (9, 5)


def test_combine_attribute_values_list():
    first = [1, 2]
    iterator = [SimpleNamespace(attr=[3, 4]), SimpleNamespace(attr=[5])]

    result = combine_attribute_values(first, iterator, "attr")

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_combine_attribute_values_ndarray():
    first = np.array([1, 2])
    iterator = [
        SimpleNamespace(attr=np.array([3, 4])),
        SimpleNamespace(attr=np.array([5])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))


def test_combine_attribute_values_pandas():
    first = pd.Series([1, 2], index=[0, 1])
    iterator = [
        SimpleNamespace(attr=pd.Series([3, 4], index=[2, 3])),
        SimpleNamespace(attr=pd.Series([5], index=[4])),
    ]

    result = combine_attribute_values(first, iterator, "attr")

    pd.testing.assert_series_equal(result, pd.Series([1, 2, 3, 4, 5]))


def test_combine_attribute_values_default():
    first = "a"
    iterator = [SimpleNamespace(attr="b"), SimpleNamespace(attr="c")]

    result = combine_attribute_values(first, iterator, "attr")

    assert result == ["a", "b", "c"]


def test_subscriptablemethod_call():
    def f(x, y):
        return x + y

    sm = SubscriptableMethod(f)

    result = sm(2, 3)
    assert result == 5


def test_subscriptablemethod_getitem_passes():
    data = {"a": 1, "b": 2}

    sm = SubscriptableMethod(data)
    result = sm["a"]

    assert result == 1
    result = sm["b"]
    assert result == 2


def test_subscriptablemethod_getitem_raises():
    def f(x):
        return x * 2

    sm = SubscriptableMethod(f)

    with pytest.raises(
        NotImplementedError,
        match="Calling subscriptable methods have not been implemented",
    ):
        sm[0]
