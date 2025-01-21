"""Common Data Model (CDM) reader properties."""

from __future__ import annotations

from ..properties import numeric_types, object_types, supported_data_models  # noqa

_base = "cdm_reader_mapper.mdf_reader"

open_file = {
    "craid": "netcdf",
}

year_column = {
    "gcc": "YR",
    "icoads": ("core", "YR"),
    "craid": ("drifter_measurements", "JULD"),
}

pandas_dtypes = {}
for dtype in object_types:
    pandas_dtypes[dtype] = "object"
pandas_dtypes.update({x: x for x in numeric_types})
pandas_dtypes["datetime"] = "datetime"

pandas_int = "Int64"

# ....and how they are managed
data_type_conversion_args = {}
for dtype in numeric_types:
    data_type_conversion_args[dtype] = ["scale", "offset"]
data_type_conversion_args["str"] = ["disable_white_strip"]
data_type_conversion_args["object"] = ["disable_white_strip"]
data_type_conversion_args["key"] = ["disable_white_strip"]
data_type_conversion_args["datetime"] = ["datetime_format"]

# Misc ------------------------------------------------------------------------
dummy_level = "_SECTION_"
# Length of reports in initial read
MAX_FULL_REPORT_WIDTH = 100000
