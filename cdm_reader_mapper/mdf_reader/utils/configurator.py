"""Auxiliary functions and class for reading, converting, decoding and validating MDF files."""

from __future__ import annotations

import ast
import csv
import logging

import numpy as np
import pandas as pd

from .. import properties
from . import converters, decoders
from .utilities import convert_dtypes


class Configurator:
    """Class for configuring MDF reader information."""

    def __init__(
        self,
        df=pd.DataFrame(),
        schema={},
        order=[],
        valid=[],
    ):
        self.df = df
        self.orders = order
        self.valid = valid
        self.schema = schema

    def _add_field_length(self, index, sections_dict):
        field_length = sections_dict.get(
            "field_length", properties.MAX_FULL_REPORT_WIDTH
        )
        return index + field_length

    def _validate_sentinal(self, i, line, sentinal):
        slen = len(sentinal)
        str_start = line[i : i + slen]
        return str_start == sentinal

    def _get_index(self, section, order):
        if len(self.orders) == 1:
            return section
        else:
            return (order, section)

    def _get_ignore(self, section_dict):
        ignore = section_dict.get("ignore")
        if isinstance(ignore, str):
            ignore = ast.literal_eval(ignore)
        return ignore

    def _get_dtype(self, section_dict):
        return properties.pandas_dtypes.get(section_dict.get("column_type"))

    def _get_converter(self, section_dict):
        return converters.get(section_dict.get("column_type"))

    def _get_conv_kwargs(self, section_dict):
        column_type = section_dict.get("column_type")
        if column_type is None:
            return
        return {
            converter_arg: section_dict.get(converter_arg)
            for converter_arg in properties.data_type_conversion_args.get(column_type)
        }

    def _get_decoder(self, section_dict):
        encoding = section_dict.get("encoding")
        if encoding is None:
            return
        column_type = section_dict.get("column_type")
        if column_type is None:
            return
        return decoders.get(encoding).get(column_type)

    def _update_dtypes(self, section_dict, dtypes, index):
        dtype = self._get_dtype(section_dict)
        if dtype:
            dtypes[index] = dtype
        return dtypes

    def _update_converters(self, section_dict, converters, index):
        converter = self._get_converter(section_dict)
        if converter:
            converters[index] = converter
        return converters

    def _update_kwargs(self, section_dict, kwargs, index):
        conv_kwargs = self._get_conv_kwargs(section_dict)
        if conv_kwargs:
            kwargs[index] = conv_kwargs
        return kwargs

    def _update_decoders(self, section_dict, decoders, index):
        decoder = self._get_decoder(section_dict)
        if decoder:
            decoders[index] = decoder
        return decoders

    def _get_bad_sentinal(self, i, line, sentinal):
        return sentinal is not None and not self._validate_sentinal(i, line, sentinal)

    def _read_line(self, line: str):
        missing_values = []
        i = 0
        j = 0
        data_dict = {}
        for order in self.orders:
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                data_dict[order] = line[i : properties.MAX_FULL_REPORT_WIDTH]
                continue

            sentinal = header.get("sentinal")
            bad_sentinal = self._get_bad_sentinal(i, line, sentinal)
            section_length = header.get("length", properties.MAX_FULL_REPORT_WIDTH)
            sections = self.schema["sections"][order]["elements"]
            field_layout = header.get("field_layout")
            delimiter = header.get("delimiter")

            if delimiter is not None:
                delimiter_format = header.get("format")
                if delimiter_format == "delimited":
                    field_names = sections.keys()
                    fields = list(csv.reader([line[i:]], delimiter=delimiter))[0]
                    for field_name, field in zip(field_names, fields):
                        index = self._get_index(field_name, order)
                        data_dict[index] = field.strip()
                        i += len(field)
                    j = i
                    continue
                elif field_layout != "fixed_width":
                    logging.error(
                        f"Delimiter for {order} is set to {delimiter}. Please specify either format or field_layout in your header schema {header}."
                    )
                    return
            k = i + section_length

            for section, section_dict in sections.items():
                missing = True
                self.sections_dict = sections[section]
                index = self._get_index(section, order)
                ignore = (order not in self.valid) or self._get_ignore(section_dict)
                na_value = sections[section].get("missing_value")
                field_length = section_dict.get(
                    "field_length", properties.MAX_FULL_REPORT_WIDTH
                )

                j = (i + field_length) if not bad_sentinal else i

                if j > k:
                    missing = False
                    j = k

                if ignore is not True:
                    data_dict[index] = line[i:j]

                    if not data_dict[index].strip():
                        data_dict[index] = None
                    if data_dict[index] == na_value:
                        data_dict[index] = None

                if i == j and missing is True:
                    missing_values.append(index)

                if delimiter is not None and line[j : j + len(delimiter)] == delimiter:
                    j += len(delimiter)

                i = j

        df = pd.Series(data_dict)
        df["missing_values"] = missing_values
        return df

    def get_configuration(self):
        """Get ICOADS data model specific information."""
        disable_reads = []
        dtypes = {}
        converters = {}
        kwargs = {}
        decoders = {}
        for order in self.orders:
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                disable_reads.append(order)
                continue
            sections = self.schema["sections"][order]["elements"]
            for section, section_dict in sections.items():
                index = self._get_index(section, order)
                ignore = self._get_ignore(section_dict)
                if ignore is True:
                    continue
                dtypes = self._update_dtypes(section_dict, dtypes, index)
                converters = self._update_converters(section_dict, converters, index)
                kwargs = self._update_kwargs(section_dict, kwargs, index)
                decoders = self._update_decoders(section_dict, decoders, index)

        dtypes, parse_dates = convert_dtypes(dtypes)
        return {
            "convert_decode": {
                "converter_dict": converters,
                "converter_kwargs": kwargs,
                "decoder_dict": decoders,
                "dtype": dtypes,
            },
            "self": {
                "dtypes": dtypes,
                "disable_reads": disable_reads,
                "parse_dates": parse_dates,
            },
        }

    def open_pandas(self):
        """Open TextParser to pd.DataSeries."""
        return self.df.apply(lambda x: self._read_line(x[0]), axis=1)

    def open_netcdf(self):
        """Open netCDF to pd.Series."""

        def replace_empty_strings(series):
            if series.dtype == "object":
                series = series.str.decode("utf-8")
                series = series.str.strip().replace("", None)
            return series

        missing_values = []
        attrs = {}
        renames = {}
        disables = []
        for order in self.orders:
            header = self.schema["sections"][order]["header"]
            disable_read = header.get("disable_read")
            if disable_read is True:
                disables.append(order)
                continue
            sections = self.schema["sections"][order]["elements"]
            for section, section_dict in sections.items():
                index = self._get_index(section, order)
                ignore = (order not in self.valid) or self._get_ignore(section_dict)
                if ignore is True:
                    continue
                if section in self.df.data_vars:
                    renames[section] = index
                elif section in self.df.dims:
                    renames[section] = index
                elif section in self.df.attrs:
                    attrs[index] = self.df.attrs[index]
                else:
                    missing_values.append(index)

        df = self.df[renames.keys()].to_dataframe().reset_index()
        attrs = {k: v.replace("\n", "; ") for k, v in attrs.items()}
        df = df.rename(columns=renames)
        df = df.assign(**attrs)
        for column in disables:
            df[column] = np.nan
        df = df.apply(lambda x: replace_empty_strings(x))
        df["missing_values"] = [missing_values] * len(df)
        return df
