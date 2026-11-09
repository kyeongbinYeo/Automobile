#!/usr/bin/env python3
"""Mark an ONNX file with the backend it belongs to.

LineNet and TwinLiteNet exports have the same input and output shapes but different input
scaling, so the wrong file would run without error and give wrong masks. The backends check this
tag when they load.   python onnx_tag.py weights/linenet.onnx linenet
"""
import sys

import onnx

KINDS = ("linenet", "twinlite")


def tag(path, kind):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    model = onnx.load(path)
    for p in [p for p in model.metadata_props if p.key == "backend"]:
        model.metadata_props.remove(p)
    model.metadata_props.add(key="backend", value=kind)
    onnx.save(model, path)


if __name__ == "__main__":
    tag(sys.argv[1], sys.argv[2])
    print("tagged", sys.argv[1], "as", sys.argv[2])
