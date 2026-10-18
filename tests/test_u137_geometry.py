import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from diagnose_u137_geometry import transform
from scoring import box_iou


def test_fixed_native_contract_maps_known_rectangle():
    item={'width':640,'height':480,'processor_width':704,'processor_height':512}
    box=transform([100,200,400,600],item,'registered_native1000')
    assert box==[64,96,256,288]
    assert box_iou(box,[64,96,256,288])==1


def test_pixel_hypotheses_are_global_maps_without_label_input():
    item={'width':640,'height':480,'processor_width':1280,'processor_height':960}
    assert transform([64,96,256,288],item,'hypothetical_original_pixels')==[64,96,256,288]
    assert transform([128,192,512,576],item,'hypothetical_processor_pixels')==[64,96,256,288]


def test_unparseable_output_is_not_rescued_or_fitted():
    assert transform(None,{},'registered_native1000') is None


def test_unknown_mapping_is_rejected():
    with pytest.raises(ValueError):
        transform([1,2,3,4],{},'choose_best')
