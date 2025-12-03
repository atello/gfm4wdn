
#
# Created on Tue Feb 13 2024
# Copyright (c) 2024 Huy Truong
# ------------------------------
# Purpose: Store properties of components in a WDN  
# ------------------------------
#
from dataclasses import dataclass


NODE_PROPS :list[str] = [
    'name',
    'node_type',
    'coordinates',
    'initial_quality',
    'tag',
    'head',
    'demand',
    'leak_demand',
    'leak_status',
    'leak_area',
    'leak_discharge_coeff',
    'pressure',
    'quality',
]

JUNCTION_PROPS :list[str] = NODE_PROPS + [
    'base_demand',
    'demand_timeseries_list',
    'elevation',
    'emitter_coefficient',
    'minimum_pressure',
    'required_pressure',
    'pressure_exponent',
]

TANK_PROPS  :list[str] = NODE_PROPS +[
    'elevation',
    'init_level',
    'min_level',
    'max_level',
    'diameter',
    'min_vol',
    'vol_curve_name',
    'vol_curve',
    'overflow',
    'mixing_model',
    'mixing_fraction',
    'bulk_coeff',
    'level',
]

RESERVOIR_PROPS  :list[str] = NODE_PROPS +[
    'base_head',
    'head_pattern_name',
    'head_timeseries',
]

LINK_PROPS :list[str] =[
    'name',
    'link_type',
    'start_node_name',
    'end_node_name',
    'initial_status',
    'initial_setting',
    'tag',
    'vertices',
    'flow',
    'headloss',
    'quality',
    'status',
    'setting',
]

PIPE_PROPS  :list[str] = LINK_PROPS + [
    'length',
    'diameter',
    'roughness',
    'minor_loss',
    'cv',
    'bulk_coeff',
    'wall_coeff',
    'velocity',
    'friction_factor',
    'reaction_rate',
]

PUMP_PROPS :list[str]  = LINK_PROPS +[
    'base_speed',
    'speed_pattern_name',
    'speed_timeseries',
    'efficiency',
    'energy_price',
    'energy_pattern',
    'velocity',
]

VALVE_PROPS  :list[str] = LINK_PROPS +[
    'valve_type',
    'velocity',
]

PRV_PROPS  :list[str] = VALVE_PROPS
PSV_PROPS  :list[str] = VALVE_PROPS
PBV_PROPS  :list[str] = VALVE_PROPS
FCV_PROPS  :list[str] = VALVE_PROPS
TCV_PROPS  :list[str] = VALVE_PROPS
GPV_PROPS  :list[str] = VALVE_PROPS


@dataclass
class NodeProperties:
    name:str|None  = None
    node_type:str|None  = None
    coordinates:tuple[float]|None  = None
    initial_quality:float|None  = None
    tag:str|None  = None
    head:float|None  = None
    demand:float|None  = None
    leak_demand:float|None  = None
    leak_status:float|None  = None
    leak_area:float|None  = None
    leak_discharge_coeff:float|None  = None
    pressure:float|None  = None
    quality:float|None  = None

@dataclass
class JunctionProperties(NodeProperties):
    base_demand:float|None  = None
    demand_timeseries_list:list[float]|None  = None
    elevation:float|None  = None
    emitter_coefficient:float|None  = None
    minimum_pressure:float |None = None
    required_pressure:float|None  = None
    pressure_exponent:float |None = None


@dataclass
class TankProperties(NodeProperties):
    elevation:float|None  = None
    init_level:float |None = None
    min_level:float |None = None
    max_level:float|None  = None
    diameter:float |None = None
    min_vol:float|None  = None
    vol_curve_name:float |None = None
    vol_curve:float|None  = None
    overflow:bool |None = None
    mixing_model:float |None = None
    mixing_fraction:float |None = None
    bulk_coeff:float |None = None
    level:float|None =None

@dataclass
class ReservoirProperties(NodeProperties):
    base_head:float|None  = None
    head_pattern_name:str |None = None
    head_timeseries:list|None  = None

@dataclass
class LinkProperties:
    name:str|None  = None
    link_type:str |None = None
    start_node_name:str |None = None
    end_node_name:str |None = None
    initial_status: bool|None = None
    initial_setting: float |None = None
    tag:str |None = None
    vertices:list |None = None

    flow:float |None = None
    headloss:float |None = None
    quality:float |None = None
    status:bool |None = None
    setting:float |None = None
    
@dataclass
class PipeProperties(LinkProperties):
    length: int|None = None
    diameter: float |None = None
    roughness:float |None = None
    minor_loss:list |None = None
    cv:bool |None = None
    bulk_coeff:float |None = None
    wall_coeff:float|None  = None

    velocity:float|None  = None
    friction_factor:float |None = None
    reaction_rate:float|None  = None

@dataclass
class PumpProperties(LinkProperties):
    base_speed:float |None = None
    speed_pattern_name:str |None = None
    speed_timeseries:list |None = None
    efficiency:float|None  = None
    energy_price:float |None = None
    energy_pattern:str |None = None

    velocity:float|None  = None


@dataclass
class ValveProperties(LinkProperties):
    valve_type:int |None = None
    velocity:float |None = None

@dataclass
class PRValveProperties(ValveProperties):
    pass
@dataclass
class PSValveProperties(ValveProperties):
    pass
@dataclass
class PBValveProperties(ValveProperties):
    pass
@dataclass
class FCValveProperties(ValveProperties):
    pass
@dataclass
class TCValveProperties(ValveProperties):
    pass
@dataclass
class GPValveProperties(ValveProperties):
    pass