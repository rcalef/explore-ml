#! /usr/bin/env python3

from itertools import product

def general_range(start,stop,step):
    val = start
    while val <=stop:
        yield val
        val += step

class BaseParam(object):
    def __init__(self,name):
        self.name = name

class ValueParam(BaseParam):  
    def __init__(self,name,start,stop,step):
        super().__init__(name)
        self.start = start
        self.stop = stop
        self.step = step

    def __iter__(self):
        return general_range(self.start,self.stop,self.step)

class StateParam(BaseParam):
    def __init__(self,name,states):
        super().__init__(name)
        self.states = states

    def __iter__(self):
        return iter(self.states)

class TupleParam(BaseParam):
    def __init__(self,name,sizes,values):
        super().__init__(name)
        self.sizes = sizes
        self.values = values

    @classmethod
    def from_lists(cls,name,sizes,values):
        return cls(name,sizes,values)

    @classmethod    
    def from_ranges(cls,
                    name,
                    size_start,
                    size_stop,
                    size_step,
                    value_start,
                    value_stop,
                    value_step):
        sizes = list(general_range(size_start,size_stop,size_step))
        values = list(general_range(value_start,value_stop,value_step))
        return cls(name,sizes,values)

    def generate_values(self):
        for size in self.sizes:
            for tuple_value in product(self.values,repeat=size):
                yield tuple_value

    def __iter__(self):
        return self.generate_values()

example_params = {
    "state_param":{
        "type" : "state",
        "states" : ["foo","bar","baz"]
    },
    "value_param":{
        "type" : "value",
        "start" : 0,
        "step" : 1,
        "stop" : 2
    },
    "tuple_param":{
        "type" : "tuple",
        "sizes" : [0,1,2],
        "values" : [5,7]
    },
    "tuple_param2":{
        "type" : "tuple",
        "size_start" : 1,
        "size_end" : 2,
        "size_step" : 1,
        "value_start" : -1,
        "value_stop" : 0,
        "value_step" : 1
    }
}

def generate_params(param_dict):
    params = []
    for pname in param_dict:
        param = param_dict[pname]
        ptype = param["type"]
        if ptype == "state":
            params.append(StateParam(pname,
                                     param["states"]))
        elif ptype == "value":
            params.append(ValueParam(pname,
                                     param["start"],
                                     param["stop"],
                                     param["step"]))
        elif ptype == "tuple":
            if "size_start" in param:
                params.append(TupleParam.from_ranges(pname,
                                                     param["size_start"],
                                                     param["size_stop"],
                                                     param["size_step"],
                                                     param["value_start"],
                                                     param["value_stop"],
                                                     param["value_step"]))
            else:
                params.append(TupleParam.from_lists(pname,
                                                    param["sizes"],
                                                    param["values"]))
        else:
            print("Invalid parameter type: %s" % (param_dict))
    for param_set in product(*params):
        yield {p.name : value for p,value in zip(params,param_set)}
