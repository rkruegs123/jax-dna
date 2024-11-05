
from typing import Any

import jax
import jax_md
import ray


jax.config.update("jax_enable_x64", True)


def build_actor():
    pass







class Simulator:

    def __init__(self, f:callable):
        self.f = None

    def run(self, opt_parms:dict[str, Any]):
        return self.f(opt_parms)




