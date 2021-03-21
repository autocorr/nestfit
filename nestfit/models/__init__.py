#!/usr/bin/env python3

from nestfit.models import (ammonia, diazenylium, gaussian)


MODEL_MODULES = [ammonia, diazenylium, gaussian]
MODELS = {m.NAME: m for m in MODEL_MODULES}


