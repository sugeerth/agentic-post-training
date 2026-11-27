"""Registry of use cases. Adding one = one new module + one line here; nothing else changes."""
from selflearn.usecase import UseCase
from selflearn.usecases import math, support

USE_CASES: dict[str, UseCase] = {"math": math.USE_CASE, "support": support.USE_CASE}
