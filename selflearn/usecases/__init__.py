"""Registry of use cases. Adding one = one new module + one line here; nothing else changes."""
from selflearn.usecase import UseCase
from selflearn.usecases import math, support

USE_CASES: dict[str, UseCase] = {"math": math.USE_CASE, "support": support.USE_CASE}

# Earlier versions of an environment, for the audit's look back at runs that trained in them (not for new training).
HISTORY: dict[str, UseCase] = {"support-v1": support.USE_CASE_V1}
