from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleHouseDataInputs(
            inputs=data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return data, errors


class TitanicDatainputschema(BaseModel):
    PClass: Optional[int]
    Sex: Optional[str]
    Age: Optional[float]
    Sibsp: Optional[int]
    Parch: Optional[int]
    Fare: Optional[float]
    Cabin: Optional[str]
    embarked: Optional[str]
    title: Optional[str]


class MultipleHouseDataInputs(BaseModel):
    inputs: List[TitanicDatainputschema]
