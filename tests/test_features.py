import numpy as np
import pandas as pd

from classification_model.processing.features import ExtractLetterTransformer


def test_extract_letter_transformer():
    sample_data = {"cabin": ["E12", "A12", np.nan]}
    sample_df = pd.DataFrame(sample_data)
    transformer = ExtractLetterTransformer(variables=["cabin"])

    result = transformer.fit_transform(sample_df)

    expected_result = pd.DataFrame({"cabin": ["E", "A", np.nan]})
    pd.testing.assert_frame_equal(result, expected_result)
