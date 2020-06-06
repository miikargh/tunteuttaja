from typing import List, Tuple

import numpy as np
from transformers import Pipeline


class TunteuttajaPipeline(Pipeline):
    def __call__(self, *args, **kwargs) -> List[Tuple[int, float]]:
        """ Returns a list of tuples. The first element of the tuple
            corresponds to the label index and the second to the score.
        """
        outputs = super().__call__(*args, **kwargs)
        scores = np.exp(outputs) / np.exp(outputs).sum(-1, keepdims=True)

        return [
            {"label": i.item(), "score": scores[0][i].item()}
            for i in np.argsort(scores[0])
        ][::-1]
