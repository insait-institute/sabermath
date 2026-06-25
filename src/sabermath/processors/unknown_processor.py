from .base import ModelProcessor


class UnknownProcessor(ModelProcessor):
    processor = None

    def __init__(self, model):
        has_encode = callable(getattr(model, "encode", None))
        has_get_scores = callable(getattr(model, "get_scores", None))

        if not has_encode and not has_get_scores:
            raise ValueError(
                "Input model must have a callable .encode() or .get_scores()"
            )

        if has_encode:
            self.encode = model.encode
        else:

            def encode_not_implemented(*args, **kwargs):
                raise NotImplementedError("This model does not support .encode()")

            self.encode = encode_not_implemented
            self.get_scores = model.get_scores
