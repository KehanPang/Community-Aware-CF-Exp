from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionSpec:
    kind: str
    num_classes: int
    threshold: float = 0.5
    positive_class: int = 1

    def __post_init__(self):
        if self.kind not in {'argmax_change', 'positive_to_negative'}:
            raise ValueError('Unknown decision event')
        if self.num_classes < 2 or not 0 < self.threshold < 1:
            raise ValueError('Invalid decision specification')
        if self.kind == 'positive_to_negative' and self.num_classes != 2:
            raise ValueError('Binary decision required')

    def prediction(self, probabilities):
        if self.kind == 'argmax_change':
            return probabilities.argmax(-1)
        return (probabilities[..., self.positive_class] >= self.threshold).long()

    def event(self, original_probabilities, intervened_probabilities):
        before = self.prediction(original_probabilities)
        after = self.prediction(intervened_probabilities)
        return before.ne(after) if self.kind == 'argmax_change' else before.eq(1) & after.eq(0)
