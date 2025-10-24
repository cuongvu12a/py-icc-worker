import os
from typing import Dict, Any, List

class ActionCommand:
    def __init__(self, action_type: str, params: Any):
        self.action_type = action_type
        self.params = params

    def execute(self, processor):
        if not hasattr(processor, self.action_type):
            raise AttributeError(f"Processor has no action '{self.action_type}'")
        method = getattr(processor, self.action_type)

        # Nếu params là dict → dùng **kwargs
        if isinstance(self.params, dict):
            return method(**self.params)
        # Nếu params là 1 giá trị đơn (int, str) → truyền trực tiếp
        else:
            return method(self.params)


class PartialPipeline:
    """
    Một partial với steps và location
    """
    def __init__(self, partial_id: str, steps: List[ActionCommand], location: Dict[str, int]):
        self.id = partial_id
        self.steps = steps
        self.location = location

    @classmethod
    def from_json(cls, partial_json: Dict, asset_dir: str = "") -> 'PartialPipeline':
        steps = []
        for step in partial_json.get("steps", []):
            action = step.get("action")
            data = step.get("data")
            if action == 'mask':
                action = 'erase_by_mask'
                data = os.path.join(asset_dir, data)

            steps.append(ActionCommand(action, data))
        location = partial_json.get("location", {"top":0, "left":0})
        return cls(partial_json.get("id"), steps, location)
