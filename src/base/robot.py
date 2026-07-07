import json


class Robot:
    def __init__(self):
        pass

    def get_command(self, command):
        command = self.parse_command(command)
        self.apply_command(command)

    def parse_command(self, command: str) -> list:
        try:
            command = json.loads(command)
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSONDecodeError: {e} | Raw input: {command}")
            return []

        return command

    def apply_command(self, command: list):
        raise NotImplementedError("Subclasses must implement this method.")
