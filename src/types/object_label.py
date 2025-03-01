class ObjectLabel(str):
    def __new__(cls, value):
        if ':' not in value:
            raise ValueError("ObjectName must include a colon ':'")
        return super().__new__(cls, value)

    def __init__(self, value):
        super().__init__()
        self._name, self._type = value.split(':', 1)

    @property
    def name(self):
        return self._name

    @property
    def type(self):
        return self._type

    def __repr__(self):
        return f"ObjectLabel(name='{self.name}', _type='{self._type}')"
