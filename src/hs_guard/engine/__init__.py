__all__ = ["SlotStreamEngineV8"]

def __getattr__(name):
    if name == "SlotStreamEngineV8":
        from .slot_engine_v8 import SlotStreamEngineV8
        return SlotStreamEngineV8
    raise AttributeError(name)
