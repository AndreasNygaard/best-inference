from importlib.resources import files

def path():
    """Return path to LCDM emulator directory."""
    return files("best.client_emulators.lcdm")
