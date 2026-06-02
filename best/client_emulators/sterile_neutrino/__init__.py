from importlib.resources import files

def path():
    """Return path to sterile neutrino emulator directory."""
    return files("best.client_emulators.sterile_neutrino")
