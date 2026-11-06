every preset builds on ../config.toml (your model and key setup) and changes
only what the experiment needs. run one with:

  python sim.py --config presets/<name>.toml --view
  python eval.py                                   # then score it

try any of them for free first with --agent echo. remove the mystery from any
preset with --set remove_mystery.task=true (and peers, grid, text, ...).
