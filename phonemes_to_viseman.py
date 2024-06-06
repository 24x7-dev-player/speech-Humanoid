viseme_map = {
  "aa": "open_mouth",
  "ih": "smile",
  "eh": "neutral",
  "uh": "frown",
  # Add more visemes and mappings here
}

def map_phonemes_to_visemes(phonemes):
  visemes = []
  for phoneme in phonemes:
    viseme = viseme_map.get(phoneme, "neutral")
    visemes.append(viseme)
  return visemes
