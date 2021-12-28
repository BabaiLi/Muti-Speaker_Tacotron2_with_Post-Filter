""" from https://github.com/keithito/tacotron """

import re

valid_symbols = ['a1', 'a2', 'a3', 'a4', 'a5', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 
    'n', 'ng', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'b', 'ei1', 'ei2', 'ei3', 'ei4', 
    'e1', 'e2', 'e3', 'e4', 'e5', 'i1', 'i2', 'i3', 'i4', 'i5', 'ia1', 'ia2', 'ia3', 
    'ia4', 'ia5', 'iao1', 'iao2', 'iao3', 'iao4', 'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 
    'o1', 'o2', 'o3', 'o4', 'o5', 'u1', 'u2', 'u3', 'u4', 'u5', 'c', 'ch', 'ii1', 'ii2', 
    'ii3', 'ii4', 'ii5', 'ou1', 'ou2', 'ou3', 'ou4', 'ou5', 'uai1', 'uai2', 'uai3', 'uai4', 
    'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'uei1', 'uei2', 'uei3', 'uei4', 'uei5', 'ue1', 'ue2', 
    'ue3', 'ue4', 'ue5', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'd', 'iou1', 'iou2', 'iou3', 'iou4', 
    'r', 'f', 'g', 'h', 'j', 'io1', 'io2', 'io3', 'io4', 'v1', 'v2', 'v3', 'v4', 'v5', 'va1', 'va2', 
    'va3', 'va4', 've1', 've2', 've3', 've4', 'k', 'l', 'm', 'p', 'q', 's', 'sh', 't', 'x', 'z', 'zh']

_valid_symbol_set = set(valid_symbols)


class CMUDict:
  '''Thin wrapper around CMUDict data. http://www.speech.cs.cmu.edu/cgi-bin/cmudict'''
  def __init__(self, file_or_path, keep_ambiguous=True):
    if isinstance(file_or_path, str):
      with open(file_or_path, encoding='latin-1') as f:
        entries = _parse_cmudict(f)
    else:
      entries = _parse_cmudict(file_or_path)
    if not keep_ambiguous:
      entries = {word: pron for word, pron in entries.items() if len(pron) == 1}
    self._entries = entries


  def __len__(self):
    return len(self._entries)


  def lookup(self, word):
    '''Returns list of ARPAbet pronunciations of the given word.'''
    return self._entries.get(word.lower())



_alt_re = re.compile(r'\([0-9]+\)')


def _parse_cmudict(file):
  cmudict = {}
  for line in file:
    if len(line) and (line[0] >= 'a' and line[0] <= 'z' or line[0] == "'"):
      parts = line.split('  ')
      word = re.sub(_alt_re, '', parts[0])
      pronunciation = _get_pronunciation(parts[1])
      if pronunciation:
        if word in cmudict:
          cmudict[word].append(pronunciation)
        else:
          cmudict[word] = [pronunciation]
  return cmudict


def _get_pronunciation(s):
  parts = s.strip().split(' ')
  for part in parts:
    if part not in _valid_symbol_set:
      return None
  return ' '.join(parts)
