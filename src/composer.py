import torch
import numpy as np

def song_to_tensor(notes, durations, middle_c, sample_rate=44100):
  """
  converts a song to a torch Tensor which can be written to 
  Tensorboard

  notes - iterable of notes, indexed either as half steps from
          middle_c, or from the lowest note on the piano
  durations - iterable of durations, in seconds
  middle_c - bool, reference from middle_c?  
  sample_rate - sample_rate of TensorBoard (no need to change this)
  """

  full_tensor = []
  for note, duration in zip(notes, durations):
    note_tensor = note_to_tensor(note, duration, middle_c, sample_rate)
    full_tensor.append(note_tensor)
  return torch.cat(tuple(full_tensor), 0)

def note_to_tensor(note, duration, middle_c, sample_rate=44100):
  """
  converts a note to a torch Tensor which can be written to
  Tensorboard

  note - int, note number, indexed either as half steps from
          middle_c, or from the lowest note on the piano
  duration - float, seconds
  middle_c - bool, reference from middle_c?
  sample_rate - sample_rate of TensorBoard (no need to change this)
  """
  frequency = piano_frequency(note, middle_c)
  n_pts = duration * sample_rate
  return torch.Tensor(np.cos(np.arange(n_pts) * frequency / sample_rate * 2 * np.pi))

def piano_frequency(note_num, middle_c=True):
  if not middle_c:
    return np.power(2, (note_num - 49.)/12) * 440
  else:
    return np.power(2, (note_num - 9.)/12) * 440


