# Play songs using TensorBoard

It turns out TensorBoard has an audio feature.  Any waveform can be written to TensorBoard, and then played.  While this is commonly used to evaluate generative audio models or examine training data for speech classification systems, we can also directly write a song directly and have TensorBoard play this for us.

## Example: Happy birthday

The melody for Happy birthday is (starting from C) 
```C, C, D, C, F, E, C, C, D, C, G, F, C, C, C^, A, F, E, D, F, F, E, C, D, C```.
Written in half steps from `C`, this is
```0, 0, 2, 0, 5, 4, 0, 0, 2, 0, 7, 5, 0, 0, 12, 9, 5, 4, 2, 5, 5, 4, 0, 2, 0```

with corresponding durations
```3, 1, 4, 4, 4, 8, 3, 1, 4, 4, 4, 8, 3, 1, 4, 4,  4, 4, 8, 3, 1, 4, 4, 4, 8```

To make the song into a tensor, run the following:
```
In [1]: from src import composer

In [2]: notes = [0, 0, 2, 0, 5, 4, 0, 0, 2, 0, 7, 5, 0, 0, 12, 9, 5, 4, 2, 5, 5, 4, 0, 2, 0]

In [3]: durations = [3, 1, 4, 4, 4, 8, 3, 1, 4, 4, 4, 8, 3, 1, 4, 4,  4, 4, 8, 3, 1, 4, 4, 4, 8]

In [4]: song = composer.song_to_tensor(notes, durations, middle_c=True)

```
Now, write the tensor to a tensorboard-readable file
```
In [6]: from torch.utils.tensorboard import SummaryWriter

In [7]: writer = SummaryWriter()

In [8]: writer.add_audio('Happy_Birthday', song)
```

Then point tensorboard to the directory containing the file, and you're good to go.

