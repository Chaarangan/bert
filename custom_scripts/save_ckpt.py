import tensorflow as tf


def print_checkpoint(save_path):
  reader = tf.train.load_checkpoint(save_path)
  shapes = reader.get_variable_to_shape_map()
  dtypes = reader.get_variable_to_dtype_map()
  print(f"Checkpoint at '{save_path}':")
  for key in shapes:
    print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
          f"value={reader.get_tensor(key)})")



def convert_tf1_to_tf2(checkpoint_path, output_prefix):
    """Converts a TF1 checkpoint to TF2.

    To load the converted checkpoint, you must build a dictionary that maps
    variable names to variable objects.
    ```
    ckpt = tf.train.Checkpoint(vars={name: variable})
    ckpt.restore(converted_ckpt_path)

        ```

        Args:
        checkpoint_path: Path to the TF1 checkpoint.
        output_prefix: Path prefix to the converted checkpoint.

        Returns:
        Path to the converted checkpoint.
        """
    vars = {}
    reader = tf.train.load_checkpoint(checkpoint_path)
    dtypes = reader.get_variable_to_dtype_map()
    for key in dtypes.keys():
        vars[key] = tf.Variable(reader.get_tensor(key))
    
    return tf.train.Checkpoint(vars=vars).save(output_prefix)


# Make sure to run the snippet in `Save a TF1 checkpoint in TF2`.
print_checkpoint(
    '/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/Bio_ClinicalBERT/checkpoints/model.ckpt-150000.index')
converted_path = convert_tf1_to_tf2('/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/Bio_ClinicalBERT/checkpoints/model.ckpt-150000.index',
                                    '/Users/charangan/Desktop/Intern/NER/pretraining/bert/pretrained_models/Bio_ClinicalBERT/checkpoints/converted-tf1-to-tf2/converted-tf1-to-tf2')
print("\n[Converted]")
print_checkpoint(converted_path)
