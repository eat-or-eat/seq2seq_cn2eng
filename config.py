config = {
    'model_path': './output/model',
    'data_path': './data/cmn.txt',
    'vocab_path': './data/vocab.txt',
    'input_max_length': 30,
    'output_max_length': 30,
    'batch_size': 350,
    'beam_size': 4,
    'optimizer': 'adam',
    'vocab_size': 11222,
    'epoch': 200,
    'learning_rate': 1e-3,
    'pad_idx': 0,
    'start_idx': 1,
    'end_idx': 2
}