# NMR_DL
This repository demonstrate a Graph Neural Network (GNN) aimed to predict Nuclear Magnetic Resonance (NMR) chemical shift,
a spectroscopy chemical technique used to determine molecular structure.
The GNN architecture is based on Residual Gated Graph ConvNets (https://arxiv.org/pdf/1711.07553v2.pdf, https://github.com/xbresson/spatial_graph_convnets/tree/master)
The Data is taken from NMRshiftDB (https://nmrshiftdb.nmr.uni-koeln.de/)

## Installation
```bash
git clone https://github.com/tsoofbaror/NMR_DL
```

## Usage
```python
net_parameters = get_model_hyperparameters()
model_runner = ModelRunner(net_parameters=net_parameters)
model_runner.train_model()
trained_model = model_runner.model
train_dataloader = model_runner.train_dataloader
test_dataloader = model_runner.test_dataloader 
```

## Authors
Tsoof Bar-or and Itamar Wallwater

## Contributing
Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
