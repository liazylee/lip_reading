go to [lip reading](Lip%20Reading.md)

# Install

```bash
python3 -m venv venv
pip install -r requirements.txt
```

# Run

```bash
cd src/pytorch_lipNet
```

make sure you data directory is in the config.py file
extract the mouth frames from the video

```bash
python pretrain.py
```

train the model

```bash
python main.py
```

# change the parameters in the config.py file

