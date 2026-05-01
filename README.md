# Dependencies
## PyAudio
sudo apt install -y portaudio19-dev

## Picamera
sudo apt instll -y python3-picamera2 libcamera-apps

## Grove
Add repository
```
echo "deb https://seeed-studio.github.io/pi_repo/ stretch main" | sudo tee /etc/apt/sources.list.d/seeed.list
```

Add GPG key
```
curl https://seeed-studio.github.io/pi_repo/public.key | sudo apt-key add -
sudo apt update
sudo apt install libbmi088 libbma456
```

# Pi config
Enable I2C interface
```
sudo raspi-config
```
Select interfacingg Options>I2C>Yes>Ok>Finish
Enable I2C interface

# Setting up venv
```
uv venv --system-site-packages
```

# Caputring data
```
python collect_training_data.py <gsr_adc_channel> [--label <label>] [--model <path>]
```
--model  Path to a trained model
--combine-mode 'any' | 'all' | 'mean' (default: 'any').
Sessions output to to ./sessions/<timestamp>_<label>/

# Running model
Look at run_ensemble.sh

# Clip Viewer
```
python app.py [--sessions <path>] [--port <port>]

```
--sessions   Root directory containing session folders (default: ../sessions)
--port       Port to serve on (default: 5000)

Then open http://localhost:5000 in your browser.
