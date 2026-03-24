# Clone repo
```
git clone https://github.com/SniperSlayerss/emotion-recap.git
git submodule update --init --recursive
```
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
