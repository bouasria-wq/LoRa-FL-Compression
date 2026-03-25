# hardwaretest04 - ME-CFL with Real USRP B200 Hardware

## What's Different

| Version | LoRa Transport |
|---------|---------------|
| onlinetest02 | Simulated (math only, no GNU Radio) |
| hardwaretest03 | EPFL tx_rx_simulation.grc (Channel Model block) |
| **hardwaretest04** | **EPFL lora_TX.grc + lora_RX.grc (REAL USRP B200 over the air)** |

## Hardware Required

- 2x Ettus USRP B200 connected via USB 3.0
- 2x LoRa antennas (868 MHz) attached to TX/RX ports
- Lab PC with USB 3.0 ports

### Your USRP Serial Numbers
- TX USRP: `32BBAD0` (home transmitter)
- RX USRP: `32BBA90` (server receiver)

Verify with: `uhd_find_devices`

## Architecture

```
Python (training/compression) --> File I/O --> EPFL lora_TX.grc --> USRP B200 TX
                                                                       |
                                                                   [OVER THE AIR]
                                                                       |
Python (aggregation/decode)   <-- stdout  <-- EPFL lora_RX.grc <-- USRP B200 RX
```

The EPFL flowgraphs are 100% untouched. We only change:
1. The input file path (data, not LoRa config)
2. The USRP device args (serial number instead of IP address)

## How to Run

### Terminal 1 — Server:
```bash
cd ~/LoRa_Fl_Compression.proj/hardwaretest04
python3 server_aggregator.py --n_homes 1 --days 7
```

### Terminal 2 — Home node:
```bash
cd ~/LoRa_Fl_Compression.proj/hardwaretest04
python3 home_node.py --home_id 1 --days 7 --epochs 100
```

### Custom serial numbers:
```bash
python3 home_node.py --home_id 1 --days 7 --epochs 100 \
    --tx_serial 32BBAD0 --rx_serial 32BBA90

python3 server_aggregator.py --n_homes 1 --days 7 \
    --tx_serial 32BBA90 --rx_serial 32BBAD0
```

## Files

```
hardwaretest04/
├── home_node.py              # Home node (trains + sends via USRP)
├── server_aggregator.py      # Server (aggregates + broadcasts via USRP)
├── compression/
│   ├── hegazy.py             # Hegazy compression with error feedback
│   └── hegazy_lora_bridge.py # struct.pack binary packing + ASCII
├── local_home/
│   ├── load_data.py          # Data loading (96 samples/day)
│   ├── model.py              # LSTM model (553 params, H=8, d=8)
│   └── train.py              # Local training
├── server/
│   └── aggregate.py          # ME-CFL aggregation
├── lora/
│   └── gr_lora_usrp.py       # Bridge: writes payload, compiles+runs EPFL .grc
└── data/
    └── home_01.csv ... home_10.csv
```

## Requirements

```bash
# On the lab PC:
sudo apt install gnuradio python3-uhd
# gr-lora_sdr must be built from source (already done)
pip install tensorflow pandas numpy scipy
```
